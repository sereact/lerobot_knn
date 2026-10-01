"""Is it the backbone or the action expert that keys on the wording? Original vs reworded LIBERO-plus instructions.

LIBERO-plus rewrites every LIBERO instruction about 38 ways (``outputs/libero_plus/language_variants.json``, exported
from the fork). Clean demo frames of each task are paired with the original and with a random rewording of their
instruction; nothing else changes. Reported per model:

* **backbone** (a plain lookup's key on the VLA's VLM):
  * ``key_shift``: median distance between the two keys of a frame / the median distance of a clean frame to the
    nearest frame of another demo (how far rewording moves the key, in units of the gap between demos)
  * ``task_acc_original`` / ``task_acc_reworded``: how often the nearest memory frame, searched over *all* 40 tasks
    (no task filter), belongs to the frame's own task
* **action expert** (the VLA itself): the motion MSE between its chunks at the same frame for
  ``same`` (a second sample, same prompt: sampling noise), ``reworded``, and ``other_task`` (another task's
  original instruction: a real task change). ``rewording_effect`` = (reworded - same) / (other_task - same):
  0 = the wording does not matter, 1 = a rewording changes the action as much as switching the task.

    uv run python -m lerobot_policy_knn.experiments.plus_language
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE

from ..vla import DATASET_CAMERAS, load_vla
from .head_vs_knn import MOTION, embed_records, space_config
from .search import MemoryFactory

OUT = Path("outputs/libero_plus")
BACKBONES = [
    "resnet18_l3",  # text-free control: the key cannot move
    "vla:smolvla_base/grid2",
    "vla:smolvla_libero/grid2",
    "vla:pi05_base/grid2@L12|zscore",
    "vla:pi05_libero/grid2@L12|zscore",
    "vla:pi0_base/grid2@L12|zscore",
    "vla:pi0_libero/grid2@L12|zscore",
]
VLAS = [
    "pi05_libero",
    "smolvla_libero",
    "pi0_libero",
    "pi0_fast@lerobot/pi0fast-libero",
    "vla_jepa@lerobot/VLA-JEPA-LIBERO",
]
HORIZON = 10


def extra_keys() -> list[dict]:
    """Backbone keys added after the layer pilots (outputs/vla/extra_keys.json, written by choose_keys)."""
    path = Path("outputs/vla/extra_keys.json")
    return json.loads(path.read_text())["keys"] if path.exists() else []


BACKBONES += [
    f"{k['encoder']}/{k['pooling']}" + (f"|{k['norm']}" if k["norm"] != "none" else "") for k in extra_keys()
]


def load_frames(
    factory: MemoryFactory, per_task: int, seed: int = 0
) -> tuple[dict, list[str], list[str], list[str]]:
    """Clean demo frames (``rec`` as ``embed_records`` expects) with the original, a reworded and another task's prompt."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    frames = factory.frames
    variants = json.loads((OUT / "language_variants.json").read_text())
    reworded: dict[str, list[str]] = {}
    for v in variants:
        reworded.setdefault(v["original"], []).append(v["reworded"])
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(len(frames.task_names)):
        cand = np.nonzero((frames.task_index == t) & (frames.frame_index % 10 == 4))[0]
        rows += rng.choice(cand, per_task, replace=False).tolist()
    rows = np.sort(np.array(rows))
    ds = LeRobotDataset("lerobot/libero")
    images = np.zeros((len(rows), 2, 3, 256, 256), dtype=np.uint8)
    state = np.zeros((len(rows), 8), dtype=np.float32)
    for i, r in enumerate(rows):
        item = ds[int(r)]
        for j, cam in enumerate(DATASET_CAMERAS):
            images[i, j] = (item[cam].clamp(0, 1) * 255).round().byte().numpy()
        state[i] = item[OBS_STATE].numpy()
    original = [frames.task_names[int(t)] for t in frames.task_index[rows]]
    rew = [str(rng.choice(reworded[o])) for o in original]
    other = [str(rng.choice([n for n in frames.task_names if n != o])) for o in original]
    rec = {"images": images, "state": state, "t": frames.frame_index[rows], "rows": rows,
           "task_index": frames.task_index[rows], "episode": frames.episode_index[rows]}  # fmt: skip
    return rec, original, rew, other


def with_prompts(rec: dict, prompts: list[str]) -> dict:
    names = sorted(set(prompts))
    return {**rec, "task_names": np.array(names), "task": np.array([names.index(p) for p in prompts])}


@torch.no_grad()
def backbone(space: str, factory: MemoryFactory, rec: dict, original: list[str], reworded: list[str]) -> dict:
    cfg = space_config(space, 1, HORIZON)
    cfg.task_conditioning = "none"  # search all 40 tasks: does the key alone still point at the right task?
    policy = factory.build(cfg)
    k_o = embed_records(policy, with_prompts(rec, original))
    k_r = embed_records(policy, with_prompts(rec, reworded))
    t = torch.from_numpy(rec["t"]).cuda()
    own = torch.from_numpy(rec["task_index"]).cuda()
    ep = torch.from_numpy(rec["episode"]).cuda()
    task_of_row = torch.from_numpy(factory.frames.task_index).cuda()
    out = {}
    for tag, k in (("original", k_o), ("reworded", k_r)):
        rows, _, d = policy.retrieve(k, t, None, exclude_episode=ep)
        top_task = task_of_row[policy.key_ptr[rows[:, 0]]]
        out[f"task_acc_{tag}"] = float((top_task == own).float().mean())
        if tag == "original":
            gap = d[:, 0].median().clamp_min(1e-9)
    # the policy's own distance between the two keys of the same frame (mean over cameras of 2 - 2 cos)
    diff = sum((2 - 2 * (k_o[name] * k_r[name]).sum(-1)).clamp_min(0) for name in k_o) / len(k_o)
    out["key_shift"] = float(diff.median() / gap)
    del policy
    factory._encoders.clear()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def action_expert(name: str, rec: dict, prompts: dict[str, list[str]], batch: int = 16) -> dict:
    policy, pre, post = load_vla(name)
    chunks: dict[str, list[np.ndarray]] = {}
    for tag, pr in (("original", prompts["original"]), ("original_2", prompts["original"]),
                    ("reworded", prompts["reworded"]), ("other_task", prompts["other"])):  # fmt: skip
        out = []
        for s in range(0, len(pr), batch):
            sl = slice(s, s + batch)
            b = {
                cam: torch.from_numpy(rec["images"][sl, i]).float() / 255.0
                for i, cam in enumerate(DATASET_CAMERAS)
            }
            b[OBS_STATE] = torch.from_numpy(rec["state"][sl]).float()
            b["task"] = pr[sl]
            policy.reset()
            out.append(post(policy.predict_action_chunk(pre(b))).float().cpu().numpy()[:, :HORIZON, MOTION])
        chunks[tag] = np.concatenate(out)
    ref = chunks["original"]
    mse = {
        tag: float(np.median(((chunks[tag] - ref) ** 2).mean((1, 2))))
        for tag in ("original_2", "reworded", "other_task")
    }
    res = {"same": mse["original_2"], "reworded": mse["reworded"], "other_task": mse["other_task"]}
    res["rewording_effect"] = (res["reworded"] - res["same"]) / max(res["other_task"] - res["same"], 1e-12)
    del policy
    torch.cuda.empty_cache()
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_task", type=int, default=10)
    parser.add_argument("--backbones", default=";".join(BACKBONES))
    parser.add_argument("--vlas", default=",".join(VLAS))
    parser.add_argument("--out", default=str(OUT / "language_sensitivity.json"))
    args = parser.parse_args()
    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {"backbones": {}, "vlas": {}}
    factory = MemoryFactory()
    rec, original, reworded, other = load_frames(factory, args.per_task)
    print(f"{len(original)} clean frames; e.g. {original[0]!r} -> {reworded[0]!r}", flush=True)
    for space in args.backbones.split(";"):
        if space and space not in res["backbones"]:
            try:
                res["backbones"][space] = backbone(space, factory, rec, original, reworded)
            except FileNotFoundError as e:
                print(f"skip {space}: {e}")
                continue
            print(space, res["backbones"][space], flush=True)
            out_path.write_text(json.dumps(res, indent=1))
    prompts = {"original": original, "reworded": reworded, "other": other}
    for name in args.vlas.split(","):
        if name and name not in res["vlas"]:
            res["vlas"][name] = action_expert(name, rec, prompts)
            print(name, res["vlas"][name], flush=True)
            out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
