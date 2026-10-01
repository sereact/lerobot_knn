"""Do close embeddings have close actions *off the demonstrations*? LAC with LIBERO-plus demos as queries.

``neighbor_metric`` scores a key on held-out LIBERO demos, which the fine-tuned VLMs were nevertheless trained on.
Here the queries are frames of ``lerobot/libero_plus`` demonstrations: the same 40 tasks under LIBERO-plus
perturbations, which none of the checkpoints saw. The candidates are the clean ``lerobot/libero`` memory, i.e. what the
kNN policy retrieves from. For each query (``frame_index % 10 == 4``) the key is computed exactly as the policy does it
online (``encode_observation`` on the query images, the policy's own key normalisation), and

* ``source_hit`` -- LIBERO-plus demos *replay* a clean LIBERO demo's exact actions in the perturbed scene (38 of 40
  checked), so every query has a clean twin: the same state, rendered without the perturbation. ``source_hit`` is the
  fraction of queries whose nearest memory frame is that twin (its source demo, within 2 frames): does the key see
  through the perturbation to the same state?
* ``shift`` = median distance to the nearest memory frame / the median nearest-other-demo distance of clean held-out
  frames: how far the perturbation moves an observation, in units of the gap between demos
* ``lac@k`` = 1 - mean action distance to its k nearest memory frames of *other* demos than its source (at most one per
  demo) / mean action distance to a random memory frame of the task (10-step motion chunks), comparable to the
  held-out ``lac@k`` of ``neighbor_metric``

``--correlate`` relates them to the LIBERO-plus closed-loop success of the plain lookup with that key
(outputs/libero_plus/results.jsonl). Only the dataset files already in the Hub cache are used (see ``--files``).

    uv run python -m lerobot_policy_knn.experiments.plus_neighbor_metric --correlate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from lerobot.utils.constants import OBS_STATE

from ..modeling_knn import camera_key_name
from .head_vs_knn import MOTION, embed_records, space_config
from .search import MemoryFactory

OUT = Path("outputs/libero_plus")
PLUS_CAMERAS = {
    "observation.images.front": 0,
    "observation.images.wrist": 1,
}  # -> image, image2 (same orientation)
# key space -> config stem of the plain lookup with that key on LIBERO-plus
SPACES = {
    "resnet18_l3": "plain_resnet18_l3",
    "resnet18_l3|file:outputs/vla/learned_key_resnet.pt": "plain_resnet18_l3_learned",
    "dinov2": "plain_dinov2",
    "vla:smolvla_base/grid2": "plain_smolvla_base_grid2",
    "vla:smolvla_libero/grid2": "plain_smolvla_libero_grid2",
    "vla:pi05_base/grid2@L12|zscore": "plain_pi05_base_grid2@L12_z",
    "vla:pi05_libero/grid2@L12|zscore": "plain_pi05_libero_grid2@L12_z",
    "vla:pi0_base/grid2@L12|zscore": "plain_pi0_base_grid2@L12_z",
    "vla:pi0_libero/grid2@L12|zscore": "plain_pi0_libero_grid2@L12_z",
}
HORIZON = 10


def extra_keys() -> list[dict]:
    """Backbone keys added after the layer pilots (outputs/vla/extra_keys.json, written by choose_keys)."""
    path = Path("outputs/vla/extra_keys.json")
    return json.loads(path.read_text())["keys"] if path.exists() else []


for _k in extra_keys():
    SPACES[f"{_k['encoder']}/{_k['pooling']}" + (f"|{_k['norm']}" if _k["norm"] != "none" else "")] = _k[
        "stem"
    ]


def load_queries(max_video_file: int, per_task: int, seed: int = 0) -> dict:
    """Query frames of the LIBERO-plus demos whose videos lie in the first ``max_video_file + 1`` files of chunk 0."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ep = pd.read_parquet(hf_hub_download("lerobot/libero_plus", "meta/episodes/chunk-000/file-000.parquet",
                                         repo_type="dataset"))  # fmt: skip
    ok = np.ones(len(ep), dtype=bool)
    for cam in PLUS_CAMERAS:
        ok &= (ep[f"videos/{cam}/chunk_index"] == 0).to_numpy() & (
            ep[f"videos/{cam}/file_index"] <= max_video_file
        ).to_numpy()
    episodes = sorted(int(e) for e in ep["episode_index"][ok])
    ds = LeRobotDataset("lerobot/libero_plus", episodes=episodes)
    cols = ds.hf_dataset.with_format(None)
    ep_idx = np.asarray(cols["episode_index"]).reshape(-1)
    fr_idx = np.asarray(cols["frame_index"]).reshape(-1)
    task_idx = np.asarray(cols["task_index"]).reshape(-1)
    action = np.asarray(cols["action"], dtype=np.float32)
    last = np.r_[np.nonzero(np.diff(ep_idx))[0], len(ep_idx) - 1]  # last position of each episode run
    end = np.repeat(last, np.diff(np.r_[-1, last]))
    rng = np.random.default_rng(seed)
    cand = np.nonzero(fr_idx % 10 == 4)[0]
    pos = np.sort(np.concatenate([rng.choice(c, min(per_task, len(c)), replace=False)
                                  for t in np.unique(task_idx[cand]) for c in [cand[task_idx[cand] == t]]]))  # fmt: skip
    images = np.zeros((len(pos), 2, 3, 256, 256), dtype=np.uint8)
    state = np.zeros((len(pos), 8), dtype=np.float32)
    for i, p in enumerate(pos):
        item = ds[int(p)]
        for cam, j in PLUS_CAMERAS.items():
            images[i, j] = (item[cam].clamp(0, 1) * 255).round().byte().numpy()
        state[i] = item[OBS_STATE].numpy()
    chunk = action[np.minimum(pos[:, None] + np.arange(HORIZON), end[pos][:, None])]
    names = list(ds.meta.tasks.index)
    source = source_episodes(ep_idx, task_idx, action, names)
    return {"images": images, "state": state, "chunk": chunk, "t": fr_idx[pos], "task_names": np.array(names),
            "task": task_idx[pos], "episode": ep_idx[pos], "source": np.array([source[e] for e in ep_idx[pos]])}  # fmt: skip


def source_episodes(
    ep_idx: np.ndarray, task_idx: np.ndarray, action: np.ndarray, names: list[str]
) -> dict[int, int]:
    """LIBERO-plus episode -> the clean lerobot/libero episode whose action sequence it replays exactly (or -1)."""
    from ..features import load_frame_index

    f = load_frame_index("lerobot/libero")
    by_task: dict[int, list[tuple[int, np.ndarray]]] = {}
    for ce in np.unique(f.episode_index):
        rows = f.episode_index == ce
        by_task.setdefault(int(f.task_index[rows][0]), []).append((int(ce), f.action[rows]))
    out = {}
    for e in np.unique(ep_idx):
        rows = ep_idx == e
        a = action[rows]
        t = f.task_names.index(names[int(task_idx[rows][0])])
        match = [ce for ce, b in by_task[t] if len(b) == len(a) and np.abs(a - b).max() < 1e-4]
        out[int(e)] = match[0] if match else -1
    return out


@torch.no_grad()
def evaluate(space: str, factory: MemoryFactory, q: dict, ks=(1, 8)) -> dict:
    frames = factory.frames
    task_of = {name: i for i, name in enumerate(frames.task_names)}
    q_task = torch.tensor([task_of[str(q["task_names"][i])] for i in q["task"]]).cuda()
    policy = factory.build(space_config(space, max(ks), HORIZON))
    keys = embed_records(policy, q)
    t = torch.from_numpy(q["t"]).cuda()
    src = torch.from_numpy(q["source"]).cuda()
    has_src = src >= 0
    # 1. invariance: is the nearest memory frame the query's clean twin?
    rows, _, d = policy.retrieve(keys, t, q_task)
    top_ep, top_fr = policy.key_episode[rows[:, 0]], policy.key_frame[rows[:, 0]]
    hit = (top_ep == src) & ((top_fr - t).abs() <= 2)
    # 2. action consistency among the other demos
    rows, _, _ = policy.retrieve(keys, t, q_task, exclude_episode=src)
    nn = policy.gather_chunks(rows, HORIZON)[..., MOTION].float()  # (Q, k, H, 6)
    y = torch.from_numpy(q["chunk"][..., MOTION]).cuda()
    d_nn = (nn - y[:, None]).pow(2).mean((2, 3))  # (Q, k)
    # random memory frames of the same task (every 2nd frame, as the memory)
    mem = np.nonzero(frames.frame_index % 2 == 0)[0]
    idx = np.minimum(mem[:, None] + np.arange(HORIZON), frames.episode_end[mem][:, None])
    table = torch.from_numpy(frames.action[idx][..., MOTION]).cuda()
    mem_task = torch.from_numpy(frames.task_index[mem]).cuda()
    rng = np.random.default_rng(0)
    d_rand = torch.empty(len(y), device="cuda")
    for tk in q_task.unique():
        sel = torch.nonzero(q_task == tk).squeeze(1)
        pool = torch.nonzero(mem_task == tk).squeeze(1)
        pool = pool[torch.from_numpy(rng.choice(len(pool), min(512, len(pool)), replace=False)).cuda()]
        d_rand[sel] = (table[pool][None] - y[sel][:, None]).pow(2).mean((2, 3)).mean(1)
    res = {f"lac@{k}": float((1 - d_nn[:, :k].mean(1) / d_rand.clamp_min(1e-9)).mean()) for k in ks}
    res["source_hit"] = float(hit[has_src].float().mean())
    res["n_queries"] = int(len(y))
    res["n_with_source"] = int(has_src.sum())
    # reference distance: held-out clean demo frames against the other demos (leave-one-episode-out)
    ref_rows = np.random.default_rng(1).choice(
        np.nonzero(frames.frame_index % 10 == 4)[0], 1000, replace=False
    )
    cfg = policy.config
    ref = {}
    for cam in cfg.active_cameras:
        raw = factory.load_raw(cfg.image_encoder, cam, cfg.image_pooling, cfg.image_resolution)
        name = camera_key_name(cam)
        ref[name] = policy.key_features(name, torch.as_tensor(np.asarray(raw[np.sort(ref_rows)])).cuda())
    ref_rows = np.sort(ref_rows)
    _, _, d_ref = policy.retrieve(ref, torch.from_numpy(frames.frame_index[ref_rows]).cuda(),
                                  torch.from_numpy(frames.task_index[ref_rows]).cuda(),
                                  exclude_episode=torch.from_numpy(frames.episode_index[ref_rows]).cuda())  # fmt: skip
    res["shift"] = float(d[:, 0].median() / d_ref[:, 0].median().clamp_min(1e-9))
    del policy
    factory._encoders.clear()
    torch.cuda.empty_cache()
    return res


def correlate(results: dict) -> dict:
    succ = {}
    for line in (OUT / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("tag") == "libero_plus" and r["policy"].startswith("knn:outputs/vla/configs/"):
            succ[Path(r["policy"]).stem] = 100 * r["all"]
    pts = {s: {"success": succ[SPACES[s]], **m} for s, m in results.items() if SPACES[s] in succ}
    out: dict = {"n": len(pts), "points": pts}
    if len(pts) >= 4:
        y = pd.Series({s: v["success"] for s, v in pts.items()})
        for metric in ("lac@1", "lac@8", "source_hit", "shift"):
            x = pd.Series({s: v[metric] for s, v in pts.items()})
            out[f"spearman_{metric}"] = float(x.corr(y, method="spearman"))
            out[f"pearson_{metric}"] = float(x.corr(y))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default=";".join(SPACES), help="';'-separated key spaces")
    parser.add_argument(
        "--files", type=int, default=1, help="use the videos of chunk-000 files 0..N (cached)"
    )
    parser.add_argument("--per_task", type=int, default=40)
    parser.add_argument("--correlate", action="store_true")
    parser.add_argument("--out", default=str(OUT / "neighbor_metric_plus.json"))
    args = parser.parse_args()
    out_path = Path(args.out)
    results = json.loads(out_path.read_text())["spaces"] if out_path.exists() else {}
    results = {k: v for k, v in results.items() if k in SPACES}
    todo = [s for s in args.spaces.split(";") if s not in results]
    if todo:
        q = load_queries(args.files, args.per_task)
        print(f"{len(q['t'])} LIBERO-plus query frames from {len(np.unique(q['episode']))} demos", flush=True)
        factory = MemoryFactory()
        for space in todo:
            try:
                results[space] = evaluate(space, factory, q)
            except FileNotFoundError as e:  # a VLM key whose memory features are not cached yet
                print(f"skip {space}: {e}")
                continue
            print(space, {k: round(v, 3) for k, v in results[space].items()}, flush=True)
            out_path.write_text(json.dumps({"spaces": results}, indent=1))
    summary: dict = {"spaces": results}
    if args.correlate:
        summary["correlation"] = correlate(results)
        print(json.dumps({k: v for k, v in summary["correlation"].items() if k != "points"}, indent=1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
