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

from ..features import load_frame_index
from .common import MOTION, PLUS_OUT, REPO_ID, action_chunks, extra_keys, read_jsonl, space_name
from .head_vs_knn import demo_queries, embed_records, space_config
from .search import MemoryFactory

PLUS_REPO_ID = "lerobot/libero_plus"
# LIBERO-plus camera -> its position among the dataset cameras (image, image2; same orientation)
PLUS_CAMERAS = {"observation.images.front": 0, "observation.images.wrist": 1}
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
SPACES.update({space_name(k["encoder"], k["pooling"], k["norm"]): k["stem"] for k in extra_keys()})
HORIZON = 10


def load_queries(max_video_file: int, per_task: int, seed: int = 0) -> dict:
    """Query frames of the LIBERO-plus demos whose videos lie in the first ``max_video_file + 1`` files of chunk 0,
    in the layout ``VLARollout`` records (see ``common.record_batch``), plus their action chunk and source demo."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta_file = hf_hub_download(PLUS_REPO_ID, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    meta = pd.read_parquet(meta_file)
    cached = np.ones(len(meta), dtype=bool)
    for cam in PLUS_CAMERAS:
        in_chunk_0 = (meta[f"videos/{cam}/chunk_index"] == 0).to_numpy()
        cached &= in_chunk_0 & (meta[f"videos/{cam}/file_index"] <= max_video_file).to_numpy()
    dataset = LeRobotDataset(PLUS_REPO_ID, episodes=sorted(int(e) for e in meta["episode_index"][cached]))
    columns = dataset.hf_dataset.with_format(None)
    episode = np.asarray(columns["episode_index"]).reshape(-1)
    frame = np.asarray(columns["frame_index"]).reshape(-1)
    task = np.asarray(columns["task_index"]).reshape(-1)
    action = np.asarray(columns["action"], dtype=np.float32)
    last = np.r_[np.nonzero(np.diff(episode))[0], len(episode) - 1]  # last position of each episode run
    episode_end = np.repeat(last, np.diff(np.r_[-1, last]))
    rng = np.random.default_rng(seed)
    candidates = np.nonzero(frame % 10 == 4)[0]
    picked = []
    for t in np.unique(task[candidates]):
        of_task = candidates[task[candidates] == t]
        picked.append(rng.choice(of_task, min(per_task, len(of_task)), replace=False))
    pos = np.sort(np.concatenate(picked))
    images = np.zeros((len(pos), 2, 3, 256, 256), dtype=np.uint8)
    state = np.zeros((len(pos), 8), dtype=np.float32)
    for i, p in enumerate(pos):
        item = dataset[int(p)]
        for cam, j in PLUS_CAMERAS.items():
            images[i, j] = (item[cam].clamp(0, 1) * 255).round().byte().numpy()
        state[i] = item[OBS_STATE].numpy()
    names = list(dataset.meta.tasks.index)
    source = source_episodes(episode, task, action, names)
    return {
        "images": images,
        "state": state,
        "chunk": action[np.minimum(pos[:, None] + np.arange(HORIZON), episode_end[pos][:, None])],
        "t": frame[pos],
        "task_names": np.array(names),
        "task": task[pos],
        "episode": episode[pos],
        "source": np.array([source[e] for e in episode[pos]]),
    }


def source_episodes(
    episode: np.ndarray, task: np.ndarray, action: np.ndarray, names: list[str]
) -> dict[int, int]:
    """LIBERO-plus episode -> the clean lerobot/libero episode whose action sequence it replays exactly (or -1)."""
    clean = load_frame_index(REPO_ID)
    by_task: dict[int, list[tuple[int, np.ndarray]]] = {}
    for e in np.unique(clean.episode_index):
        rows = clean.episode_index == e
        by_task.setdefault(int(clean.task_index[rows][0]), []).append((int(e), clean.action[rows]))
    out = {}
    for e in np.unique(episode):
        rows = episode == e
        actions = action[rows]
        clean_task = clean.task_names.index(names[int(task[rows][0])])
        matches = [
            ce for ce, b in by_task[clean_task] if len(b) == len(actions) and np.abs(actions - b).max() < 1e-4
        ]
        out[int(e)] = matches[0] if matches else -1
    return out


@torch.no_grad()
def evaluate(space: str, factory: MemoryFactory, q: dict, ks: tuple[int, ...] = (1, 8)) -> dict:
    """``source_hit``, ``lac@k`` and ``shift`` (see the module docstring) of one key space on the queries ``q``."""
    frames = factory.frames
    task_of = {name: i for i, name in enumerate(frames.task_names)}
    q_task = torch.tensor([task_of[str(q["task_names"][i])] for i in q["task"]]).cuda()
    policy = factory.build(space_config(space, max(ks), HORIZON))
    keys = embed_records(policy, q)
    t = torch.from_numpy(q["t"]).cuda()
    source = torch.from_numpy(q["source"]).cuda()
    has_source = source >= 0
    # 1. invariance: is the nearest memory frame the query's clean twin?
    rows, _, d = policy.retrieve(keys, t, q_task)
    top_episode, top_frame = policy.key_episode[rows[:, 0]], policy.key_frame[rows[:, 0]]
    hit = (top_episode == source) & ((top_frame - t).abs() <= 2)
    # 2. action consistency among the other demos
    rows, _, _ = policy.retrieve(keys, t, q_task, exclude_episode=source)
    neighbours = policy.gather_chunks(rows, HORIZON)[..., MOTION].float()  # (Q, k, H, 6)
    y = torch.from_numpy(q["chunk"][..., MOTION]).cuda()
    d_neighbours = (neighbours - y[:, None]).pow(2).mean((2, 3))  # (Q, k)
    # against random memory frames of the same task (every 2nd frame, as the memory)
    memory = np.nonzero(frames.frame_index % 2 == 0)[0]
    table = torch.from_numpy(action_chunks(frames, memory, HORIZON)[..., MOTION]).cuda()
    memory_task = torch.from_numpy(frames.task_index[memory]).cuda()
    rng = np.random.default_rng(0)
    d_random = torch.empty(len(y), device="cuda")
    for task in q_task.unique():
        queries = torch.nonzero(q_task == task).squeeze(1)
        pool = torch.nonzero(memory_task == task).squeeze(1)
        pool = pool[torch.from_numpy(rng.choice(len(pool), min(512, len(pool)), replace=False)).cuda()]
        d_random[queries] = (table[pool][None] - y[queries][:, None]).pow(2).mean((2, 3)).mean(1)
    res = {f"lac@{k}": float((1 - d_neighbours[:, :k].mean(1) / d_random.clamp_min(1e-9)).mean()) for k in ks}
    res["source_hit"] = float(hit[has_source].float().mean())
    res["n_queries"] = int(len(y))
    res["n_with_source"] = int(has_source.sum())
    # 3. reference distance: held-out clean demo frames against the other demos (leave-one-episode-out)
    clean_queries = np.nonzero(frames.frame_index % 10 == 4)[0]
    clean_rows = np.sort(np.random.default_rng(1).choice(clean_queries, 1000, replace=False))
    _, _, d_clean = policy.retrieve(
        demo_queries(policy, factory, clean_rows),
        torch.from_numpy(frames.frame_index[clean_rows]).cuda(),
        torch.from_numpy(frames.task_index[clean_rows]).cuda(),
        exclude_episode=torch.from_numpy(frames.episode_index[clean_rows]).cuda(),
    )
    res["shift"] = float(d[:, 0].median() / d_clean[:, 0].median().clamp_min(1e-9))
    del policy
    factory.release_encoders()
    return res


def correlate(results: dict) -> dict:
    """Each metric against the LIBERO-plus closed-loop success of the plain lookup with that key."""
    success = {}  # config stem -> success (%)
    for run in read_jsonl(PLUS_OUT / "results.jsonl"):
        if run.get("tag") == "libero_plus" and run["policy"].startswith("knn:outputs/vla/configs/"):
            success[Path(run["policy"]).stem] = 100 * run["all"]
    points = {s: {"success": success[SPACES[s]], **m} for s, m in results.items() if SPACES[s] in success}
    out: dict = {"n": len(points), "points": points}
    if len(points) >= 4:
        y = pd.Series({s: v["success"] for s, v in points.items()})
        for metric in ("lac@1", "lac@8", "source_hit", "shift"):
            x = pd.Series({s: v[metric] for s, v in points.items()})
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
    parser.add_argument("--out", default=str(PLUS_OUT / "neighbor_metric_plus.json"))
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
