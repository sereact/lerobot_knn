"""Where do rollouts fail? Trace, for every retrieval, the progress of the retrieved demonstration frame
(0 = start of its demo, 1 = end) and the normalised top-1 distance (1 = a typical leave-one-demo-out
nearest neighbour), then classify failed episodes.

    uv run python -m lerobot_policy_knn.experiments.diagnose --config=cfg.json --suite=libero_10 --init_states=0:3
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..configuration_knn import KNNConfig
from .search import OUT, MemoryFactory
from .visualize import rollout


def classify(progress: np.ndarray, dist: np.ndarray, success: bool) -> str:
    if success:
        return "success"
    peak = progress.max()
    final = progress[-20:].mean() if len(progress) >= 20 else progress.mean()
    if np.median(dist[-20:]) > 3.0:
        return "unseen state (distance > 3x typical)"
    if peak > 0.55 and final < peak - 0.3:
        return "phase regression (fell back to an earlier part of the demo)"
    if final < 0.5:
        return "stuck in first half"
    return "stuck in second half"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="0:10")
    parser.add_argument("--init_states", default="0:3")
    parser.add_argument("--out", default=str(OUT / "diagnose_libero_10.json"))
    args = parser.parse_args()
    cfg = KNNConfig(device="cuda", **json.loads(Path(args.config).read_text()))
    factory = MemoryFactory()
    policy = factory.build(cfg)
    frames = factory.frames
    calib = policy.task_distance_scale().cpu().numpy()
    t_lo, t_hi = (int(x) for x in args.tasks.split(":"))
    lo, hi = (int(x) for x in args.init_states.split(":"))
    episodes = []
    for task_id in range(t_lo, t_hi):
        for init in range(lo, hi):
            ep = rollout(policy, args.suite, task_id, init)
            tid = cfg.task_names.index(ep["task"])
            top = np.array([int(e["rows"][0][0]) for e in ep["log"]])
            ptr = policy.key_ptr[top].cpu().numpy()
            progress = frames.frame_index[ptr] / np.maximum(frames.episode_length[ptr] - 1, 1)
            dist = np.array([float(e["dists"][0][0]) for e in ep["log"]]) / calib[tid]
            kind = classify(progress, dist, ep["success"])
            episodes.append({
                "task_id": task_id, "init_state": init, "task": ep["task"], "success": ep["success"],
                "steps": ep["steps"], "kind": kind, "peak_progress": round(float(progress.max()), 3),
                "progress": [round(float(p), 3) for p in progress[::4]],
                "distance": [round(float(d), 2) for d in dist[::4]],
            })  # fmt: skip
            print(f"task {task_id} init {init}: {'OK ' if ep['success'] else 'FAIL'} {ep['steps']:4d} steps, "
                  f"peak progress {progress.max():.2f}, final dist {np.median(dist[-20:]):.1f}x -> {kind}", flush=True)  # fmt: skip
    Path(args.out).write_text(json.dumps(episodes))
    kinds: dict[str, int] = {}
    for e in episodes:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(json.dumps(kinds, indent=1))


if __name__ == "__main__":
    main()
