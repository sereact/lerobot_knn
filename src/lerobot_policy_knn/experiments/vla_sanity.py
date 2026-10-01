"""Does a VLA checkpoint reproduce its own training demos? An offline check before spending simulator time on it.

Runs the exact rollout path (``build_runner`` -> ``VLARollout._predict``: the spec's camera renames, processors and
action transforms) on 48 ``lerobot/libero`` frames, optionally of one suite (per-suite checkpoints), and compares the
predicted 8-step motion with the demonstrated one and with a shuffled pairing (predictions against other frames'
demos). A checkpoint no better than half the shuffled error is reported as broken; the exit status is 1 then, so run
scripts can gate on it. Results are merged into ``outputs/vla/hub_sanity.json``.

    uv run python -m lerobot_policy_knn.experiments.vla_sanity groot_libero_spatial --suite=libero_spatial
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE

from ..features import load_frame_index
from ..vla import DATASET_CAMERAS
from .vla_rollouts import build_runner

OUT = Path("outputs/vla/hub_sanity.json")
HORIZON = 8


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="VLA registry name or <family>@<checkpoint>")
    parser.add_argument("--suite", default=None, help="only frames of this suite's tasks")
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()
    f = load_frame_index("lerobot/libero")
    suite_of = {}  # task prompt -> suite, from earlier closed-loop runs
    for line in Path("outputs/vla/results_episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        suite_of[e["task"]] = e["suite"]
    starts = np.nonzero(f.frame_index == 0)[0]
    eligible = [
        r
        for r in starts
        if args.suite is None or suite_of.get(f.task_names[int(f.task_index[r])]) == args.suite
    ]
    rng = np.random.default_rng(0)
    picked = rng.choice(eligible, args.frames, replace=False)
    rows = [int(rng.integers(r, max(r + 1, int(f.episode_end[r]) - 10))) for r in picked]
    demo = np.stack([f.action[np.minimum(r + np.arange(HORIZON), f.episode_end[r])] for r in rows])
    ds = LeRobotDataset("lerobot/libero")
    items = [ds[r] for r in rows]
    runner, _, _ = build_runner(args.name, 10)
    preds = []
    for s in range(0, len(items), 8):
        batch = items[s : s + 8]
        sub = {cam: torch.stack([it[cam] for it in batch]) for cam in DATASET_CAMERAS}
        sub[OBS_STATE] = torch.stack([it[OBS_STATE] for it in batch])
        sub["task"] = [it["task"] for it in batch]
        with torch.no_grad():
            chunk = runner._predict(sub)  # noqa: SLF001
        if runner.action_transform is not None:
            b, c = chunk.shape[:2]
            chunk = runner.action_transform(chunk.reshape(b * c, -1)).reshape(b, c, -1)
        preds.append(chunk[:, :HORIZON].float().numpy())
    pred = np.concatenate(preds)
    mse = float(((pred[..., :6] - demo[..., :6]) ** 2).mean())
    shuffled = float(((pred[..., :6] - np.roll(demo, 1, axis=0)[..., :6]) ** 2).mean())
    grip = float((np.sign(pred[..., 6]) == np.sign(demo[..., 6])).mean())
    entry = {
        "mse": round(mse, 4),
        "shuffled": round(shuffled, 4),
        "gripper": round(grip, 3),
        "ok": mse < 0.5 * shuffled,
    }
    if args.suite:
        entry["suite"] = args.suite
    data = json.loads(OUT.read_text()) if OUT.exists() else {"checkpoints": {}}
    data["checkpoints"][args.name] = entry
    OUT.write_text(json.dumps(data, indent=1))
    print(json.dumps({args.name: entry}), flush=True)
    sys.exit(0 if entry["ok"] else 1)


if __name__ == "__main__":
    main()
