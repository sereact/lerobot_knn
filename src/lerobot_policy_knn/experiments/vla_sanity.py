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

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE

from ..features import load_frame_index
from ..vla import DATASET_CAMERAS
from .common import GRIPPER, MOTION, REPO_ID, VLA_OUT, action_chunks, read_jsonl
from .vla_rollouts import build_runner

OUT = VLA_OUT / "hub_sanity.json"
HORIZON = 8
BATCH = 8


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="VLA registry name or <family>@<checkpoint>")
    parser.add_argument("--suite", default=None, help="only frames of this suite's tasks")
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()
    frames = load_frame_index(REPO_ID)
    # task prompt -> suite, from earlier closed-loop runs
    suite_of = {e["task"]: e["suite"] for e in read_jsonl(VLA_OUT / "results_episodes.jsonl")}
    eligible = [  # first frames of the demonstrations to draw from
        row
        for row in np.nonzero(frames.frame_index == 0)[0]
        if args.suite is None or suite_of.get(frames.task_names[int(frames.task_index[row])]) == args.suite
    ]
    rng = np.random.default_rng(0)
    starts = rng.choice(eligible, args.frames, replace=False)
    # one random frame of each drawn demonstration, at least 10 steps before its end
    rows = np.array([int(rng.integers(r, max(r + 1, int(frames.episode_end[r]) - 10))) for r in starts])
    demo = action_chunks(frames, rows, HORIZON)
    dataset = LeRobotDataset(REPO_ID)
    items = [dataset[int(r)] for r in rows]
    runner, _, _ = build_runner(args.name, 10)
    preds = []
    for s in range(0, len(items), BATCH):
        batch = items[s : s + BATCH]
        sub = {cam: torch.stack([item[cam] for item in batch]) for cam in DATASET_CAMERAS}
        sub[OBS_STATE] = torch.stack([item[OBS_STATE] for item in batch])
        sub["task"] = [item["task"] for item in batch]
        with torch.no_grad():
            chunk = runner._predict(sub)  # noqa: SLF001
        if runner.action_transform is not None:
            chunk = runner.action_transform(chunk.flatten(0, 1)).unflatten(0, chunk.shape[:2])
        preds.append(chunk[:, :HORIZON].float().numpy())
    pred = np.concatenate(preds)
    mse = float(((pred[..., MOTION] - demo[..., MOTION]) ** 2).mean())
    shuffled = float(((pred[..., MOTION] - np.roll(demo, 1, axis=0)[..., MOTION]) ** 2).mean())
    gripper = float((np.sign(pred[..., GRIPPER]) == np.sign(demo[..., GRIPPER])).mean())
    entry = {
        "mse": round(mse, 4),
        "shuffled": round(shuffled, 4),
        "gripper": round(gripper, 3),
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
