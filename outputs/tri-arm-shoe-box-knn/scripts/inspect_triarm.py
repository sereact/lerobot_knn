"""Summarise the tri-arm-gate-testing datasets: episodes, lengths, tasks, feature sets."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-gate-testing")
ref_features = None
for d in sorted(ROOT.iterdir(), key=lambda p: p.name.split("_")[-2:]):
    meta = d / "original" / "meta"
    info = json.loads((meta / "info.json").read_text())
    tasks = pd.read_parquet(meta / "tasks.parquet")
    eps = pd.concat(pd.read_parquet(f) for f in sorted((meta / "episodes").rglob("*.parquet")))
    md = json.loads((d / "original" / "_metadata.json").read_text())
    feats = sorted(info["features"])
    if ref_features is None:
        ref_features = feats
    diff = set(feats) ^ set(ref_features)
    print(
        f"{d.name}: {info['total_episodes']} eps, {info['total_frames']} frames, lengths {eps['length'].tolist()}, "
        f"tasks {tasks.index.tolist()}, subtype {md.get('robot_subtype')}, task_type {md.get('task_type')}"
        + (f", feature diff {sorted(diff)}" if diff else "")
    )
print("episode columns:", [c for c in eps.columns if not c.startswith(("stats/", "videos/", "data/"))])
