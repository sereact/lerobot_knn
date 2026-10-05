"""All tri-arm episodes in recording order, with gaps and gripper / TCP events (grasp and release positions)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-gate-testing")
OUT = Path("/home/denis-office/.claude/jobs/34f24b0e/tmp/episodes.json")

episodes = []
for d in sorted(ROOT.iterdir()):
    meta = d / "original" / "meta"
    rc = {e["episode_index"]: e for e in json.loads((meta / "recording_configs.json").read_text())["episodes"]}
    data = pd.concat(pd.read_parquet(f) for f in sorted((d / "original" / "data").rglob("*.parquet")))
    for ep_idx, frames in data.groupby("episode_index"):
        state = np.stack(frames["observation.state"].to_numpy())
        action = np.stack(frames["action"].to_numpy())
        head = np.stack(frames["observation.head_state"].to_numpy())
        si = rc[ep_idx]["source_info"]
        episodes.append(
            {
                "dataset": d.name,
                "episode": int(ep_idx),
                "t_ms": si["recorded_at_ms"],
                "date": si["date"],
                "length": len(frames),
                "grip_l": state[:, 9].round(3).tolist(),
                "grip_r": state[:, 19].round(3).tolist(),
                "xyz_l": state[:, 0:3].round(4).tolist(),
                "xyz_r": state[:, 10:13].round(4).tolist(),
                "act_grip_l": action[:, 9].round(3).tolist(),
                "act_grip_r": action[:, 19].round(3).tolist(),
                "head_move": float(np.abs(head - head[0]).max()),
            }
        )
episodes.sort(key=lambda e: e["t_ms"])
OUT.write_text(json.dumps(episodes))
print(f"{len(episodes)} episodes")
prev_end = None
for i, e in enumerate(episodes):
    start = e["t_ms"] / 1000
    gap = "" if prev_end is None else f"gap {start - prev_end:6.1f}s"
    prev_end = start + e["length"] / 30
    gl, gr = np.array(e["grip_l"]), np.array(e["grip_r"])
    xl, xr = np.array(e["xyz_l"]), np.array(e["xyz_r"])
    print(
        f"{i:3d} {e['dataset'][3:15]} ep{e['episode']} {e['date']} len {e['length']:4d} {gap} | "
        f"grip L {gl.min():.2f}-{gl.max():.2f} R {gr.min():.2f}-{gr.max():.2f} | "
        f"L moved {np.linalg.norm(xl - xl[0], axis=1).max():.2f} m, R moved {np.linalg.norm(xr - xr[0], axis=1).max():.2f} m"
        f" | head {e['head_move']:.3f}"
    )
