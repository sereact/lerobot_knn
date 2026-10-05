"""Gripper series and the active arm's TCP at grasp / release, per episode (recording order)."""

import json

import numpy as np

episodes = json.load(open("/home/denis-office/.claude/jobs/34f24b0e/tmp/episodes.json"))
for i, e in enumerate(episodes[1:8], start=1):
    side = "l" if np.ptp(np.array(e["xyz_l"]), axis=0).max() > np.ptp(np.array(e["xyz_r"]), axis=0).max() else "r"
    g = np.array(e[f"grip_{side}"])
    a = np.array(e[f"act_grip_{side}"])
    print(i, side, "state grip every 10th:", g[::10].round(2).tolist())
    print("   action grip every 10th:", a[::10].round(2).tolist())

print()
rows = []
for i, e in enumerate(episodes):
    if e["date"] != "2026-09-30":
        continue
    xl, xr = np.array(e["xyz_l"]), np.array(e["xyz_r"])
    side = "l" if np.ptp(xl, axis=0).max() > np.ptp(xr, axis=0).max() else "r"
    x = xl if side == "l" else xr
    g = np.array(e[f"act_grip_{side}"])
    closed = g < (g.min() + g.max()) / 2  # low = closed? printed above to check
    idx = np.flatnonzero(closed)
    if len(idx) == 0:
        print(i, side, "no grasp")
        continue
    grasp, release = idx[0], idx[-1]
    rows.append((i, side, x[0], x[grasp], x[release], x[-1]))
    print(
        f"{i:3d} {side} len {e['length']:3d} grasp@{grasp:3d} {x[grasp].round(2).tolist()} "
        f"release@{release:3d} {x[release].round(2).tolist()} start {x[0].round(2).tolist()} end {x[-1].round(2).tolist()}"
    )
