"""Group episodes (recording order) into chains: same arm, and the shoe is grasped where the previous episode put it."""

import json

import numpy as np

TMP = "/home/denis-office/.claude/jobs/34f24b0e/tmp"
episodes = json.load(open(f"{TMP}/episodes.json"))
info = []
for i, e in enumerate(episodes):
    if e["date"] != "2026-09-30":
        continue
    xl, xr = np.array(e["xyz_l"]), np.array(e["xyz_r"])
    side = "l" if np.ptp(xl, axis=0).max() > np.ptp(xr, axis=0).max() else "r"
    x = xl if side == "l" else xr
    g = np.array(e[f"act_grip_{side}"])
    idx = np.flatnonzero(g < (g.min() + g.max()) / 2)
    info.append({"i": i, "side": side, "grasp": x[idx[0]][:2], "release": x[idx[-1]][:2], "t": e["t_ms"] / 1000})

chains, cur = [], [info[0]]
for prev, e in zip(info, info[1:], strict=False):
    d = np.linalg.norm(prev["release"] - e["grasp"])
    if e["side"] == prev["side"] and d < 0.08:
        cur.append(e)
    else:
        chains.append(cur)
        cur = [e]
        print(f"break before #{e['i']}: side {prev['side']}->{e['side']}, release->grasp {d:.2f} m")
chains.append(cur)
out = []
for c in chains:
    print(f"chain #{c[0]['i']}-#{c[-1]['i']} ({len(c)} eps, arm {c[0]['side']}): first grasp {c[0]['grasp'].round(2)}, first release {c[0]['release'].round(2)}")
    out.append([e["i"] for e in c])
json.dump(out, open(f"{TMP}/chains.json", "w"))
