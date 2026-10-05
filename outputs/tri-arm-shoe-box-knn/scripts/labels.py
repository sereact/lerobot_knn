"""Visual task labels (read off each episode's first wrist frames) and their consistency with the trajectories."""

import json

import numpy as np

TMP = "/home/denis-office/.claude/jobs/34f24b0e/tmp"
OUT_EPS = {1, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53,
           55, 57, 59, 60, 62, 64, 66, 68, 69, 71, 73, 75, 77, 79, 81, 83, 84, 86, 88, 90, 92, 94, 95, 97, 99, 101,
           103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127}  # fmt: skip
INCOMPLETE = {2}  # the shoe is grasped in the last frames and never placed
episodes = json.load(open(f"{TMP}/episodes.json"))
labels = {}
prev = None
for i, e in enumerate(episodes):
    if e["date"] != "2026-09-30":
        continue
    xl, xr = np.array(e["xyz_l"]), np.array(e["xyz_r"])
    side = "l" if np.ptp(xl, axis=0).max() > np.ptp(xr, axis=0).max() else "r"
    x = xl if side == "l" else xr
    g = np.array(e[f"act_grip_{side}"])
    closed = np.flatnonzero(g < (g.min() + g.max()) / 2)
    grasp, release = x[closed[0]][:2], x[closed[-1]][:2]
    label = "incomplete" if i in INCOMPLETE else ("out" if i in OUT_EPS else "in")
    box = grasp if label == "out" else release
    labels[i] = {"label": label, "side": side, "box": box.round(3).tolist(), "grasp": grasp.round(3).tolist(),
                 "release": release.round(3).tolist(), "dataset": e["dataset"], "episode": e["episode"],
                 "t_ms": e["t_ms"], "length": e["length"]}  # fmt: skip
    if prev is not None and label != "incomplete" and labels[prev]["label"] != "incomplete":
        p = labels[prev]
        gap = e["t_ms"] / 1000 - (p["t_ms"] / 1000 + p["length"] / 30)
        linked = p["side"] == side and np.linalg.norm(np.array(p["release"]) - grasp) < 0.08
        if label == p["label"]:
            print(f"#{prev}->#{i}: both {label}, arm {p['side']}->{side}, gap {gap:.0f}s, linked {linked}")
        box_jump = np.linalg.norm(np.array(p["box"]) - box) if p["side"] == side else 0.0
        if label != p["label"] and box_jump > 0.08:
            print(f"#{prev}->#{i}: box moved {box_jump:.2f} m between episodes (gap {gap:.0f}s)")
    prev = i
json.dump(labels, open(f"{TMP}/labels.json", "w"), indent=0)
n = {k: sum(v["label"] == k for v in labels.values()) for k in ("in", "out", "incomplete")}
print(n)
