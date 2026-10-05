"""Recompute the numeric stats of the shoe-box dataset (the split state features have none yet)."""

from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats

OUT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box")
ds = recompute_stats(LeRobotDataset("local/tri_arm_shoe_box", root=OUT))
ds = LeRobotDataset("local/tri_arm_shoe_box", root=OUT)
for key in ("observation.state", "observation.tcp_wrench", "action"):
    s = ds.meta.stats[key]
    print(key, "mean", [round(float(x), 3) for x in s["mean"][:10]], "std", [round(float(x), 3) for x in s["std"][:10]])
