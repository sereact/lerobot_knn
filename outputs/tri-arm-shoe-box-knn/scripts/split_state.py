"""observation.state (32: left/right xyz+rot6d+gripper, then 12 TCP wrench values) -> observation.state (20, the layout
derroboter's inference node sends for the two UR arms and the layout of `action`) + observation.tcp_wrench (12)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import modify_features

OUT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box")
TMP = Path("/home/denis-office/.claude/jobs/34f24b0e/tmp/split")
REPO_ID = "local/tri_arm_shoe_box"
if TMP.exists():
    shutil.rmtree(TMP)

ds = LeRobotDataset(REPO_ID, root=OUT)
info = ds.meta.features["observation.state"]
assert list(info["shape"]) == [32], info
names = info["names"]
assert names[20:] == ["left_fx", "left_fy", "left_fz", "left_mx", "left_my", "left_mz",
                      "right_fx", "right_fy", "right_fz", "right_mx", "right_my", "right_mz"], names  # fmt: skip
assert names[:20] == ds.meta.features["action"]["names"], (names[:20], ds.meta.features["action"]["names"])
data = pd.concat(pd.read_parquet(f) for f in sorted((OUT / "data").rglob("*.parquet"))).sort_values("index")
assert (data["index"].to_numpy() == np.arange(len(data))).all()
state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)

pass1 = modify_features(
    ds,
    add_features={
        "observation.state_pose": (state[:, :20], {"dtype": "float32", "shape": [20], "names": names[:20]}),
        "observation.tcp_wrench": (state[:, 20:], {"dtype": "float32", "shape": [12], "names": names[20:]}),
    },
    remove_features="observation.state",
    output_dir=TMP / "pass1",
    repo_id=f"{REPO_ID}_pass1",
)
pass2 = modify_features(
    pass1,
    add_features={"observation.state": (state[:, :20], {"dtype": "float32", "shape": [20], "names": names[:20]})},
    remove_features="observation.state_pose",
    output_dir=TMP / "final",
    repo_id=REPO_ID,
)
shutil.copy(OUT / "episode_sources.json", TMP / "final" / "episode_sources.json")
shutil.rmtree(OUT)
shutil.move(TMP / "final", OUT)
shutil.rmtree(TMP)

final = LeRobotDataset(REPO_ID, root=OUT)
item = final[1234]
assert np.allclose(item["observation.state"].numpy(), state[1234, :20])
assert np.allclose(item["observation.tcp_wrench"].numpy(), state[1234, 20:])
print("features:", {k: v["shape"] for k, v in final.meta.features.items()})
print("tasks:", final.meta.tasks.index.tolist(), "episodes:", final.meta.total_episodes)
print("episode 5 task:", list(final.meta.episodes[5]["tasks"]), "| sources says:", json.loads((OUT / "episode_sources.json").read_text())[5]["task"])
print("stats keys:", sorted(final.meta.stats))
