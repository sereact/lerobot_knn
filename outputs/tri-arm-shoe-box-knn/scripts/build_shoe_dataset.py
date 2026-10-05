"""Build one LeRobot dataset from the tri-arm-gate-testing recordings of 2026-09-30, with the two tasks labelled.

- drops the incomplete episode (#2 in recording order: e5cfa880 ep0, the shoe is grasped in the last frames) and the
  2026-09-15 gate-testing recording (no grasp);
- merges the 27 datasets;
- removes observation.images.head_left / head_right (mock camera streams: a coloured frame with a frame counter);
- sets each episode's task from the visual labels (labels.json).
Writes OUT and OUT/episode_sources.json (merged episode -> source dataset / episode / recording time / arm / task).
"""

import json
import logging
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets, modify_tasks, remove_feature

logging.basicConfig(level=logging.INFO)
SRC = Path("/home/denis-office/git/lerobot/outputs/tri-arm-gate-testing")
OUT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box")
TMP = Path("/home/denis-office/.claude/jobs/34f24b0e/tmp/build")
TASKS = {"in": "put the shoe in the box", "out": "take the shoe out of the box"}
REPO_ID = "local/tri_arm_shoe_box"



def fix_camera_names(root: Path) -> None:
    """The converter names a camera's axes after the camera (["left_rgb"]); LeRobot's feature tooling expects axis
    names. The shape is already [C, H, W]."""
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    for ft in info["features"].values():
        if ft["dtype"] in ("video", "image") and len(ft.get("names") or []) != 3:
            ft["names"] = ["channels", "height", "width"]
    path.write_text(json.dumps(info, indent=4))


labels = json.load(open("/home/denis-office/.claude/jobs/34f24b0e/tmp/labels.json"))
dropped = {(v["dataset"], v["episode"]) for v in labels.values() if v["label"] == "incomplete"}
by_source = {(v["dataset"], v["episode"]): v for v in labels.values() if v["label"] != "incomplete"}
names = sorted({v["dataset"] for v in labels.values()})
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)

datasets, kept_episodes = [], []
for name in names:
    ds = LeRobotDataset(f"local/{name}", root=SRC / name / "original")
    drop = sorted(ep for d, ep in dropped if d == name)
    if drop:
        ds = delete_episodes(ds, drop, output_dir=TMP / f"{name}_trimmed", repo_id=f"local/{name}_trimmed")
    keep = [ep for ep in range(ds.meta.total_episodes + len(drop)) if ep not in drop]
    assert len(keep) == ds.meta.total_episodes, (name, keep, ds.meta.total_episodes)
    datasets.append(ds)
    kept_episodes += [(name, ep) for ep in keep]

merged = merge_datasets(datasets, output_repo_id=f"{REPO_ID}_merged", output_dir=TMP / "merged")
assert merged.meta.total_episodes == len(kept_episodes), (merged.meta.total_episodes, len(kept_episodes))
if OUT.exists():
    shutil.rmtree(OUT)
final = remove_feature(
    merged,
    ["observation.images.head_left", "observation.images.head_right"],
    output_dir=OUT,
    repo_id=REPO_ID,
)
episode_tasks = {i: TASKS[by_source[src]["label"]] for i, src in enumerate(kept_episodes)}
final = modify_tasks(final, episode_tasks=episode_tasks)
fix_camera_names(OUT)

sources = []
for i, (name, ep) in enumerate(kept_episodes):
    v = by_source[(name, ep)]
    length = int(final.meta.episodes[i]["length"])
    assert length == v["length"], (i, name, ep, length, v["length"])  # merged episode i is that source episode
    sources.append({"episode_index": i, "dataset": name, "source_episode": ep, "recorded_at_ms": v["t_ms"],
                    "arm": {"l": "left", "r": "right"}[v["side"]], "task": TASKS[v["label"]], "length": length})  # fmt: skip
(OUT / "episode_sources.json").write_text(json.dumps(sources, indent=1))
shutil.rmtree(TMP)
print(f"{final.meta.total_episodes} episodes, {final.meta.total_frames} frames, tasks:")
print(final.meta.tasks)
print("features:", sorted(final.meta.features))
