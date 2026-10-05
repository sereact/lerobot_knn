"""policy -> policy_relative: same memory (hard link), config with relative_action_dims = every non-gripper dim."""

import json
import os
import shutil
from pathlib import Path

BASE = Path("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box-knn")
src, dst = BASE / "policy", BASE / "policy_relative"
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir()
for f in src.iterdir():
    if f.name == "model.safetensors":
        os.link(f, dst / f.name)
    elif f.name != "config.json":
        shutil.copy(f, dst / f.name)
cfg = json.loads((src / "config.json").read_text())
assert cfg["state_axis_angle_start"] is None
cfg["relative_action_dims"] = [i for i in range(20) if i not in (9, 19)]  # xyz + rot6d of both arms
(dst / "config.json").write_text(json.dumps(cfg, indent=4))
print(sorted(p.name for p in dst.iterdir()), cfg["relative_action_dims"])
