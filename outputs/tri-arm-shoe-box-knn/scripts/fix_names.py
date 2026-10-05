"""Apply build_shoe_dataset.fix_camera_names to the already built dataset."""

import json
from pathlib import Path

from lerobot.utils.feature_utils import dataset_to_policy_features

path = Path("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box/meta/info.json")
info = json.loads(path.read_text())
for key, ft in info["features"].items():
    if ft["dtype"] in ("video", "image") and len(ft.get("names") or []) != 3:
        print(key, ft["names"], "->", ["channels", "height", "width"])
        ft["names"] = ["channels", "height", "width"]
path.write_text(json.dumps(info, indent=4))
print({k: (v.type.value, v.shape) for k, v in dataset_to_policy_features(info["features"]).items()})
