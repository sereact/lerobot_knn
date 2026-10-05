"""Contact sheet: first and last frame of the given episodes (recording order index), for each camera."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.datasets import LeRobotDataset

ROOT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-gate-testing")
TMP = Path("/home/denis-office/.claude/jobs/34f24b0e/tmp")
episodes = json.load(open(TMP / "episodes.json"))
order = [int(i) for i in sys.argv[1].split(",")]
cams = sys.argv[2].split(",") if len(sys.argv) > 2 else ["head_left"]
out = Path(sys.argv[3]) if len(sys.argv) > 3 else TMP / "sheet.png"
W, H = 320, 240
sheet = Image.new("RGB", (W * 2 * len(cams), H * len(order)), "white")
cache: dict[str, LeRobotDataset] = {}
for row, i in enumerate(order):
    e = episodes[i]
    if e["dataset"] not in cache:
        cache[e["dataset"]] = LeRobotDataset(f"local/{e['dataset']}", root=ROOT / e["dataset"] / "original")
    ds = cache[e["dataset"]]
    start = int(ds.meta.episodes[e["episode"]]["dataset_from_index"])
    end = int(ds.meta.episodes[e["episode"]]["dataset_to_index"]) - 1
    for c, cam in enumerate(cams):
        for k, idx in enumerate((start, end)):
            img = ds[idx][f"observation.images.{cam}"].permute(1, 2, 0).numpy()
            tile = Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))
            ImageDraw.Draw(tile).text((5, 5), f"#{i} {cam} {'first' if k == 0 else 'last'}", fill=(255, 0, 0))
            sheet.paste(tile, ((2 * c + k) * W, row * H))
sheet.save(out)
print("saved", out)
