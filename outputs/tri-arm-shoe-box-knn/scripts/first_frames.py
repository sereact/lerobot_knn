"""Grid sheets of each episode's first frame from the active arm's wrist camera (recording-order index on each tile)."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.datasets import LeRobotDataset

ROOT = Path("/home/denis-office/git/lerobot/outputs/tri-arm-gate-testing")
TMP = Path("/home/denis-office/.claude/jobs/34f24b0e/tmp")
episodes = json.load(open(TMP / "episodes.json"))
W, H, COLS, ROWS = 320, 240, 6, 4
tiles = []
cache: dict[str, LeRobotDataset] = {}
for i, e in enumerate(episodes):
    if e["date"] != "2026-09-30":
        continue
    xl, xr = np.array(e["xyz_l"]), np.array(e["xyz_r"])
    cam = "wrist1" if np.ptp(xl, axis=0).max() > np.ptp(xr, axis=0).max() else "wrist2"
    if e["dataset"] not in cache:
        cache[e["dataset"]] = LeRobotDataset(f"local/{e['dataset']}", root=ROOT / e["dataset"] / "original")
    ds = cache[e["dataset"]]
    start = int(ds.meta.episodes[e["episode"]]["dataset_from_index"])
    img = ds[start][f"observation.images.{cam}"].permute(1, 2, 0).numpy()
    tile = Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 70, 22), fill=(0, 0, 0))
    draw.text((4, 4), f"#{i} {cam[-1]}", fill=(255, 255, 0))
    tiles.append(tile)
per = COLS * ROWS
for s in range(0, len(tiles), per):
    sheet = Image.new("RGB", (W * COLS, H * ROWS), "white")
    for k, tile in enumerate(tiles[s : s + per]):
        sheet.paste(tile, ((k % COLS) * W, (k // COLS) * H))
    sheet.save(TMP / f"first_{s // per}.png")
print("sheets:", (len(tiles) + per - 1) // per)
