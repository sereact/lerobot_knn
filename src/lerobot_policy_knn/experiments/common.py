"""Paths, constants and small helpers shared by the experiment scripts.

The scripts run from the repository root and read / write under ``outputs/`` (see docs/experiments/README.md).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE

from ..features import FrameIndex
from ..vla import DATASET_CAMERAS

REPO_ID = "lerobot/libero"
KNN_OUT = Path("outputs/knn")
VLA_OUT = Path("outputs/vla")
PLUS_OUT = Path("outputs/libero_plus")
# {"train": [...], "heldout": [...]}: demonstration episodes; the held-out 10% never enter a memory built on "train"
SPLIT = VLA_OUT / "split.json"

# LIBERO actions: 6 motion dimensions (translation, rotation) followed by the gripper command
MOTION = slice(0, 6)
GRIPPER = 6


def parse_range(spec: str) -> range:
    """``"lo:hi"`` -> ``range(lo, hi)`` (how the scripts take initial states)."""
    lo, hi = (int(x) for x in spec.split(":"))
    return range(lo, hi)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def suite_scores(summary: dict[str, Any], suites: Iterable[str]) -> str:
    """``"spatial=0.900 object=0.980 ..."`` for the suites a result summary holds."""
    return " ".join(f"{s.split('_')[1]}={summary[s]:.3f}" for s in suites if s in summary)


def action_chunks(frames: FrameIndex, rows: np.ndarray, horizon: int) -> np.ndarray:
    """(len(rows), horizon, A): the actions from each dataset row on, holding its episode's last action at the end."""
    index = np.minimum(rows[:, None] + np.arange(horizon), frames.episode_end[rows][:, None])
    return frames.action[index]


def record_batch(rec: dict, rows: slice, device: str | None = None) -> dict[str, Any]:
    """Dataset-format observations of recorded steps.

    ``rec`` holds what ``VLARollout`` records: uint8 ``images`` (N, cameras, 3, H, W), ``state`` and the prompt of
    each step as an index (``task``) into ``task_names``.
    """

    def tensor(array: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(array)
        return t.to(device) if device is not None else t

    batch: dict[str, Any] = {
        cam: tensor(rec["images"][rows, i]).float() / 255.0 for i, cam in enumerate(DATASET_CAMERAS)
    }
    batch[OBS_STATE] = tensor(rec["state"][rows]).float()
    batch["task"] = [str(rec["task_names"][i]) for i in rec["task"][rows]]
    return batch


def extra_keys() -> list[dict]:
    """Backbone keys added after the layer pilots (``outputs/vla/extra_keys.json``, written by ``choose_keys``)."""
    path = VLA_OUT / "extra_keys.json"
    return json.loads(path.read_text())["keys"] if path.exists() else []


def space_name(encoder: str, pooling: str, key_norm: str = "none") -> str:
    """``<encoder>/<pooling>[|<key_norm>]``: how ``head_vs_knn.space_config`` names a VLM key space."""
    return f"{encoder}/{pooling}" + (f"|{key_norm}" if key_norm != "none" else "")
