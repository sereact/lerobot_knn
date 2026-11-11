"""Tests of the helpers the experiment scripts share."""

import numpy as np
import torch

from lerobot_policy_knn.encoders import finalize_features, grid_pool, normalize_keys
from lerobot_policy_knn.experiments.common import action_chunks, parse_range, record_batch, space_name
from lerobot_policy_knn.features import FrameIndex
from lerobot_policy_knn.vla import DATASET_CAMERAS


def _frames() -> FrameIndex:
    """Two episodes of 3 and 2 frames; the action of frame i is [i]."""
    return FrameIndex(
        state=np.zeros((5, 8), dtype=np.float32),
        action=np.arange(5, dtype=np.float32)[:, None],
        episode_index=np.array([0, 0, 0, 1, 1]),
        frame_index=np.array([0, 1, 2, 0, 1]),
        task_index=np.zeros(5, dtype=np.int64),
        episode_end=np.array([2, 2, 2, 4, 4]),
        episode_length=np.array([3, 3, 3, 2, 2]),
        task_names=["task"],
    )


def test_action_chunks_hold_the_last_action_of_the_episode():
    chunks = action_chunks(_frames(), np.array([1, 3]), horizon=3)
    assert chunks.shape == (2, 3, 1)
    assert chunks[..., 0].tolist() == [[1, 2, 2], [3, 4, 4]]


def test_parse_range_and_space_name():
    assert list(parse_range("20:23")) == [20, 21, 22]
    assert space_name("vla:pi05_base", "grid2@L12", "zscore") == "vla:pi05_base/grid2@L12|zscore"
    assert space_name("vla:smolvla_base", "grid2") == "vla:smolvla_base/grid2"


def test_record_batch_scales_images_and_names_prompts():
    rec = {
        "images": np.full((3, 2, 3, 4, 4), 255, dtype=np.uint8),
        "state": np.ones((3, 8), dtype=np.float32),
        "task": np.array([1, 0, 1]),
        "task_names": np.array(["open the drawer", "close the drawer"]),
    }
    batch = record_batch(rec, slice(1, 3))
    assert set(batch) == {*DATASET_CAMERAS, "observation.state", "task"}
    assert batch[DATASET_CAMERAS[0]].shape == (2, 3, 4, 4)
    assert float(batch[DATASET_CAMERAS[1]].max()) == 1.0
    assert batch["task"] == ["open the drawer", "close the drawer"]


def test_grid_pool_square_and_rectangular():
    tokens = torch.arange(2 * 16 * 3, dtype=torch.float32).view(2, 16, 3)
    square = grid_pool(tokens, 2)
    assert square.shape == (2, 4, 3)
    # top-left cell of a 4x4 grid: tokens 0, 1, 4, 5
    assert torch.allclose(square[:, 0], tokens[:, [0, 1, 4, 5]].mean(1))
    wide = grid_pool(tokens, 2, shape=(2, 8))
    assert torch.allclose(wide[:, 0], tokens[:, :4].mean(1))


def test_normalize_keys_matches_finalize_without_statistics():
    raw = torch.randn(5, 4 * 6)
    assert torch.equal(normalize_keys(raw, "grid2", None, None, None), finalize_features(raw, "grid2"))
    mean, std = raw.mean(0), raw.std(0)
    keys = normalize_keys(raw, "cls", mean, std, torch.eye(24)[:, :8])
    assert keys.shape == (5, 8)
    assert torch.allclose(keys.norm(dim=-1), torch.ones(5), atol=1e-6)
