"""Fit the kNN on the tri-arm shoe-box dataset (wrist cameras + state, absolute TCP actions)."""

import logging
from pathlib import Path

from lerobot_policy_knn.configuration_knn import KNNConfig
from lerobot_policy_knn.fit import KNNFitConfig, fit

logging.basicConfig(level=logging.INFO)
BASE = Path("/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box-knn")
cfg = KNNFitConfig(
    repo_id="local/tri_arm_shoe_box",
    root="/home/denis-office/git/lerobot/outputs/tri-arm-shoe-box",
    output_dir=BASE / "policy",
    cache_dir=BASE / "feature_cache",
    batch_size=128,
    policy=KNNConfig(
        device="cuda",
        image_encoder="resnet18_l3",
        image_pooling="grid4",
        camera_weights={"observation.images.wrist1": 1.0, "observation.images.wrist2": 1.0},
        state_weight=1.0,
        state_axis_angle_start=None,  # state is xyz + rot6d + gripper per arm (+ wrenches), no axis-angle
        gripper_mode="mean",  # grippers are continuous position_normalized in [0, 1]
        task_conditioning="filter",
        time_weight=0.0,  # the policy server answers every request without episode-time state
        history_weight=0.0,
        k=8,
        temperature=0.1,
        one_per_episode=True,
        action_mode_bandwidth=0.05,
        chunk_size=30,
        n_action_steps=30,
    ),
)
print("saved to", fit(cfg))
