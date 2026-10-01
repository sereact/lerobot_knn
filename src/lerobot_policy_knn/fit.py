"""``lerobot-knn-fit``: embed a LeRobot dataset and save a ready-to-evaluate KNN policy.

Example::

    lerobot-knn-fit --repo_id=lerobot/libero --output_dir=outputs/knn/libero_dinov2 \\
        --policy.image_encoder=dinov2_small --policy.k=8 --policy.chunk_size=16

    lerobot-eval --policy.path=outputs/knn/libero_dinov2 --env.type=libero --env.task=libero_spatial
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.utils.feature_utils import dataset_to_policy_features

from .configuration_knn import KNNConfig
from .features import DEFAULT_CACHE_DIR, extract_features
from .memory import make_policy_from_cache
from .processor_knn import make_knn_pre_post_processors

logger = logging.getLogger(__name__)


@dataclass
class KNNFitConfig:
    repo_id: str = "lerobot/libero"
    root: str | None = None
    output_dir: Path = Path("outputs/knn/policy")
    cache_dir: Path = DEFAULT_CACHE_DIR
    # Restrict the memory to these episodes (default: all).
    episodes: list[int] | None = None
    batch_size: int = 256
    policy: KNNConfig = field(default_factory=KNNConfig)


def fit(cfg: KNNFitConfig) -> Path:
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root)
    policy_cfg = cfg.policy
    features = dataset_to_policy_features(meta.features)
    policy_cfg.input_features = {k: v for k, v in features.items() if k.startswith("observation.")}
    policy_cfg.output_features = {k: v for k, v in features.items() if k == "action"}
    policy_cfg.validate_features()

    cameras = policy_cfg.active_cameras
    if cameras:
        extract_features(
            cfg.repo_id,
            [policy_cfg.image_encoder],
            cameras=cameras,
            root=cfg.root,
            cache_dir=cfg.cache_dir,
            resolution=policy_cfg.image_resolution,
            batch_size=cfg.batch_size,
            device=policy_cfg.device or "cuda",
            poolings={policy_cfg.image_encoder: (policy_cfg.image_pooling,)},
        )
    episodes = np.asarray(cfg.episodes) if cfg.episodes is not None else None
    policy = make_policy_from_cache(policy_cfg, cfg.repo_id, cfg.cache_dir, episodes=episodes)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(cfg.output_dir)
    preprocessor, postprocessor = make_knn_pre_post_processors(policy_cfg)
    preprocessor.save_pretrained(cfg.output_dir)
    postprocessor.save_pretrained(cfg.output_dir)
    logger.info("Saved KNN policy with %d memory keys to %s", policy.memory_size, cfg.output_dir)
    return cfg.output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fit(draccus.parse(KNNFitConfig))


if __name__ == "__main__":
    main()
