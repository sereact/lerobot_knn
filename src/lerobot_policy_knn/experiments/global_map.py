"""2-D map (PCA + t-SNE) of the whole memory across all tasks, in the policy's retrieval metric.

uv run python -m lerobot_policy_knn.experiments.global_map --config=outputs/knn/best_v1.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..configuration_knn import KNNConfig
from ..modeling_knn import camera_key_name
from .search import MemoryFactory
from .visualize import OUT_DIR, metric_space

SUITE_OF_TASK_BLOCK = {0: "libero_10", 1: "libero_goal", 2: "libero_object", 3: "libero_spatial"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--per_task", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cfg = KNNConfig(device="cuda", **json.loads(Path(args.config).read_text()))
    factory = MemoryFactory()
    policy = factory.build(cfg)
    frames = factory.frames
    rng = np.random.default_rng(0)
    offsets = policy.task_offsets.cpu().numpy()
    rows = np.concatenate(
        [
            np.sort(
                rng.choice(
                    np.arange(offsets[t], offsets[t + 1]),
                    min(args.per_task, offsets[t + 1] - offsets[t]),
                    replace=False,
                )
            )
            for t in range(len(offsets) - 1)
        ]
    )
    names = [camera_key_name(c) for c in cfg.active_cameras] + (["key_state"] if cfg.state_weight > 0 else [])
    with torch.no_grad():
        x = metric_space(
            policy, {n: getattr(policy, n)[torch.from_numpy(rows).to(policy.device)] for n in names}
        )
    x = x.cpu().numpy()
    reduced = PCA(n_components=50).fit_transform(x)
    tsne = TSNE(n_components=2, perplexity=50, init="pca", random_state=0, n_jobs=args.threads).fit_transform(
        reduced
    )
    ptr = policy.key_ptr[torch.from_numpy(rows).to(policy.device)].cpu().numpy()
    task = frames.task_index[ptr]
    progress = frames.frame_index[ptr] / np.maximum(frames.episode_length[ptr] - 1, 1)
    out = {
        "config": json.loads(Path(args.config).read_text()),
        "task_names": cfg.task_names,
        "suite_of_task": [SUITE_OF_TASK_BLOCK[t // 10] for t in range(len(cfg.task_names))],
        "points": [
            [
                round(float(tx), 2),
                round(float(ty), 2),
                round(float(px), 3),
                round(float(py), 3),
                int(tk),
                round(float(pr), 2),
            ]
            for (tx, ty), (px, py), tk, pr in zip(tsne, reduced[:, :2], task, progress, strict=True)
        ],
        "columns": ["tsne_x", "tsne_y", "pca_x", "pca_y", "task", "progress"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "global_map.json").write_text(json.dumps(out))
    print("wrote", OUT_DIR / "global_map.json", len(rows), "points")


if __name__ == "__main__":
    main()
