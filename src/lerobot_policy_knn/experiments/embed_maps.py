"""What does an embedding space consider "near"? Phase agreement of neighbours, and a 2-D map of one task (CPU).

For frames of every task, the nearest frame from *another* demonstration of the same task (cosine distance on
the retrieval keys) should be at the same phase of the task (progress = frame / episode length) if the space
encodes what the robot should do next. Reports the median |progress difference| per space, and a t-SNE map of
one task's frames coloured by progress for the poster.

    uv run python -m lerobot_policy_knn.experiments.embed_maps \\
        --spaces=dinov2,resnet18_l3,vla:smolvlm2/img,vla:smolvla_base/img,vla:smolvla_libero/img
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..features import DEFAULT_CACHE_DIR, load_features, load_frame_index
from ..vla import DATASET_CAMERAS, VLA_CAMERA

IMAGE_SPACES = {"dinov2": ("dinov2_small", "cls"), "resnet18_l3": ("resnet18_l3", "grid4")}


def keys(space: str, rows: np.ndarray) -> np.ndarray:
    if space.startswith("vla:"):
        enc, pooling = space.rsplit("/", 1)
        cams = list(DATASET_CAMERAS) if pooling in ("img", "grid2") else [VLA_CAMERA]
    else:
        (enc, pooling), cams = IMAGE_SPACES[space], list(DATASET_CAMERAS)
    parts = []
    for cam in cams:
        x = np.asarray(load_features(DEFAULT_CACHE_DIR, "lerobot/libero", enc, cam, pooling)[rows]).astype(
            np.float32
        )
        if pooling.startswith("grid"):
            cells = int(pooling[4:]) ** 2
            x = x.reshape(len(x), cells, -1)
            x = (x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)).reshape(len(x), -1) / cells**0.5
        else:
            x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
        parts.append(x / len(cams) ** 0.5)
    return np.concatenate(parts, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", required=True)
    parser.add_argument("--map_task", type=int, default=0, help="dataset task index for the t-SNE map")
    parser.add_argument("--map_demos", type=int, default=16)
    parser.add_argument("--out", default="outputs/vla/embed_maps.json")
    args = parser.parse_args()
    f = load_frame_index("lerobot/libero")
    progress = f.frame_index / np.maximum(f.episode_length - 1, 1)
    rng = np.random.default_rng(0)
    # queries: every 10th (even) frame of every episode; candidates: even frames of the same task
    even = np.nonzero(f.frame_index % 2 == 0)[0]
    queries = np.nonzero(f.frame_index % 10 == 4)[0]
    queries = np.sort(rng.choice(queries, 8000, replace=False))
    map_eps = rng.choice(
        np.unique(f.episode_index[f.task_index == args.map_task]), args.map_demos, replace=False
    )
    map_rows = even[np.isin(f.episode_index[even], map_eps)][::2]
    out: dict = {"task": f.task_names[args.map_task], "map_progress": progress[map_rows].tolist(),
                 "map_episode": f.episode_index[map_rows].tolist(), "spaces": {}}  # fmt: skip
    for space in args.spaces.split(","):
        errs = []
        for t in np.unique(f.task_index):
            cand = even[f.task_index[even] == t]
            q = queries[f.task_index[queries] == t]
            kc, kq = keys(space, cand), keys(space, q)
            sim = kq @ kc.T
            sim[f.episode_index[q][:, None] == f.episode_index[cand][None, :]] = -np.inf
            nn = cand[sim.argmax(1)]
            errs.append(np.abs(progress[q] - progress[nn]))
        errs = np.concatenate(errs)
        from sklearn.manifold import TSNE

        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
            keys(space, map_rows)
        )
        out["spaces"][space] = {"phase_err_median": float(np.median(errs)), "phase_err_mean": float(errs.mean()),
                                "map_xy": xy.round(3).tolist()}  # fmt: skip
        print(
            space,
            "median |progress diff| of nearest other-demo frame:",
            round(float(np.median(errs)), 4),
            flush=True,
        )
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
