"""Which layer / normalisation of a VLM makes the best lookup key? Offline sweep on a task subset (CPU).

Reads the ``*_pilot`` caches written by ``vla_features --tasks=... --tag_suffix=_pilot --taps=...``. For every model,
tap (``grid2`` = final layer, ``grid2@L<n>``, ``grid2@vision``) and normalisation (``raw`` cosine, per-dimension
``zscore``, ``pca256`` whitening), it predicts the action chunk of held-out demonstration frames with the plain
kNN (same task, other demos, k = 8) and reports R² of the motion dimensions and gripper agreement.

    uv run python -m lerobot_policy_knn.experiments.embed_pilot vla_smolvla_base_pilot vla_pi05_libero_pilot ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..features import DEFAULT_CACHE_DIR, cache_root, load_frame_index
from ..vla import DATASET_CAMERAS
from .common import GRIPPER, MOTION, REPO_ID, VLA_OUT, action_chunks

CELLS = 4  # a grid2 key: 2x2 cells per camera


def normalise(x: torch.Tensor, mode: str, cells: int) -> torch.Tensor:
    """Unit-norm keys from raw ``grid2`` features: per-cell cosine (``raw``), after a per-dimension ``zscore``, or a
    whitened 256-D PCA of the whole vector (``pca256``)."""
    if mode == "pca256":
        flat = x.reshape(len(x), -1)
        flat = flat - flat.mean(0)
        sub = flat[torch.randperm(len(flat))[:20000]]
        _, s, v = torch.pca_lowrank(sub, q=256, center=False)
        return torch.nn.functional.normalize(flat @ v / (s / (len(sub) - 1) ** 0.5), dim=-1)
    x = x.view(len(x), cells, -1)
    if mode == "zscore":
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-3)
    return torch.nn.functional.normalize(x, dim=-1).reshape(len(x), -1) / cells**0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--out", default=str(VLA_OUT / "embed_pilot.json"))
    args = parser.parse_args()
    torch.set_num_threads(16)
    frames = load_frame_index(REPO_ID)
    horizon, k = args.horizon, args.k
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for name in args.dirs:
        root = cache_root(DEFAULT_CACHE_DIR, REPO_ID) / name
        rows = np.load(root / "rows.npy")  # the cached (memory) frames
        position = {int(r): i for i, r in enumerate(rows)}
        queries = rows[frames.frame_index[rows] % 10 == 4]
        qi = torch.tensor([position[int(r)] for r in queries])
        chunks = torch.from_numpy(action_chunks(frames, rows, horizon))
        target = chunks[qi]
        task, query_task = frames.task_index[rows], frames.task_index[queries]
        episode = torch.from_numpy(frames.episode_index[rows])
        poolings = sorted({p.name.split(".")[-2] for p in root.glob(f"{DATASET_CAMERAS[0]}.*.npy")})
        results.setdefault(name, {})
        for pooling in poolings:
            raw = [
                torch.from_numpy(np.load(root / f"{cam}.{pooling}.npy").astype(np.float32))
                for cam in DATASET_CAMERAS
            ]
            for mode in ("raw", "zscore", "pca256"):
                keys = [normalise(x.clone(), mode, CELLS) for x in raw]
                sse = sst = gripper_hits = 0.0
                for t in np.unique(query_task):
                    mem = torch.from_numpy(np.nonzero(task == t)[0])
                    qs = torch.from_numpy(np.nonzero(query_task == t)[0])
                    dist = sum(2 - 2 * key[qi[qs]] @ key[mem].T for key in keys) / len(keys)
                    dist[episode[qi[qs]][:, None] == episode[mem][None, :]] = float("inf")  # other demos only
                    nearest, idx = dist.topk(k, largest=False)
                    weights = torch.softmax(-nearest / (0.1 * nearest.mean(1, keepdim=True)), 1)
                    pred, y = (weights[..., None, None] * chunks[mem][idx]).sum(1), target[qs]
                    sse += float(((pred[..., MOTION] - y[..., MOTION]) ** 2).sum())
                    sst += float(((y[..., MOTION] - y[..., MOTION].mean(0)) ** 2).sum())
                    gripper_hits += float(
                        (torch.sign(pred[..., GRIPPER]) == torch.sign(y[..., GRIPPER])).float().sum()
                    )
                r2, gripper = 1 - sse / sst, gripper_hits / (len(queries) * horizon)
                results[name][f"{pooling}|{mode}"] = {"r2": r2, "gripper": gripper, "n_queries": len(queries)}
                print(f"{name:28s} {pooling:14s} {mode:7s} R2={r2:.3f} grip={gripper:.3f}", flush=True)
            del raw
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
