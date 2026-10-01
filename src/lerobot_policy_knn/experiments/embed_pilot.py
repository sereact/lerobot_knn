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

from ..features import DEFAULT_CACHE_DIR, _cache_root, load_frame_index

CAMS = ("observation.images.image", "observation.images.image2")


def normalise(x: torch.Tensor, mode: str, cells: int) -> torch.Tensor:
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
    parser.add_argument("--out", default="outputs/vla/embed_pilot.json")
    args = parser.parse_args()
    torch.set_num_threads(16)
    f = load_frame_index("lerobot/libero")
    horizon, k = args.horizon, args.k
    act, end = torch.from_numpy(f.action), torch.from_numpy(f.episode_end)
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for d in args.dirs:
        root = _cache_root(DEFAULT_CACHE_DIR, "lerobot/libero") / d
        rows = np.load(root / "rows.npy")
        pos = {int(r): i for i, r in enumerate(rows)}
        q = rows[f.frame_index[rows] % 10 == 4]
        qi = torch.tensor([pos[int(r)] for r in q])
        ym = act[torch.minimum(torch.from_numpy(rows)[:, None] + torch.arange(horizon), end[rows][:, None])]
        yq = ym[qi]
        tm, tq = f.task_index[rows], f.task_index[q]
        em = torch.from_numpy(f.episode_index[rows])
        eq = em[qi]
        poolings = sorted({p.name.split(".")[-2] for p in root.glob("observation.images.image.*.npy")})
        results.setdefault(d, {})
        for pooling in poolings:
            raw = [torch.from_numpy(np.load(root / f"{c}.{pooling}.npy").astype(np.float32)) for c in CAMS]
            for mode in ("raw", "zscore", "pca256"):
                keys = [normalise(x.clone(), mode, 4) for x in raw]
                sse = sst = grip = 0.0
                for t in np.unique(tq):
                    mm = torch.from_numpy(np.nonzero(tm == t)[0])
                    qm = torch.from_numpy(np.nonzero(tq == t)[0])
                    dist = sum(2 - 2 * k[qi[qm]] @ k[mm].T for k in keys) / len(keys)
                    dist[eq[qm][:, None] == em[mm][None, :]] = float("inf")
                    dd, idx = dist.topk(k, largest=False)
                    w = torch.softmax(-dd / (0.1 * dd.mean(1, keepdim=True)), 1)
                    pred, y = (w[..., None, None] * ym[mm][idx]).sum(1), yq[qm]
                    sse += float(((pred[..., :6] - y[..., :6]) ** 2).sum())
                    sst += float(((y[..., :6] - y[..., :6].mean(0)) ** 2).sum())
                    grip += float((torch.sign(pred[..., 6]) == torch.sign(y[..., 6])).float().sum())
                r2, g = 1 - sse / sst, grip / (len(q) * horizon)
                results[d][f"{pooling}|{mode}"] = {"r2": r2, "gripper": g, "n_queries": len(q)}
                print(f"{d:28s} {pooling:14s} {mode:7s} R2={r2:.3f} grip={g:.3f}", flush=True)
            del raw
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
