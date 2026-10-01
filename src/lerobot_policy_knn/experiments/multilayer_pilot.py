"""Use every layer of a VLM as the lookup key? Offline sweep over layer combinations on the pilot caches (CPU).

pi0.5's action expert attends to the keys/values of *every* backbone layer. Here each tapped layer (``grid2@vision``,
``grid2@L3`` ... ``grid2``) becomes a z-scored, unit-norm block and the retrieval distance is a weighted sum of
per-layer cosine distances, ``d = sum_l w_l d_l``. Per-layer distance matrices are computed once per task, so any
weighting is cheap to score (held-out action R², k = 8). Weights are fitted on half of the pilot tasks by a Dirichlet
random search and scored on the other half.

    uv run python -m lerobot_policy_knn.experiments.multilayer_pilot vla_pi05_base_pilot vla_pi05_libero_pilot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..features import DEFAULT_CACHE_DIR, _cache_root, load_frame_index

CAMS = ("observation.images.image", "observation.images.image2")


def layer_distances(
    root: Path, pooling: str, rows: np.ndarray, q_mask: np.ndarray, tasks: np.ndarray, f
) -> dict:
    """task -> (Q, C) mean-over-cameras cosine distance of z-scored, per-cell unit-norm features."""
    out = {}
    feats = []
    for cam in CAMS:
        x = torch.from_numpy(np.load(root / f"{cam}.{pooling}.npy").astype(np.float32))
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-3)
        x = torch.nn.functional.normalize(x.view(len(x), 4, -1), dim=-1).reshape(len(x), -1) / 2.0
        feats.append(x)
    for t in np.unique(tasks):
        c = torch.from_numpy(np.nonzero(tasks == t)[0])
        q = torch.from_numpy(np.nonzero((tasks == t) & q_mask)[0])
        d = sum(2 - 2 * x[q] @ x[c].T for x in feats) / len(feats)
        same = (
            torch.from_numpy(f.episode_index[rows[q]])[:, None]
            == torch.from_numpy(f.episode_index[rows[c]])[None, :]
        )
        out[int(t)] = d.masked_fill(same, float("inf"))
    return out


def score(dists: list[dict], w: np.ndarray, tasks_subset, ym: dict, yq: dict, k: int = 8) -> float:
    sse = sst = 0.0
    for t in tasks_subset:
        d = sum(wi * dl[t] for wi, dl in zip(w, dists, strict=True) if wi > 0)
        dd, idx = d.topk(k, largest=False)
        wts = torch.softmax(-dd / (0.1 * dd.mean(1, keepdim=True)), 1)
        pred, y = (wts[..., None] * ym[t][idx]).sum(1), yq[t]
        sse += float(((pred - y) ** 2).sum())
        sst += float(((y - y.mean(0)) ** 2).sum())
    return 1 - sse / sst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--out", default="outputs/vla/multilayer_pilot.json")
    args = parser.parse_args()
    torch.set_num_threads(16)
    f = load_frame_index("lerobot/libero")
    act, end = torch.from_numpy(f.action), torch.from_numpy(f.episode_end)
    results = {}
    for d in args.dirs:
        root = _cache_root(DEFAULT_CACHE_DIR, "lerobot/libero") / d
        rows = np.load(root / "rows.npy")
        tasks = f.task_index[rows]
        q_mask = f.frame_index[rows] % 10 == 4
        chunks = act[torch.minimum(torch.from_numpy(rows)[:, None] + torch.arange(10), end[rows][:, None])][
            ..., :6
        ]
        chunks = chunks.reshape(len(rows), -1)
        ym = {int(t): chunks[torch.from_numpy(np.nonzero(tasks == t)[0])] for t in np.unique(tasks)}
        yq = {
            int(t): chunks[torch.from_numpy(np.nonzero((tasks == t) & q_mask)[0])] for t in np.unique(tasks)
        }
        poolings = sorted({p.name.split(".")[-2] for p in root.glob("observation.images.image.*.npy")},
                          key=lambda p: (-1 if p.endswith("vision") else int(p.split("@L")[1]) if "@L" in p else 99))  # fmt: skip
        dists = [layer_distances(root, p, rows, q_mask, tasks, f) for p in poolings]
        all_tasks = sorted(ym)
        fit, val = all_tasks[::2], all_tasks[1::2]
        res = {"poolings": poolings, "single": {}, "combos": {}}
        for i, p in enumerate(poolings):
            w = np.eye(len(poolings))[i]
            res["single"][p] = {
                "all": score(dists, w, all_tasks, ym, yq),
                "val": score(dists, w, val, ym, yq),
            }
        named = {"all layers (equal)": np.ones(len(poolings)),
                 "middle (L6, L9, L12)": np.array([p in ("grid2@L6", "grid2@L9", "grid2@L12") for p in poolings], float),
                 "upper (L12, L15, last)": np.array([p in ("grid2@L12", "grid2@L15", "grid2") for p in poolings], float),
                 "L12 + L15": np.array([p in ("grid2@L12", "grid2@L15") for p in poolings], float)}  # fmt: skip
        for name, w in named.items():
            res["combos"][name] = {
                "all": score(dists, w, all_tasks, ym, yq),
                "val": score(dists, w, val, ym, yq),
            }
        rng = np.random.default_rng(0)
        best_w, best_fit = None, -1e9
        for _ in range(args.trials):
            w = rng.dirichlet(np.full(len(poolings), 0.5))
            s = score(dists, w, fit, ym, yq)
            if s > best_fit:
                best_fit, best_w = s, w
        res["learned"] = {"weights": dict(zip(poolings, best_w.round(3).tolist(), strict=True)), "fit": best_fit,
                          "val": score(dists, best_w, val, ym, yq), "all": score(dists, best_w, all_tasks, ym, yq)}  # fmt: skip
        results[d] = res
        print(d)
        for p, v in res["single"].items():
            print(f"  single {p:14s} all={v['all']:.3f} val={v['val']:.3f}")
        for n, v in res["combos"].items():
            print(f"  combo  {n:24s} all={v['all']:.3f} val={v['val']:.3f}")
        print(
            f"  learned {res['learned']['weights']} fit={best_fit:.3f} val={res['learned']['val']:.3f}",
            flush=True,
        )
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
