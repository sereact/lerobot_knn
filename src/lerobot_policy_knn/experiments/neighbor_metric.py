"""Do close embeddings have close actions? A local embedding -> action consistency metric, and how it predicts kNN.

For query frames of held-out demonstrations (``frame_index % 10 == 4``), candidates are the memory frames (even
``frame_index``) of the same task from *other* demonstrations. With ``d_e`` the retrieval distance of a key space and
``d_a`` the mean squared difference of the 10-step motion chunks that follow two frames:

* ``lac@k`` (local action consistency) = 1 - mean d_a(query, its k nearest in d_e) / mean d_a(query, random candidate)
  -- only close neighbours count; 1 = neighbours act identically, 0 = no better than a random frame of the task
* ``local_rho`` = Spearman correlation of d_e and d_a within the 64 nearest (does "closer" still mean "more alike")
* ``global_rho`` = the same over all candidates (dominated by far-apart pairs, which a lookup never uses)

Every key space is evaluated on the same queries (CPU); ``--correlate`` relates the metrics to the closed-loop success
of the plain lookup with that key (outputs/knn/compare.jsonl), over all keys and over the *fair* keys, those whose
weights never saw LIBERO actions for the query demos (the fine-tuned VLAs were trained on every demonstration, which
inflates their offline neighbours). ``--heldout`` draws queries only from the held-out demos of outputs/vla/split.json,
which makes the learnt key (trained on the other 90%) fair as well.

    uv run python -m lerobot_policy_knn.experiments.neighbor_metric --correlate [--heldout]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..features import DEFAULT_CACHE_DIR, load_features, load_frame_index
from ..memory import fit_key_norm, normalize_keys

CAMS = ("observation.images.image", "observation.images.image2")
# key space -> (encoder, pooling, key_norm, closed-loop config stem of the plain lookup with that key)
SPACES = {
    "ResNet-18 grid": ("resnet18_l3", "grid4", "none", "plain_resnet18_l3"),
    "DINOv2-S": ("dinov2_small", "cls", "none", "plain_dinov2"),
    "SmolVLA base grid": ("vla:smolvla_base", "grid2", "none", "plain_smolvla_base_grid2"),
    "SmolVLA base mean": ("vla:smolvla_base", "img", "none", "plain_smolvla_base_img"),
    "SmolVLA LIBERO grid": ("vla:smolvla_libero", "grid2", "none", "plain_smolvla_libero_grid2"),
    "SmolVLA LIBERO mean": ("vla:smolvla_libero", "img", "none", "plain_smolvla_libero_img"),
    "π0.5 base last": ("vla:pi05_base", "grid2", "zscore", "plain_pi05_base_grid2_z"),
    "π0.5 base L12": ("vla:pi05_base", "grid2@L12", "zscore", "plain_pi05_base_grid2@L12_z"),
    "π0.5 base L15": ("vla:pi05_base", "grid2@L15", "zscore", "plain_pi05_base_grid2@L15_z"),
    "π0.5 LIBERO last raw": ("vla:pi05_libero", "grid2", "none", "plain_pi05_libero_grid2"),
    "π0.5 LIBERO last": ("vla:pi05_libero", "grid2", "zscore", "plain_pi05_libero_grid2_z"),
    "π0.5 LIBERO L12": ("vla:pi05_libero", "grid2@L12", "zscore", "plain_pi05_libero_grid2@L12_z"),
    "π0.5 LIBERO L15": ("vla:pi05_libero", "grid2@L15", "zscore", "plain_pi05_libero_grid2@L15_z"),
    "π0.5 base L12 PCA": ("vla:pi05_base", "grid2@L12", "zscore_pca512", "plain_pi05_base_grid2@L12_pca"),
    "π0.5 base L12+L15+last PCA": ("vla:pi05_base", "grid2@L12+L15+last", "zscore_pca512",
                                   "plain_pi05_base_grid2@L12+L15+last_pca"),
    "π0.5 LIBERO L12 PCA": ("vla:pi05_libero", "grid2@L12", "zscore_pca512", "plain_pi05_libero_grid2@L12_pca"),
    "π0.5 LIBERO L12+L15+last PCA": ("vla:pi05_libero", "grid2@L12+L15+last", "zscore_pca512",
                                     "plain_pi05_libero_grid2@L12+L15+last_pca"),
    "ResNet-18 grid (learnt key)": ("resnet18_l3", "grid4", "file:outputs/vla/learned_key_resnet.pt",
                                   "plain_resnet18_l3_learned"),
}  # fmt: skip
# keys whose weights never saw LIBERO actions (the learnt key only for held-out queries, see --heldout)
NEVER_TRAINED_ON_LIBERO = (
    "ResNet-18",
    "DINOv2",
    "SmolVLA base",
    "π0.5 base",
)


def extra_keys() -> list[dict]:
    """Backbone keys added after the layer pilots (outputs/vla/extra_keys.json, written by choose_keys)."""
    path = Path("outputs/vla/extra_keys.json")
    return json.loads(path.read_text())["keys"] if path.exists() else []


for _k in extra_keys():
    SPACES[_k["label"]] = (_k["encoder"], _k["pooling"], _k["norm"], _k["stem"])
    if _k["fair"]:
        NEVER_TRAINED_ON_LIBERO = (*NEVER_TRAINED_ON_LIBERO, _k["label"])


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of (Q, M) tensors."""
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-8)


def fit_norms(enc: str, pooling: str, norm: str, rows: np.ndarray) -> list[tuple]:
    """Per camera, the key normalisation of the plain lookup (fitted on its memory ``rows``, as ``build_memory``)."""
    out = []
    for cam in CAMS:
        arr = load_features(DEFAULT_CACHE_DIR, "lerobot/libero", enc, cam, pooling)
        rows_of = lambda part, arr=arr: torch.from_numpy(np.asarray(arr[part]))  # noqa: E731
        out.append(fit_key_norm(norm, cam, rows, rows_of, "cpu", chunk=8192))
    return out


def key_features(enc: str, pooling: str, rows: np.ndarray, norms: list[tuple]) -> list[torch.Tensor]:
    out = []
    for cam, norm in zip(CAMS, norms, strict=True):
        x = torch.from_numpy(
            np.asarray(load_features(DEFAULT_CACHE_DIR, "lerobot/libero", enc, cam, pooling)[rows])
        )
        out.append(normalize_keys(x.float(), pooling, *norm))
    return out


@torch.no_grad()
def evaluate(space: str, f, horizon: int = 10, ks=(1, 8), local_m: int = 64, queries_per_task: int = 120,
             query_episodes: np.ndarray | None = None) -> dict:  # fmt: skip
    enc, pooling, norm, _ = SPACES[space]
    mem_all = np.nonzero(f.frame_index % 2 == 0)[0]
    norms = fit_norms(enc, pooling, norm, mem_all)
    act = torch.from_numpy(f.action)
    end = torch.from_numpy(f.episode_end)
    rng = np.random.default_rng(0)
    lac = {k: [] for k in ks}
    local_rho, global_rho = [], []
    for t in np.unique(f.task_index):
        cand = mem_all[f.task_index[mem_all] == t]
        q_all = cand[f.frame_index[cand] % 10 == 4]
        if query_episodes is not None:
            q_all = q_all[np.isin(f.episode_index[q_all], query_episodes)]
        q = np.sort(rng.choice(q_all, min(queries_per_task, len(q_all)), replace=False))
        kc = key_features(enc, pooling, cand, norms)
        pos = {int(r): i for i, r in enumerate(cand)}
        qi = torch.tensor([pos[int(r)] for r in q])
        d_e = sum(2 - 2 * k[qi] @ k.T for k in kc) / len(kc)  # (Q, C)
        ya = act[torch.minimum(torch.from_numpy(cand)[:, None] + torch.arange(horizon), end[cand][:, None])][
            ..., :6
        ]
        ya = ya.reshape(len(cand), -1)
        d_a = torch.cdist(ya[qi], ya).pow(2) / ya.shape[1]  # (Q, C) mean squared motion difference
        other = (
            torch.from_numpy(f.episode_index[cand])[None, :] != torch.from_numpy(f.episode_index[q])[:, None]
        )
        d_e = d_e.masked_fill(~other, float("inf"))
        rand = (d_a * other).sum(1) / other.sum(1)
        order = d_e.argsort(1)
        for k in ks:
            nn = d_a.gather(1, order[:, :k]).mean(1)
            lac[k].append((1 - nn / rand.clamp_min(1e-9)).numpy())
        top = order[:, :local_m]
        local_rho.append(spearman(d_e.gather(1, top), d_a.gather(1, top)).numpy())
        sub = torch.stack(
            [torch.from_numpy(rng.choice(np.nonzero(o.numpy())[0], 256, replace=False)) for o in other]
        )
        global_rho.append(spearman(d_e.gather(1, sub), d_a.gather(1, sub)).numpy())
    res = {f"lac@{k}": float(np.concatenate(v).mean()) for k, v in lac.items()}
    res["local_rho"] = float(np.concatenate(local_rho).mean())
    res["global_rho"] = float(np.concatenate(global_rho).mean())
    return res


def correlate(results: dict, heldout: bool) -> dict:
    runs = {}
    for line in Path("outputs/knn/compare.jsonl").read_text().splitlines():
        r = json.loads(line)
        runs[Path(r["config_file"]).stem] = 100 * r["success"]
    rows = [(name, m, runs[SPACES[name][3]]) for name, m in results.items() if SPACES[name][3] in runs]

    def fair(name: str) -> bool:
        return name.startswith(NEVER_TRAINED_ON_LIBERO) and ("learnt" not in name or heldout)

    out: dict = {"points": {n: {"success": s, "fair": fair(n), **m} for n, m, s in rows}}
    for subset, members in (("all", rows), ("fair", [r for r in rows if fair(r[0])])):
        out[subset] = {"n": len(members)}
        if len(members) < 4:
            continue
        succ = torch.tensor([r[2] for r in members])
        for metric in members[0][1]:
            vals = torch.tensor([r[1][metric] for r in members])
            out[subset][f"spearman_{metric}"] = float(spearman(vals[None], succ[None])[0])
            out[subset][f"pearson_{metric}"] = float(np.corrcoef(vals.numpy(), succ.numpy())[0, 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default=",".join(SPACES))
    parser.add_argument("--correlate", action="store_true")
    parser.add_argument(
        "--heldout", action="store_true", help="queries from the held-out demos of the split only"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    f = load_frame_index("lerobot/libero")
    query_episodes = (
        np.array(json.loads(Path("outputs/vla/split.json").read_text())["heldout"]) if args.heldout else None
    )
    out_path = Path(args.out or f"outputs/vla/neighbor_metric{'_heldout' if args.heldout else ''}.json")
    results = json.loads(out_path.read_text())["spaces"] if out_path.exists() else {}
    results = {k: v for k, v in results.items() if k in SPACES}  # spaces dropped from the study
    for space in args.spaces.split(","):
        if space in results:
            continue
        try:
            results[space] = evaluate(space, f, query_episodes=query_episodes)
        except FileNotFoundError as e:
            print(f"skip {space}: {e}")
            continue
        print(space, {k: round(v, 3) for k, v in results[space].items()}, flush=True)
        out_path.write_text(json.dumps({"spaces": results}, indent=1))
    summary = {"spaces": results}
    if args.correlate:
        summary["correlation"] = correlate(results, args.heldout)
        print(json.dumps({k: v for k, v in summary["correlation"].items() if k != "points"}, indent=1))
    out_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
