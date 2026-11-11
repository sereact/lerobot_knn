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

from ..encoders import normalize_keys
from ..features import DEFAULT_CACHE_DIR, FrameIndex, load_features, load_frame_index
from ..memory import fit_key_norm
from ..vla import DATASET_CAMERAS
from .common import KNN_OUT, MOTION, REPO_ID, SPLIT, VLA_OUT, action_chunks, extra_keys, read_jsonl

LEARNT_KEY = "file:outputs/vla/learned_key_resnet.pt"
PI05_MULTILAYER = "grid2@L12+L15+last"
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
    "π0.5 base L12+L15+last PCA": (
        "vla:pi05_base",
        PI05_MULTILAYER,
        "zscore_pca512",
        f"plain_pi05_base_{PI05_MULTILAYER}_pca",
    ),
    "π0.5 LIBERO L12 PCA": (
        "vla:pi05_libero",
        "grid2@L12",
        "zscore_pca512",
        "plain_pi05_libero_grid2@L12_pca",
    ),
    "π0.5 LIBERO L12+L15+last PCA": (
        "vla:pi05_libero",
        PI05_MULTILAYER,
        "zscore_pca512",
        f"plain_pi05_libero_{PI05_MULTILAYER}_pca",
    ),
    "ResNet-18 grid (learnt key)": ("resnet18_l3", "grid4", LEARNT_KEY, "plain_resnet18_l3_learned"),
    "π0 base L12": ("vla:pi0_base", "grid2@L12", "zscore", "plain_pi0_base_grid2@L12_z"),
    "π0 LIBERO L12": ("vla:pi0_libero", "grid2@L12", "zscore", "plain_pi0_libero_grid2@L12_z"),
}
# name prefixes of the keys whose weights never saw LIBERO actions (the learnt key only for held-out queries, see
# --heldout)
NEVER_TRAINED_ON_LIBERO = ("ResNet-18", "DINOv2", "SmolVLA base", "π0.5 base", "π0 base")

# the backbones added after the layer pilots
_EXTRA_KEYS = extra_keys()
SPACES.update({k["label"]: (k["encoder"], k["pooling"], k["norm"], k["stem"]) for k in _EXTRA_KEYS})
NEVER_TRAINED_ON_LIBERO += tuple(k["label"] for k in _EXTRA_KEYS if k["fair"])


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of (Q, M) tensors."""
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-8)


def fit_norms(enc: str, pooling: str, norm: str, rows: np.ndarray) -> list[tuple]:
    """Per camera, the key normalisation of the plain lookup (fitted on its memory ``rows``, as ``build_memory``)."""
    out = []
    for cam in DATASET_CAMERAS:
        raw = load_features(DEFAULT_CACHE_DIR, REPO_ID, enc, cam, pooling)

        def rows_of(part: np.ndarray, raw=raw) -> torch.Tensor:
            return torch.from_numpy(np.asarray(raw[part]))

        out.append(fit_key_norm(norm, cam, rows, rows_of, "cpu", chunk=8192))
    return out


def key_features(enc: str, pooling: str, rows: np.ndarray, norms: list[tuple]) -> list[torch.Tensor]:
    """Per camera, the plain lookup's unit-norm keys of dataset ``rows`` (``norms`` from :func:`fit_norms`)."""
    out = []
    for cam, norm in zip(DATASET_CAMERAS, norms, strict=True):
        raw = load_features(DEFAULT_CACHE_DIR, REPO_ID, enc, cam, pooling)
        out.append(normalize_keys(torch.from_numpy(np.asarray(raw[rows])), pooling, *norm))
    return out


@torch.no_grad()
def evaluate(
    space: str,
    frames: FrameIndex,
    horizon: int = 10,
    ks: tuple[int, ...] = (1, 8),
    local_m: int = 64,
    queries_per_task: int = 120,
    query_episodes: np.ndarray | None = None,
) -> dict:
    """``lac@k``, ``local_rho`` and ``global_rho`` (see the module docstring) of one key space."""
    enc, pooling, norm, _ = SPACES[space]
    memory = np.nonzero(frames.frame_index % 2 == 0)[0]
    norms = fit_norms(enc, pooling, norm, memory)
    rng = np.random.default_rng(0)
    lac = {k: [] for k in ks}
    local_rho, global_rho = [], []
    for task in np.unique(frames.task_index):
        cand = memory[frames.task_index[memory] == task]
        queries = cand[frames.frame_index[cand] % 10 == 4]
        if query_episodes is not None:
            queries = queries[np.isin(frames.episode_index[queries], query_episodes)]
        queries = np.sort(rng.choice(queries, min(queries_per_task, len(queries)), replace=False))
        position = {int(r): i for i, r in enumerate(cand)}
        qi = torch.tensor([position[int(r)] for r in queries])  # the queries' positions among the candidates
        keys = key_features(enc, pooling, cand, norms)
        d_e = sum(2 - 2 * k[qi] @ k.T for k in keys) / len(keys)  # (Q, C) mean camera distance
        motion = torch.from_numpy(action_chunks(frames, cand, horizon)[..., MOTION]).reshape(len(cand), -1)
        d_a = (
            torch.cdist(motion[qi], motion).pow(2) / motion.shape[1]
        )  # (Q, C) mean squared motion difference
        episode = torch.from_numpy(frames.episode_index[cand])
        other = episode[None, :] != episode[qi][:, None]  # candidates of other demonstrations
        d_e = d_e.masked_fill(~other, float("inf"))
        d_a_random = (d_a * other).sum(1) / other.sum(1)
        order = d_e.argsort(1)
        for k in ks:
            d_a_neighbours = d_a.gather(1, order[:, :k]).mean(1)
            lac[k].append((1 - d_a_neighbours / d_a_random.clamp_min(1e-9)).numpy())
        nearest = order[:, :local_m]
        local_rho.append(spearman(d_e.gather(1, nearest), d_a.gather(1, nearest)).numpy())
        sample = torch.stack(
            [torch.from_numpy(rng.choice(np.nonzero(o.numpy())[0], 256, replace=False)) for o in other]
        )
        global_rho.append(spearman(d_e.gather(1, sample), d_a.gather(1, sample)).numpy())
    res = {f"lac@{k}": float(np.concatenate(v).mean()) for k, v in lac.items()}
    res["local_rho"] = float(np.concatenate(local_rho).mean())
    res["global_rho"] = float(np.concatenate(global_rho).mean())
    return res


def correlate(results: dict, heldout: bool, protocol: str = "full50") -> dict:
    """``protocol``: "full50" relates to the lookups on the official protocol (initial states 0-49, all 40 tasks) only;
    "any": the latest run of each key, whatever its episodes."""
    success = {}  # config stem -> closed-loop success (%)
    for run in read_jsonl(KNN_OUT / "compare.jsonl"):
        if protocol == "full50" and not (run.get("tag") == "full50" and run.get("init_states") == "0:50"):
            continue
        success[Path(run["config_file"]).stem] = 100 * run["success"]
    rows = [(name, m, success[SPACES[name][3]]) for name, m in results.items() if SPACES[name][3] in success]

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
        "--protocol", default="full50", choices=("full50", "any"), help="which closed-loop runs to relate to"
    )
    parser.add_argument(
        "--heldout", action="store_true", help="queries from the held-out demos of the split only"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    frames = load_frame_index(REPO_ID)
    query_episodes = np.array(json.loads(SPLIT.read_text())["heldout"]) if args.heldout else None
    out_path = Path(args.out or VLA_OUT / f"neighbor_metric{'_heldout' if args.heldout else ''}.json")
    results = json.loads(out_path.read_text())["spaces"] if out_path.exists() else {}
    results = {k: v for k, v in results.items() if k in SPACES}  # spaces dropped from the study
    for space in args.spaces.split(","):
        if space in results:
            continue
        try:
            results[space] = evaluate(space, frames, query_episodes=query_episodes)
        except FileNotFoundError as e:
            print(f"skip {space}: {e}")
            continue
        print(space, {k: round(v, 3) for k, v in results[space].items()}, flush=True)
        out_path.write_text(json.dumps({"spaces": results}, indent=1))
    summary = {"spaces": results}
    if args.correlate:
        summary["correlation"] = correlate(results, args.heldout, args.protocol)
        print(json.dumps({k: v for k, v in summary["correlation"].items() if k != "points"}, indent=1))
    out_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
