"""2-D maps of every lookup key: PCA and t-SNE of the same demonstration frames, before and after fine-tuning (CPU).

For a fixed sample of frames (``--demos`` demonstrations per task, every ``--every``-th frame), every key space of
``neighbor_metric`` (the generic encoders, the learnt key, and each VLM before / after its LIBERO fine-tuning at the layer
the lookup uses) gives the plain lookup's key: per camera standardised as in the memory and unit-normalised, cameras
concatenated. Both maps are computed from these keys: PCA (first two components) and t-SNE (on the top 50 principal
components). Two numbers, measured in the full key space, say what the maps show:

* ``task_nn``: how often a frame's nearest neighbour from another demonstration belongs to the same task
* ``progress_gap``: median |progress difference| (progress = frame / episode length) between a frame and its nearest
  neighbour from another demonstration of the same task; 0 = the key knows how far along the task the robot is

and both per suite (in LIBERO-Goal all ten tasks share one scene, so only the instruction tells them apart).

    uv run python -m lerobot_policy_knn.experiments.embedding_maps
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..features import load_frame_index
from .neighbor_metric import SPACES, extra_keys, fit_norms, key_features

OUT = Path("outputs/vla/embedding_maps.json")
# key space (neighbor_metric.SPACES) -> (family, stage); the other backbones come from extra_keys.json
MAP_SPACES = {
    "ResNet-18 grid": ("ResNet-18", "generic"),
    "DINOv2-S": ("DINOv2", "generic"),
    "ResNet-18 grid (learnt key)": ("ResNet-18, learnt key", "learnt"),
    "SmolVLA base grid": ("SmolVLA", "before"),
    "SmolVLA LIBERO grid": ("SmolVLA", "after"),
    "π0.5 base L12": ("π0.5", "before"),
    "π0.5 LIBERO L12": ("π0.5", "after"),
    "π0 base L12": ("π0", "before"),
    "π0 LIBERO L12": ("π0", "after"),
}
FAMILIES = {"xvla": "X-VLA", "pi0fast": "π0-FAST", "vla_jepa": "VLA-JEPA", "molmoact2": "MolmoAct2", "evo1": "Evo-1",
            "groot": "GR00T N1.7"}  # fmt: skip
for _k in extra_keys():
    _family = next(v for k, v in FAMILIES.items() if _k["name"].startswith(k))
    MAP_SPACES[_k["label"]] = (_family, "after" if " LIBERO " in _k["label"] else "before")


def suite_of_tasks(task_names: list[str]) -> list[str]:
    """The LIBERO suite of every dataset task (matched on the instruction)."""
    from libero.libero import benchmark

    suites = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        for task in benchmark.get_benchmark_dict()[suite]().tasks:
            suites[task.language] = suite
    return [suites[t] for t in task_names]


def sample_rows(f, demos: int, every: int, seed: int = 0) -> np.ndarray:
    """``demos`` random demonstrations per task, every ``every``-th frame of each (even frames: in every cache)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in np.unique(f.task_index):
        for e in np.sort(rng.choice(np.unique(f.episode_index[f.task_index == t]), demos, replace=False)):
            r = np.nonzero(f.episode_index == e)[0]
            rows.append(r[f.frame_index[r] % every == 0])
    return np.concatenate(rows)


def neighbour_stats(x: torch.Tensor, task: np.ndarray, episode: np.ndarray, progress: np.ndarray,
                    suite: np.ndarray) -> dict:  # fmt: skip
    """task_nn and progress_gap (see the module docstring) of unit-norm keys ``x``, overall and per suite."""
    sim = x @ x.T
    task_t, ep_t = torch.from_numpy(task), torch.from_numpy(episode)
    sim.masked_fill_(ep_t[:, None] == ep_t[None], -torch.inf)  # neighbours from other demonstrations only
    same_task = (task_t[sim.argmax(1)] == task_t).numpy()
    sim.masked_fill_(task_t[:, None] != task_t[None], -torch.inf)
    gap = np.abs(progress - progress[sim.argmax(1).numpy()])
    out = {"task_nn": float(same_task.mean()), "progress_gap": float(np.median(gap))}
    for s in sorted(set(suite)):
        out[f"task_nn/{s}"] = float(same_task[suite == s].mean())
        out[f"progress_gap/{s}"] = float(np.median(gap[suite == s]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=int, default=6, help="demonstrations per task")
    parser.add_argument("--every", type=int, default=10, help="keep every n-th frame of a demonstration")
    parser.add_argument(
        "--norm_rows", type=int, default=20000, help="memory frames the z-score statistics use"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    f = load_frame_index("lerobot/libero")
    rows = sample_rows(f, args.demos, args.every)
    task, episode = f.task_index[rows], f.episode_index[rows]
    progress = f.frame_index[rows] / np.maximum(f.episode_length[rows] - 1, 1)
    memory = np.nonzero(f.frame_index % 2 == 0)[0]
    norm_rows = np.sort(
        np.random.default_rng(0).choice(memory, min(args.norm_rows, len(memory)), replace=False)
    )
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    if out.get("rows") != rows.tolist():
        out = {"spaces": {}}
    suite_of_task = suite_of_tasks(f.task_names)
    suite = np.array([suite_of_task[t] for t in task])
    out.update(rows=rows.tolist(), task=task.tolist(), suite=suite.tolist(), episode=episode.tolist(),
               progress=np.round(progress, 3).tolist(), task_names=f.task_names)  # fmt: skip
    for space, (family, stage) in MAP_SPACES.items():
        if space in out["spaces"]:
            continue
        enc, pooling, norm, _stem = SPACES[space]
        keys = key_features(enc, pooling, rows, fit_norms(enc, pooling, norm, norm_rows))
        x = (
            torch.cat(keys, 1) / len(keys) ** 0.5
        )  # unit norm: squared distance = the lookup's mean camera distance
        pca = PCA(n_components=50, random_state=0)
        z = pca.fit_transform(x.numpy())
        tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)
        out["spaces"][space] = {
            "family": family,
            "stage": stage,
            "pca": np.round(z[:, :2] / np.abs(z[:, :2]).max(), 3).tolist(),
            "pca_variance": pca.explained_variance_ratio_[:2].round(4).tolist(),
            "tsne": np.round(tsne / np.abs(tsne).max(), 3).tolist(),
            **neighbour_stats(x, task, episode, progress, suite),
        }
        print(
            space,
            {k: v for k, v in out["spaces"][space].items() if k in ("task_nn", "progress_gap")},
            flush=True,
        )
        OUT.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
