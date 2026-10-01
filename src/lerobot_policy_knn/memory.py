"""Build the retrieval memory of a :class:`KNNPolicy` from a dataset's cached features."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .configuration_knn import KNNConfig
from .encoders import TextEncoder, finalize_features
from .features import DEFAULT_CACHE_DIR, FrameIndex, load_features, load_frame_index
from .modeling_knn import KNNPolicy, camera_key_name, featurize_state

logger = logging.getLogger(__name__)

# (encoder, camera, pooling, resolution) -> raw features, shared across policies built in one process
FeatureLoader = Callable[[str, str, str, int], np.ndarray | Tensor]


def select_key_rows(frames: FrameIndex, stride: int, episodes: np.ndarray | None = None) -> np.ndarray:
    """Global rows used as retrieval keys, sorted by (task, episode, frame)."""
    keep = frames.frame_index % stride == 0
    if episodes is not None:
        keep &= np.isin(frames.episode_index, episodes)
    rows = np.nonzero(keep)[0]
    order = np.lexsort((frames.frame_index[rows], frames.episode_index[rows], frames.task_index[rows]))
    return rows[order]


def _median_pair_distances(
    keys: dict[str, Tensor], frame: Tensor, task: Tensor, n_pairs: int = 20000, seed: int = 0
) -> dict[str, float]:
    """Median squared distance per modality between random frames of the same task."""
    g = torch.Generator().manual_seed(seed)
    n = len(task)
    i = torch.randint(0, n, (n_pairs * 4,), generator=g)
    j = torch.randint(0, n, (n_pairs * 4,), generator=g)
    same = (task.cpu()[i] == task.cpu()[j]) & (i != j)
    i, j = i[same][:n_pairs], j[same][:n_pairs]
    scales: dict[str, float] = {}
    for name, value in keys.items():
        a, b = value[i.to(value.device)].float(), value[j.to(value.device)].float()
        d = (a - b).pow(2).sum(-1)
        if name == "key_state":
            d = d / value.shape[-1]
        scales[name] = float(d.median().clamp_min(1e-8))
    ft = frame.float()
    scales["time"] = float((ft[i.to(ft.device)] - ft[j.to(ft.device)]).pow(2).median().clamp_min(1.0))
    return scales


@torch.no_grad()
def _median_nn_distances(
    keys: dict[str, Tensor], task: Tensor, episode: Tensor, text: Tensor, n_queries: int = 2048, seed: int = 0
) -> dict[str, float]:
    """Median nearest-neighbour squared distance per modality: for random keys, the closest key of the
    same task in another demonstration (the scale at which retrieval actually discriminates)."""
    g = torch.Generator().manual_seed(seed)
    n = len(task)
    sample = torch.randperm(n, generator=g)[: min(n_queries, n)]
    device = next(iter(keys.values())).device
    task, episode, sample = task.to(device), episode.to(device), sample.to(device)
    scales: dict[str, float] = {}
    for name, value in keys.items():
        mins = []
        v = value.float() if name == "key_state" else value
        for s in range(0, len(sample), 256):
            q_idx = sample[s : s + 256]
            if name == "key_state":
                d = torch.cdist(v[q_idx], v).pow(2) / v.shape[-1]
            else:
                d = (2.0 - 2.0 * torch.mm(v[q_idx].half(), v.T, out_dtype=torch.float32)).clamp_(min=0.0)
            invalid = (task[None] != task[q_idx][:, None]) | (episode[None] == episode[q_idx][:, None])
            mins.append(d.masked_fill_(invalid, float("inf")).min(-1).values)
        m = torch.cat(mins)
        scales[name] = float(m[torch.isfinite(m)].median().clamp_min(1e-8))
    # text: distance from each task prompt to the closest other task prompt
    td = (2.0 - 2.0 * text @ text.T).clamp_(min=0.0).fill_diagonal_(float("inf"))
    scales["text"] = float(td.min(-1).values.median().clamp_min(1e-8))
    return scales


def fit_key_norm(
    key_norm: str,
    cam: str,
    rows: np.ndarray,
    rows_of: Callable[[np.ndarray], Tensor],
    device: str | torch.device,
    chunk: int = 32768,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """(mean, std, projection) of a camera's ``key_norm`` fitted on the memory ``rows`` (``None`` where unused).

    ``rows_of(part)`` returns the raw features of dataset rows ``part``; statistics stream ``chunk`` rows at a time.
    """
    if key_norm == "none":
        return None, None, None
    if key_norm.startswith("file:"):  # a learnt key (experiments/learn_key.py): mean, std, projection
        learnt = torch.load(key_norm.removeprefix("file:"), map_location="cpu")["cameras"][cam]
        mean, std, proj = (learnt[k].float().to(device) for k in ("mean", "std", "proj"))
        return mean, std, proj
    if key_norm != "zscore" and not key_norm.startswith("zscore_pca"):
        raise ValueError(f"Unknown key_norm {key_norm!r}")
    total = sq = None
    for s in range(0, len(rows), chunk):
        x = rows_of(rows[s : s + chunk]).to(device).double()
        total = x.sum(0) if total is None else total + x.sum(0)
        sq = (x * x).sum(0) if sq is None else sq + (x * x).sum(0)
    mean = total / len(rows)
    std = (sq / len(rows) - mean * mean).clamp_min(0).sqrt().clamp_min(1e-3)
    mean, std = mean.float(), std.float()
    if key_norm == "zscore":
        return mean, std, None
    # whitened PCA of the standardised keys: shrinks multi-layer keys (e.g. 3 x 8192 dims) to n dims and
    # lets directions shared across layers count once
    n = int(key_norm.removeprefix("zscore_pca"))
    sample = np.sort(np.random.default_rng(0).choice(rows, min(len(rows), 20000), replace=False))
    x = (rows_of(sample).to(device).float() - mean) / std
    with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):  # pca_lowrank is randomised
        torch.manual_seed(0)
        _, sv, v = torch.pca_lowrank(x, q=n, center=False)
    return mean, std, v[:, :n] / (sv[:n] / (len(sample) - 1) ** 0.5).clamp_min(1e-6)


def normalize_keys(
    x: Tensor, pooling: str, mean: Tensor | None, std: Tensor | None, proj: Tensor | None
) -> Tensor:
    """Unit-norm retrieval keys from raw features and a fitted :func:`fit_key_norm`."""
    if mean is not None:
        x = (x - mean) / std
    if proj is not None:
        return torch.nn.functional.normalize(x @ proj, dim=-1)
    return finalize_features(x, pooling)


def build_memory(
    config: KNNConfig,
    frames: FrameIndex,
    load_raw: FeatureLoader,
    episodes: np.ndarray | None = None,
    device: str | torch.device = "cpu",
    text_encoder: TextEncoder | None = None,
    task_text_emb: Tensor | None = None,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Return the memory buffers of a policy and the per-modality distance scales.

    ``config.memory_shapes``, ``modality_scales`` and ``task_names`` are updated in place.
    """
    rows = select_key_rows(frames, config.memory_stride, episodes)
    rows_t = torch.from_numpy(rows)
    buffers: dict[str, Tensor] = {}
    for cam in config.active_cameras:
        raw = load_raw(config.image_encoder, cam, config.image_pooling, config.image_resolution)
        keys = None

        def rows_of(part: np.ndarray, raw: np.ndarray | Tensor = raw) -> Tensor:
            if isinstance(raw, Tensor):
                return raw[torch.from_numpy(part).to(raw.device)]
            return torch.as_tensor(np.asarray(raw[part]))

        mean, std, proj = fit_key_norm(config.key_norm, cam, rows, rows_of, device)
        name = camera_key_name(cam).removeprefix("key_cam_")
        for kind, value in (("mean", mean), ("std", std), ("proj", proj)):
            if value is not None:
                buffers[f"norm_{kind}_{name}"] = value
        # finalise in chunks: a full-precision copy of every key at once can exceed free GPU memory
        for s in range(0, len(rows), 32768):
            part = rows[s : s + 32768]
            chunk = normalize_keys(
                rows_of(part).to(device).float(), config.image_pooling, mean, std, proj
            ).half()
            if keys is None:
                keys = torch.empty((len(rows), chunk.shape[1]), dtype=torch.float16, device=device)
            keys[s : s + len(part)] = chunk
        buffers[camera_key_name(cam)] = keys

    state_all = featurize_state(torch.from_numpy(frames.state), config.state_axis_angle_start)
    mean, std = state_all.mean(0), state_all.std(0).clamp_min(1e-3)
    buffers["state_mean"], buffers["state_std"] = mean, std
    buffers["key_state"] = (state_all[rows_t] - mean) / std

    task = torch.from_numpy(frames.task_index[rows])
    buffers["key_frame"] = torch.from_numpy(frames.frame_index[rows])
    buffers["key_episode"] = torch.from_numpy(frames.episode_index[rows])
    buffers["key_ptr"] = rows_t.clone()
    n_tasks = len(frames.task_names)
    counts = torch.bincount(task, minlength=n_tasks)
    buffers["task_offsets"] = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)])
    buffers["actions"] = torch.from_numpy(frames.action)
    buffers["action_episode_end"] = torch.from_numpy(frames.episode_end)

    if task_text_emb is None:
        text_encoder = text_encoder or TextEncoder(config.text_encoder)
        task_text_emb = text_encoder(frames.task_names).cpu()
    buffers["task_text_emb"] = task_text_emb.float().cpu()

    buffers = {k: v.to(device) for k, v in buffers.items()}
    scale_keys = {k: v for k, v in buffers.items() if k.startswith("key_cam_") or k == "key_state"}
    scales = _median_pair_distances(scale_keys, buffers["key_frame"], task)
    text = buffers["task_text_emb"]
    pair_text = (2.0 - 2.0 * text @ text.T).clamp(min=0.0)
    off_diag = ~torch.eye(len(text), dtype=torch.bool, device=text.device)
    scales["text"] = float(pair_text[off_diag].median().clamp_min(1e-8))
    nn_scales = _median_nn_distances(scale_keys, task, buffers["key_episode"], text)

    config.memory_shapes = {k: list(v.shape) for k, v in buffers.items()}
    config.modality_scales = scales
    config.modality_nn_scales = nn_scales
    config.task_names = list(frames.task_names)
    return buffers, scales


def cache_loader(cache_dir: Path | str, repo_id: str) -> FeatureLoader:
    def load(encoder: str, camera: str, pooling: str, resolution: int) -> np.ndarray:
        return load_features(cache_dir, repo_id, encoder, camera, pooling, resolution)

    return load


def make_policy_from_cache(
    config: KNNConfig,
    repo_id: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    episodes: np.ndarray | None = None,
    device: str | None = None,
) -> KNNPolicy:
    frames = load_frame_index(repo_id, cache_dir=cache_dir)
    device = device or config.device or "cpu"
    buffers, _ = build_memory(config, frames, cache_loader(cache_dir, repo_id), episodes, device=device)
    config.dataset_repo_id = repo_id
    policy = KNNPolicy(config)
    policy.load_memory(buffers)
    return policy.to(device).eval()
