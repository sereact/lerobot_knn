"""Dataset-side utilities: per-frame index, cached image features, and memory construction.

Features are extracted once per (dataset, encoder) by decoding every episode video *sequentially*
(much faster than the random access a ``LeRobotDataset`` does) and are cached as ``.npy`` files
of raw pooled features, one row per dataset frame (global ``index`` order). Building a policy
memory from the cache is then a cheap, in-memory operation, which is what makes large
hyper-parameter searches feasible.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from lerobot.datasets import LeRobotDatasetMetadata

from .encoders import IMAGE_ENCODERS, ImageEncoder

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("outputs/knn/feature_cache")


def _cache_root(cache_dir: Path | str, repo_id: str) -> Path:
    return Path(cache_dir) / repo_id.replace("/", "__")


@dataclass
class FrameIndex:
    """Per-frame arrays of a dataset, in global ``index`` order."""

    state: np.ndarray  # (N, S) float32
    action: np.ndarray  # (N, A) float32
    episode_index: np.ndarray  # (N,) int64
    frame_index: np.ndarray  # (N,) int64
    task_index: np.ndarray  # (N,) int64
    episode_end: np.ndarray  # (N,) int64, global index of the last frame of the frame's episode
    episode_length: np.ndarray  # (N,) int64
    task_names: list[str]  # indexed by task_index

    def __len__(self) -> int:
        return len(self.state)


def load_frame_index(
    repo_id: str, root: str | Path | None = None, cache_dir: Path | str = DEFAULT_CACHE_DIR
) -> FrameIndex:
    cache = _cache_root(cache_dir, repo_id) / "frame_index.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        names = json.loads(str(data["task_names"]))
        return FrameIndex(
            **{k: data[k] for k in data.files if k != "task_names"},
            task_names=names,
        )

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    files = sorted((meta.root / "data").glob("*/*.parquet"))
    cols = ["observation.state", "action", "episode_index", "frame_index", "index", "task_index"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values("index")
    n = len(df)
    if not np.array_equal(df["index"].to_numpy(), np.arange(n)):
        raise ValueError("Dataset global index is not contiguous")
    episode_index = df["episode_index"].to_numpy().astype(np.int64)
    ep_len = df.groupby("episode_index")["frame_index"].transform("count").to_numpy().astype(np.int64)
    ep_start = df.groupby("episode_index")["index"].transform("min").to_numpy().astype(np.int64)
    tasks = meta.tasks
    task_names = [""] * len(tasks)
    for name, row in tasks.iterrows():
        task_names[int(row["task_index"])] = str(name)
    fi = FrameIndex(
        state=np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        action=np.stack(df["action"].to_numpy()).astype(np.float32),
        episode_index=episode_index,
        frame_index=df["frame_index"].to_numpy().astype(np.int64),
        task_index=df["task_index"].to_numpy().astype(np.int64),
        episode_end=ep_start + ep_len - 1,
        episode_length=ep_len,
        task_names=task_names,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        state=fi.state,
        action=fi.action,
        episode_index=fi.episode_index,
        frame_index=fi.frame_index,
        task_index=fi.task_index,
        episode_end=fi.episode_end,
        episode_length=fi.episode_length,
        task_names=np.array(json.dumps(task_names)),
    )
    return fi


def feature_path(
    cache_dir: Path | str, repo_id: str, encoder: str, camera: str, pooling: str, resolution: int = 224
) -> Path:
    if encoder.startswith("vla:"):
        from .vla import cache_tag

        return _cache_root(cache_dir, repo_id) / cache_tag(encoder[4:]) / f"{camera}.{pooling}.npy"
    tag = encoder if resolution == 224 else f"{encoder}@{resolution}"
    return _cache_root(cache_dir, repo_id) / tag / f"{camera}.{pooling}.npy"


def _episode_video_layout(meta: LeRobotDatasetMetadata, camera: str) -> dict[int, list[tuple[int, int, int]]]:
    """file_index -> list of (first_frame_in_file, global_from_index, length), one per episode."""
    eps = meta.episodes.to_pandas()
    layout: dict[int, list[tuple[int, int, int]]] = {}
    fcol = f"videos/{camera}/file_index"
    ccol = f"videos/{camera}/chunk_index"
    tcol = f"videos/{camera}/from_timestamp"
    for _, r in eps.iterrows():
        if int(r[ccol]) != 0:
            raise NotImplementedError("Only single-chunk datasets are supported by the fast extractor")
        start = int(round(float(r[tcol]) * meta.fps))
        layout.setdefault(int(r[fcol]), []).append((start, int(r["dataset_from_index"]), int(r["length"])))
    return layout


@torch.no_grad()
def extract_features(
    repo_id: str,
    encoders: list[str],
    cameras: list[str] | None = None,
    root: str | Path | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    resolution: int = 224,
    batch_size: int = 256,
    device: str = "cuda",
    poolings: dict[str, tuple[str, ...]] | None = None,
    overwrite: bool = False,
) -> None:
    """Decode every video frame of ``repo_id`` once and cache raw pooled features for ``encoders``."""
    from torchcodec.decoders import VideoDecoder

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    cameras = cameras or list(meta.video_keys)
    n_total = meta.total_frames
    load_frame_index(repo_id, root=root, cache_dir=cache_dir)

    for camera in cameras:
        todo: dict[str, tuple[ImageEncoder, dict[str, np.memmap]]] = {}
        for enc_name in encoders:
            wanted = (poolings or {}).get(enc_name, IMAGE_ENCODERS[enc_name].poolings)
            paths = {p: feature_path(cache_dir, repo_id, enc_name, camera, p, resolution) for p in wanted}
            if not overwrite and all(path.exists() for path in paths.values()):
                logger.info("Features for %s/%s already cached", enc_name, camera)
                continue
            encoder = ImageEncoder(enc_name, resolution=resolution).to(device)
            probe = encoder(torch.zeros(2, 3, 256, 256, device=device))
            outs = {}
            for p, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                outs[p] = open_memmap(
                    str(path) + ".tmp.npy", mode="w+", dtype=np.float16, shape=(n_total, probe[p].shape[1])
                )
            todo[enc_name] = (encoder, outs)
        if not todo:
            continue

        layout = _episode_video_layout(meta, camera)
        written = np.zeros(n_total, dtype=bool)
        t0 = time.time()
        for file_index in sorted(layout):
            video = meta.root / f"videos/{camera}/chunk-000/file-{file_index:03d}.mp4"
            decoder = VideoDecoder(str(video), device="cpu")
            n_frames = decoder.metadata.num_frames
            # map each frame of the file to its global dataset index (-1 if unused)
            g = np.full(n_frames, -1, dtype=np.int64)
            for start, gfrom, length in layout[file_index]:
                g[start : start + length] = np.arange(gfrom, gfrom + length)
            for s in range(0, n_frames, batch_size):
                e = min(n_frames, s + batch_size)
                gi = g[s:e]
                keep = gi >= 0
                if not keep.any():
                    continue
                frames = decoder.get_frames_in_range(s, e).data[torch.from_numpy(keep)]
                images = frames.to(device, non_blocking=True).float().div_(255.0)
                for encoder, outs in todo.values():
                    feats = encoder(images)
                    for p, arr in outs.items():
                        arr[gi[keep]] = feats[p].half().cpu().numpy()
                written[gi[keep]] = True
            logger.info("[%s] file %d/%d done (%.0fs)", camera, file_index + 1, len(layout), time.time() - t0)
        if not written.all():
            raise RuntimeError(f"{(~written).sum()} frames were not decoded for {camera}")
        for _encoder, outs in todo.values():
            for arr in outs.values():
                arr.flush()
                tmp = Path(arr.filename)
                del arr
                tmp.rename(str(tmp).removesuffix(".tmp.npy"))
        del todo
        torch.cuda.empty_cache()


def layer_parts(pooling: str) -> list[str]:
    """``grid2@L12+L15+last`` -> [``grid2@L12``, ``grid2@L15``, ``grid2``] (a multi-layer VLA key)."""
    head, _, taps = pooling.partition("@")
    return [head if t == "last" else f"{head}@{t}" for t in taps.split("+")]


class ConcatFeatures:
    """Row-indexable concatenation of several cached feature arrays (one per layer)."""

    def __init__(self, parts: list):
        self.parts = parts
        self.shape = (len(parts[0]), sum(p.shape[1] for p in parts))
        self.dtype = parts[0].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        return np.concatenate([np.asarray(p[idx]) for p in self.parts], axis=1)


def load_features(
    cache_dir: Path | str, repo_id: str, encoder: str, camera: str, pooling: str, resolution: int = 224
) -> np.ndarray:
    if "@" in pooling and "+" in pooling:
        return ConcatFeatures(
            [load_features(cache_dir, repo_id, encoder, camera, p, resolution) for p in layer_parts(pooling)]
        )
    path = feature_path(cache_dir, repo_id, encoder, camera, pooling, resolution)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features at {path}. Run `lerobot-knn-fit` (or features.extract_features) first."
        )
    arr = np.load(path, mmap_mode="r")
    rows_file = path.parent / "rows.npy"
    if rows_file.exists():  # compact cache holding a subset of rows (e.g. VLA embeddings of even frames)
        return RowMapped(arr, np.load(rows_file))
    return arr


class RowMapped:
    """Indexes a compact feature array by global dataset row (rows that were not stored raise)."""

    def __init__(self, arr: np.ndarray, rows: np.ndarray):
        self.arr = arr
        self.rows = rows
        self.lookup = np.full(int(rows.max()) + 1, -1, dtype=np.int64)
        self.lookup[rows] = np.arange(len(rows))
        self.shape = (len(self.lookup), arr.shape[1])
        self.dtype = arr.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx):
        local = self.lookup[np.asarray(idx)]
        if (local < 0).any():
            raise IndexError("Requested rows that are not in this compact feature cache")
        return self.arr[local]
