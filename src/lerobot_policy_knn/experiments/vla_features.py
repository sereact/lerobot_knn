"""Embed every (even) frame of ``lerobot/libero`` with a VLA backbone and cache the pooled hidden states.

    uv run python -m lerobot_policy_knn.experiments.vla_features smolvla_libero smolvla_base

Unlike image encoders, a VLA embeds both cameras, the prompt and the state together, so episodes are decoded
camera-by-camera in lockstep. Only rows with ``frame_index % stride == 0`` are stored (compact arrays plus a
``rows.npy`` map), which is exactly the memory of a policy with ``memory_stride == stride``.
"""

import argparse
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_STATE

from ..features import DEFAULT_CACHE_DIR, FrameIndex, cache_root, episode_video_layout, load_frame_index
from ..vla import DATASET_CAMERAS, VLA_CAMERA, VLAEmbedder, cache_tag
from .common import REPO_ID

logger = logging.getLogger(__name__)


def _decode_worker(
    meta: LeRobotDatasetMetadata,
    frames: FrameIndex,
    rows: np.ndarray,
    out: queue.Queue,
    stop: threading.Event,
) -> None:
    """Put (global rows, {camera: uint8 (n, 3, H, W)}) per episode on ``out``, in dataset order, then ``None``."""
    from torchcodec.decoders import VideoDecoder

    # camera -> first global index of an episode -> (video file, the episode's first frame in it)
    location: dict[str, dict[int, tuple[int, int]]] = {}
    for cam in DATASET_CAMERAS:
        location[cam] = {
            first_row: (file_index, start)
            for file_index, episodes in episode_video_layout(meta, cam).items()
            for start, first_row, _length in episodes
        }
    decoders: dict[str, tuple[int, VideoDecoder]] = {}  # camera -> (open file, its decoder)
    episode_starts = np.nonzero(frames.frame_index == 0)[0]
    episode_of_row = np.searchsorted(episode_starts, rows, side="right") - 1
    for episode in np.unique(episode_of_row):
        if stop.is_set():
            return
        episode_rows = rows[episode_of_row == episode]
        first_row = int(episode_starts[episode])
        images = {}
        for cam in DATASET_CAMERAS:
            file_index, start = location[cam][first_row]
            if cam not in decoders or decoders[cam][0] != file_index:
                video = meta.root / f"videos/{cam}/chunk-000/file-{file_index:03d}.mp4"
                decoders[cam] = (file_index, VideoDecoder(str(video), device="cpu"))
            in_file = (episode_rows - first_row + start).tolist()
            images[cam] = decoders[cam][1].get_frames_at(indices=in_file).data
        out.put((episode_rows, images))
    out.put(None)


@torch.no_grad()
def extract(
    name: str,
    stride: int = 2,
    batch_size: int = 64,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    poolings: tuple[str, ...] | None = None,
    device: str = "cuda",
    overwrite: bool = False,
    taps: tuple[str, ...] = (),
    tasks: list[int] | None = None,
    tag_suffix: str = "",
) -> Path:
    """Cache the pooled hidden states of VLA ``name`` for every ``stride``-th frame; returns the cache directory."""
    frames = load_frame_index(REPO_ID, cache_dir=cache_dir)
    meta = LeRobotDatasetMetadata(REPO_ID)
    root = cache_root(cache_dir, REPO_ID) / (cache_tag(name) + tag_suffix)
    if (root / "rows.npy").exists() and not overwrite:
        cached = {f.name.split(".")[-2] for f in root.glob("*.npy") if f.name != "rows.npy"}
        if poolings is None or set(poolings) <= cached:
            logger.info("%s already cached at %s", name, root)
            return root
    keep = frames.frame_index % stride == 0
    if tasks is not None:  # a subset of tasks (pilot studies)
        keep &= np.isin(frames.task_index, tasks)
    rows = np.nonzero(keep)[0]
    embedder = VLAEmbedder(name, device=device, taps=taps)
    poolings = poolings or embedder.poolings
    root.mkdir(parents=True, exist_ok=True)
    # one compact array per (pooling, camera); prompt-level poolings live under the pseudo camera
    outs: dict[tuple[str, str], np.memmap] = {}
    for pooling in poolings:
        per_camera = pooling in ("img", "grid2") or pooling.startswith("grid2@")
        dim = embedder.hidden_size * (4 if pooling.startswith("grid2") else 1)
        for cam in embedder.present if per_camera else [VLA_CAMERA]:
            path = root / f"{cam}.{pooling}.tmp.npy"
            outs[(pooling, cam)] = open_memmap(str(path), mode="w+", dtype=np.float16, shape=(len(rows), dim))
    position = {int(r): i for i, r in enumerate(rows)}

    def embed(model: VLAEmbedder, batch_rows: np.ndarray, images: dict[str, torch.Tensor]) -> None:
        batch = {c: images[c].to(device, non_blocking=True).float().div_(255.0) for c in DATASET_CAMERAS}
        batch[OBS_STATE] = torch.from_numpy(frames.state[batch_rows]).float().to(device)
        batch["task"] = [frames.task_names[int(t)] for t in frames.task_index[batch_rows]]
        feats = model(batch)
        index = np.array([position[int(r)] for r in batch_rows])
        for (pooling, cam), array in outs.items():
            array[index] = feats[pooling][cam].half().cpu().numpy()

    # a thread decodes whole episodes; they are regrouped here into batches of batch_size frames
    decoded: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    threading.Thread(target=_decode_worker, args=(meta, frames, rows, decoded, stop), daemon=True).start()
    pending_rows: list[np.ndarray] = []
    pending_images: dict[str, list[torch.Tensor]] = {c: [] for c in DATASET_CAMERAS}
    done, t0, last_log = 0, time.time(), 0.0
    try:
        finished = False
        while not finished:
            item = decoded.get()
            if item is None:
                finished = True
            else:
                pending_rows.append(item[0])
                for c in DATASET_CAMERAS:
                    pending_images[c].append(item[1][c])
            n_pending = sum(len(x) for x in pending_rows)
            if n_pending < batch_size and not (finished and n_pending):
                continue
            all_rows = np.concatenate(pending_rows)
            all_images = {c: torch.cat(pending_images[c]) for c in DATASET_CAMERAS}
            n_full = len(all_rows) if finished else (len(all_rows) // batch_size) * batch_size
            for s in range(0, n_full, batch_size):
                e = min(s + batch_size, n_full)
                embed(embedder, all_rows[s:e], {c: all_images[c][s:e] for c in DATASET_CAMERAS})
            done += n_full
            leftover = n_full < len(all_rows)
            pending_rows[:] = [all_rows[n_full:]] if leftover else []
            for c in DATASET_CAMERAS:
                pending_images[c][:] = [all_images[c][n_full:]] if leftover else []
            if time.time() - last_log > 60:
                last_log = time.time()
                rate = done / (time.time() - t0)
                eta_min = (len(rows) - done) / max(rate, 1e-6) / 60
                logger.info(
                    "%s: %d/%d rows (%.0f rows/s, eta %.0f min)", name, done, len(rows), rate, eta_min
                )
    finally:
        stop.set()
    if done != len(rows):
        raise RuntimeError(f"Embedded {done} of {len(rows)} rows")
    for (pooling, cam), array in outs.items():
        array.flush()
        tmp = Path(array.filename)
        del array
        tmp.rename(root / f"{cam}.{pooling}.npy")
    np.save(root / "rows.npy", rows)
    logger.info("%s: %d rows in %.1f min -> %s", name, len(rows), (time.time() - t0) / 60, root)
    del embedder
    torch.cuda.empty_cache()
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--poolings", default=None, help="comma-separated subset, e.g. img,all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--taps", default="", help="extra hidden states, e.g. vision,L6,L12 (pooled as grid2@<tap>)"
    )
    parser.add_argument("--tasks", default=None, help="dataset task indices, e.g. 0,5,12 (pilot subsets)")
    parser.add_argument("--tag_suffix", default="", help="cache directory suffix, e.g. _pilot")
    args = parser.parse_args()
    for name in args.names:
        extract(
            name,
            args.stride,
            args.batch_size,
            poolings=tuple(args.poolings.split(",")) if args.poolings else None,
            overwrite=args.overwrite,
            taps=tuple(t for t in args.taps.split(",") if t),
            tasks=[int(t) for t in args.tasks.split(",")] if args.tasks else None,
            tag_suffix=args.tag_suffix,
        )


if __name__ == "__main__":
    main()
