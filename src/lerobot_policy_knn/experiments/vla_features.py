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

from ..features import DEFAULT_CACHE_DIR, _cache_root, _episode_video_layout, load_frame_index
from ..vla import DATASET_CAMERAS, VLAEmbedder, cache_tag

logger = logging.getLogger(__name__)
REPO_ID = "lerobot/libero"


def _decode_worker(meta, frames, rows, out_q, stop):
    """Yield (global rows, {camera: uint8 (n, 3, H, W)}) per episode, in dataset order."""
    from torchcodec.decoders import VideoDecoder

    layouts = {}
    for cam in DATASET_CAMERAS:
        where = {}
        for file_index, eps in _episode_video_layout(meta, cam).items():
            for start, gfrom, length in eps:
                where[gfrom] = (file_index, start, length)
        layouts[cam] = where
    decoders: dict[str, tuple[int, VideoDecoder]] = {}
    starts = np.nonzero(frames.frame_index == 0)[0]  # global index of every episode's first frame
    by_ep = np.searchsorted(starts, rows, side="right") - 1
    for e in np.unique(by_ep):
        if stop.is_set():
            return
        ep_rows = rows[by_ep == e]
        gfrom = int(starts[e])
        imgs = {}
        for cam in DATASET_CAMERAS:
            file_index, start, _length = layouts[cam][gfrom]
            cur = decoders.get(cam)
            if cur is None or cur[0] != file_index:
                video = meta.root / f"videos/{cam}/chunk-000/file-{file_index:03d}.mp4"
                decoders[cam] = (file_index, VideoDecoder(str(video), device="cpu"))
            local = (ep_rows - gfrom + start).tolist()
            imgs[cam] = decoders[cam][1].get_frames_at(indices=local).data
        out_q.put((ep_rows, imgs))
    out_q.put(None)


@torch.no_grad()
def extract(name: str, stride: int = 2, batch_size: int = 64, cache_dir: Path = DEFAULT_CACHE_DIR,
            poolings: tuple[str, ...] | None = None, device: str = "cuda", overwrite: bool = False,
            taps: tuple[str, ...] = (), tasks: list[int] | None = None, tag_suffix: str = "") -> Path:  # fmt: skip
    frames = load_frame_index(REPO_ID, cache_dir=cache_dir)
    meta = LeRobotDatasetMetadata(REPO_ID)
    root = _cache_root(cache_dir, REPO_ID) / (cache_tag(name) + tag_suffix)
    wanted = poolings
    if (root / "rows.npy").exists() and not overwrite:
        have = {f.name.split(".")[-2] for f in root.glob("*.npy") if f.name != "rows.npy"}
        if wanted is None or set(wanted) <= have:
            logger.info("%s already cached at %s", name, root)
            return root
    keep = frames.frame_index % stride == 0
    if tasks is not None:  # a subset of tasks (pilot studies)
        keep &= np.isin(frames.task_index, tasks)
    rows = np.nonzero(keep)[0]
    emb = VLAEmbedder(name, device=device, taps=taps)
    poolings = poolings or emb.poolings
    root.mkdir(parents=True, exist_ok=True)
    outs: dict[tuple[str, str], np.memmap] = {}
    for p in poolings:
        cams = emb.present if p in ("img", "grid2") or p.startswith("grid2@") else ["observation.vla"]
        dim = emb.hidden_size * (4 if p.startswith("grid2") else 1)
        for cam in cams:
            outs[(p, cam)] = open_memmap(
                str(root / f"{cam}.{p}.tmp.npy"), mode="w+", dtype=np.float16, shape=(len(rows), dim)
            )
    pos = {int(r): i for i, r in enumerate(rows)}
    q: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    th = threading.Thread(target=_decode_worker, args=(meta, frames, rows, q, stop), daemon=True)
    th.start()
    buf_rows: list[np.ndarray] = []
    buf_imgs: dict[str, list[torch.Tensor]] = {c: [] for c in DATASET_CAMERAS}
    done, t0, last_log = 0, time.time(), 0.0

    def embed(model: VLAEmbedder, r: np.ndarray, imgs: dict[str, torch.Tensor]) -> None:
        batch = {c: imgs[c].to(device, non_blocking=True).float().div_(255.0) for c in DATASET_CAMERAS}
        batch[OBS_STATE] = torch.from_numpy(frames.state[r]).float().to(device)
        batch["task"] = [frames.task_names[int(t)] for t in frames.task_index[r]]
        feats = model(batch)
        idx = np.array([pos[int(x)] for x in r])
        for (p, cam), arr in outs.items():
            arr[idx] = feats[p][cam].half().cpu().numpy()

    try:
        finished = False
        while not finished:
            item = q.get()
            if item is None:
                finished = True
            else:
                buf_rows.append(item[0])
                for c in DATASET_CAMERAS:
                    buf_imgs[c].append(item[1][c])
            n_buf = sum(len(x) for x in buf_rows)
            if n_buf < batch_size and not (finished and n_buf):
                continue
            allr = np.concatenate(buf_rows)
            alli = {c: torch.cat(buf_imgs[c]) for c in DATASET_CAMERAS}
            n_full = len(allr) if finished else (len(allr) // batch_size) * batch_size
            for s in range(0, n_full, batch_size):
                e = min(s + batch_size, n_full)
                embed(emb, allr[s:e], {c: alli[c][s:e] for c in DATASET_CAMERAS})
            done += n_full
            buf_rows[:] = [allr[n_full:]] if n_full < len(allr) else []
            for c in DATASET_CAMERAS:
                buf_imgs[c][:] = [alli[c][n_full:]] if n_full < len(allr) else []
            if time.time() - last_log > 60:
                last_log = time.time()
                rate = done / (time.time() - t0)
                logger.info("%s: %d/%d rows (%.0f rows/s, eta %.0f min)", name, done, len(rows), rate,
                            (len(rows) - done) / max(rate, 1e-6) / 60)  # fmt: skip
    finally:
        stop.set()
    if done != len(rows):
        raise RuntimeError(f"Embedded {done} of {len(rows)} rows")
    for (p, cam), arr in outs.items():
        arr.flush()
        tmp = Path(arr.filename)
        del arr
        tmp.rename(root / f"{cam}.{p}.npy")
    np.save(root / "rows.npy", rows)
    logger.info("%s: %d rows in %.1f min -> %s", name, len(rows), (time.time() - t0) / 60, root)
    del emb
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
    pools = tuple(args.poolings.split(",")) if args.poolings else None
    taps = tuple(t for t in args.taps.split(",") if t)
    tasks = [int(t) for t in args.tasks.split(",")] if args.tasks else None
    for n in args.names:
        extract(n, args.stride, args.batch_size, poolings=pools, overwrite=args.overwrite, taps=taps, tasks=tasks,
                tag_suffix=args.tag_suffix)  # fmt: skip


if __name__ == "__main__":
    main()
