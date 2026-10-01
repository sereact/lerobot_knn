"""How much of the live-vs-dataset gap is the dataset's AV1 compression, and which cheap query-time
transform best imitates it?

Replays demonstration actions in the live simulator (so frames are realistic), round-trips the frames
through LeRobot's default video encoder (SVT-AV1, crf 30, GOP 2, yuv420p), and compares frozen-encoder
features of raw / transformed / compressed frames. Distances are in the retrieval metric
(squared L2 between unit-norm keys) and reported next to the typical nearest-neighbour distance.

    uv run python -m lerobot_policy_knn.experiments.codec_gap --encoder=resnet18 --pooling=grid2
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..encoders import ImageEncoder, finalize_features
from ..features import load_frame_index
from ..transforms import apply_query_transform


def av1_roundtrip(frames: np.ndarray, crf: int = 30, gop: int = 2, preset: int = 12) -> np.ndarray:
    """(T, H, W, 3) uint8 -> the same frames after an SVT-AV1 encode/decode (LeRobot defaults)."""
    from torchcodec.decoders import VideoDecoder

    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f"{tmp}/{i:05d}.png")
        out = f"{tmp}/out.mp4"
        cmd = [
            "ffmpeg", "-loglevel", "error", "-framerate", "10", "-i", f"{tmp}/%05d.png",
            "-c:v", "libsvtav1", "-crf", str(crf), "-g", str(gop), "-preset", str(preset),
            "-pix_fmt", "yuv420p", out,
        ]  # fmt: skip
        subprocess.run(cmd, check=True)  # noqa: S603
        decoded = VideoDecoder(out).get_frames_in_range(0, len(frames)).data  # (T, 3, H, W)
    return decoded.permute(0, 2, 3, 1).numpy()


def collect_frames(suite_name: str, task_id: int, n_steps: int = 120) -> dict[str, np.ndarray]:
    """Replay the actions of one demonstration of the task open-loop and record flipped frames."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(suite_name)
    env = LiberoEnv(
        suite, task_id, suite_name, obs_type="pixels_agent_pos", observation_height=256, observation_width=256
    )
    frames_idx = load_frame_index("lerobot/libero")
    task = suite.get_task(task_id).language
    tid = frames_idx.task_names.index(task)
    ep = frames_idx.episode_index[frames_idx.task_index == tid][0]
    actions = frames_idx.action[frames_idx.episode_index == ep][:n_steps]
    obs, _ = env.reset(seed=0)
    out: dict[str, list[np.ndarray]] = {"image": [], "image2": []}
    for a in actions:
        for cam in out:
            out[cam].append(obs["pixels"][cam][::-1, ::-1].copy())  # LiberoProcessorStep flips H and W
        obs, *_ = env.step(a)
    env.close()
    return {cam: np.stack(v) for cam, v in out.items()}


@torch.no_grad()
def keys(encoder: ImageEncoder, frames: np.ndarray, pooling: str, transform: str = "none") -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda() / 255.0
    x = apply_query_transform(x, transform)
    return finalize_features(encoder(x)[pooling], pooling)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--pooling", default="grid2")
    parser.add_argument(
        "--transforms", default="none,blur:0.5,blur:0.75,blur:1.0,jpeg:30,jpeg:50,jpeg:70,yuv420"
    )
    parser.add_argument("--out", default="outputs/knn/codec_gap.json")
    args = parser.parse_args()
    encoder = ImageEncoder(args.encoder).cuda()
    results: dict[str, dict[str, float]] = {}
    tasks = [("libero_spatial", 0), ("libero_object", 3), ("libero_goal", 1), ("libero_10", 2)]
    per_cam: dict[str, dict[str, list[float]]] = {}
    for suite, task_id in tasks:
        frames = collect_frames(suite, task_id)
        for cam, raw in frames.items():
            compressed = av1_roundtrip(raw)
            target = keys(encoder, compressed, args.pooling)
            for t in args.transforms.split(","):
                d = (keys(encoder, raw, args.pooling, t) - target).pow(2).sum(-1)
                per_cam.setdefault(cam, {}).setdefault(t, []).extend(d.cpu().tolist())
            # pixel-level error of the codec, for reference
            per_cam[cam].setdefault("_pixel_mae", []).append(
                float(np.abs(raw.astype(float) - compressed).mean())
            )
    for cam, by_t in per_cam.items():
        results[cam] = {t: float(np.median(v)) for t, v in by_t.items()}
    print(json.dumps(results, indent=1))
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
