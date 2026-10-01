"""Offline leave-one-episode-out (LOEO) retrieval error: a cheap proxy for comparing embeddings.

For query frames of every demonstration, retrieve neighbours from the *other* demonstrations of the same
task (exactly the policy's retrieval code path) and compare the aggregated action chunk to the actions
that actually followed. It says nothing about compounding errors in closed loop, so it is only used to
shortlist encoders/poolings before simulator evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

from ..configuration_knn import KNNConfig
from ..encoders import finalize_features
from ..features import DEFAULT_CACHE_DIR, FrameIndex, load_frame_index
from ..memory import FeatureLoader, cache_loader, make_policy_from_cache
from ..modeling_knn import KNNPolicy, camera_key_name, featurize_state


def query_rows(frames: FrameIndex, every: int = 10, tasks: list[int] | None = None) -> np.ndarray:
    keep = frames.frame_index % every == every // 2
    if tasks is not None:
        keep &= np.isin(frames.task_index, tasks)
    return np.nonzero(keep)[0]


@torch.no_grad()
def loeo_metrics(
    policy: KNNPolicy,
    frames: FrameIndex,
    load_raw: FeatureLoader,
    rows: np.ndarray,
    horizon: int = 8,
    batch: int = 256,
) -> dict[str, float]:
    cfg = policy.config
    device = policy.device
    rows_t = torch.from_numpy(rows)
    raws = {
        cam: load_raw(cfg.image_encoder, cam, cfg.image_pooling, cfg.image_resolution)
        for cam in cfg.active_cameras
    }
    state_all = featurize_state(torch.from_numpy(frames.state), cfg.state_axis_angle_start)
    actions = torch.from_numpy(frames.action).to(device)
    ep_end = torch.from_numpy(frames.episode_end).to(device)
    sq_err, grip_ok, pos_err, n = 0.0, 0.0, 0.0, 0
    for s in range(0, len(rows), batch):
        r = rows_t[s : s + batch]
        rn = r.numpy()
        q: dict[str, torch.Tensor] = {}
        for cam, raw in raws.items():
            q[camera_key_name(cam)] = finalize_features(
                torch.as_tensor(np.asarray(raw[rn])).to(device), cfg.image_pooling
            )
        if cfg.state_weight > 0:
            q["key_state"] = ((state_all[r].to(device)) - policy.state_mean) / policy.state_std
        t = torch.from_numpy(frames.frame_index[rn]).to(device)
        task = (
            torch.from_numpy(frames.task_index[rn]).to(device) if cfg.task_conditioning == "filter" else None
        )
        excl = torch.from_numpy(frames.episode_index[rn]).to(device)
        idx, w, _ = policy.retrieve(q, t, task, exclude_episode=excl)
        pred = policy.aggregate(idx, w)[:, :horizon]
        gt_idx = torch.minimum(
            r.to(device)[:, None] + torch.arange(horizon, device=device), ep_end[r.to(device)][:, None]
        )
        gt = actions[gt_idx]
        g = cfg.gripper_index
        sq_err += float((pred[..., :g] - gt[..., :g]).pow(2).mean(-1).sum())
        pos_err += float((pred[..., :3] - gt[..., :3]).pow(2).mean(-1).sum())
        grip_ok += float((torch.sign(pred[..., g]) == torch.sign(gt[..., g])).float().sum())
        n += pred.shape[0] * pred.shape[1]
    return {"mse": sq_err / n, "pos_mse": pos_err / n, "gripper_acc": grip_ok / n}


def evaluate_config(
    config: KNNConfig,
    repo_id: str = "lerobot/libero",
    cache_dir=DEFAULT_CACHE_DIR,
    every: int = 10,
    tasks: list[int] | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    frames = load_frame_index(repo_id, cache_dir=cache_dir)
    policy = make_policy_from_cache(config, repo_id, cache_dir, device=device)
    return loeo_metrics(
        policy,
        frames,
        cache_loader(cache_dir, repo_id),
        query_rows(frames, every, tasks),
        config.n_action_steps,
    )
