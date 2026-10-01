"""Is there a rendering gap between live LIBERO observations and the (AV1-compressed) dataset frames?

Compares, per retrieval modality, the nearest-neighbour distance (within the prompted task) of
  * live queries recorded during simulator rollouts, and
  * dataset frames queried leave-one-episode-out against the other demonstrations.
Both are "new episode" queries; if live queries sit systematically farther from the memory, the
difference is attributable to the observation pipeline (render resolution, compression, lighting).

    uv run python -m lerobot_policy_knn.experiments.domain_gap --config=best.json --workers=14
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..configuration_knn import KNNConfig
from ..encoders import finalize_features
from ..modeling_knn import KNNPolicy, camera_key_name, featurize_state
from ..processor_knn import make_knn_pre_post_processors
from .libero_pool import LiberoPool, make_jobs
from .offline import query_rows
from .search import MemoryFactory


@torch.no_grad()
def nn_distance(
    policy: KNNPolicy, name: str, q: torch.Tensor, task: torch.Tensor, exclude: torch.Tensor | None
) -> torch.Tensor:
    """Per-query minimal squared distance of one modality to keys of the same task."""
    keys = getattr(policy, name)
    out = torch.empty(len(q))
    offsets = policy.task_offsets.cpu().tolist()
    for i in range(0, len(q), 256):
        qb = q[i : i + 256].to(policy.device)
        tb = task[i : i + 256].to(policy.device)
        if name == "key_state":
            d = torch.cdist(qb.float(), keys.float()).pow(2) / keys.shape[-1]
        else:
            d = (2.0 - 2.0 * torch.mm(qb.half(), keys.T, out_dtype=torch.float32)).clamp_min(0.0)
        key_task = torch.searchsorted(
            torch.tensor(offsets[1:], device=policy.device),
            torch.arange(len(keys), device=policy.device),
            right=True,
        )
        d.masked_fill_(key_task[None] != tb[:, None], float("inf"))
        if exclude is not None:
            d.masked_fill_(
                policy.key_episode[None] == exclude[i : i + 256].to(policy.device)[:, None], float("inf")
            )
        out[i : i + 256] = d.min(-1).values.cpu()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON dict of KNNConfig overrides")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--init_states", default="0:2")
    parser.add_argument("--out", default="outputs/knn/domain_gap.json")
    args = parser.parse_args()
    overrides = json.loads(Path(args.config).read_text())
    cfg = KNNConfig(device="cuda", **overrides)
    factory = MemoryFactory()
    policy = factory.build(cfg)
    policy.query_log = []
    lo, hi = (int(x) for x in args.init_states.split(":"))
    pre, post = make_knn_pre_post_processors(cfg)
    with LiberoPool(n_workers=args.workers) as pool:
        results = pool.evaluate(policy, pre, post, make_jobs(init_states=range(lo, hi)), progress=False)
    live = {k: torch.cat([e["query"][k] for e in policy.query_log]) for k in policy.query_log[0]["query"]}
    live_task = torch.cat([e["task"] for e in policy.query_log])

    frames = factory.frames
    rows = query_rows(frames, every=10)
    ds: dict[str, torch.Tensor] = {}
    for cam in cfg.active_cameras:
        raw = factory.load_raw(cfg.image_encoder, cam, cfg.image_pooling, cfg.image_resolution)
        ds[camera_key_name(cam)] = finalize_features(
            torch.as_tensor(np.asarray(raw[rows])), cfg.image_pooling
        )
    if cfg.state_weight > 0:
        state = featurize_state(torch.from_numpy(frames.state[rows]), cfg.state_axis_angle_start)
        ds["key_state"] = (state - policy.state_mean.cpu()) / policy.state_std.cpu()
    ds_task = torch.from_numpy(frames.task_index[rows])
    ds_ep = torch.from_numpy(frames.episode_index[rows])

    live_t = torch.cat([e["t"] for e in policy.query_log])
    first = np.nonzero(frames.frame_index == 0)[0]
    ds_first: dict[str, torch.Tensor] = {}
    for cam in cfg.active_cameras:
        raw = factory.load_raw(cfg.image_encoder, cam, cfg.image_pooling, cfg.image_resolution)
        ds_first[camera_key_name(cam)] = finalize_features(
            torch.as_tensor(np.asarray(raw[first])), cfg.image_pooling
        )
    if cfg.state_weight > 0:
        state = featurize_state(torch.from_numpy(frames.state[first]), cfg.state_axis_angle_start)
        ds_first["key_state"] = (state - policy.state_mean.cpu()) / policy.state_std.cpu()

    report: dict[str, dict[str, float]] = {"success": {"rate": float(np.mean([r.success for r in results]))}}
    for name in live:
        d_live = nn_distance(policy, name, live[name], live_task, None)
        d_ds = nn_distance(policy, name, ds[name], ds_task, ds_ep)
        t0 = live_t == 0
        d_live0 = nn_distance(policy, name, live[name][t0], live_task[t0], None)
        d_ds0 = nn_distance(
            policy,
            name,
            ds_first[name],
            torch.from_numpy(frames.task_index[first]),
            torch.from_numpy(frames.episode_index[first]),
        )
        q = [0.25, 0.5, 0.75]
        report[name] = {
            **{f"live_q{int(x * 100)}": float(d_live.quantile(x)) for x in q},
            **{f"dataset_loeo_q{int(x * 100)}": float(d_ds.quantile(x)) for x in q},
            "median_ratio": float(d_live.median() / d_ds.median().clamp_min(1e-9)),
            "first_frame_live_q50": float(d_live0.median()),
            "first_frame_dataset_loeo_q50": float(d_ds0.median()),
            "first_frame_ratio": float(d_live0.median() / d_ds0.median().clamp_min(1e-9)),
        }
    print(json.dumps(report, indent=1))
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
