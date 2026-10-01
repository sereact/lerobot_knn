"""Is a diffusion policy "secretly kNN"? Compare its action chunks with the training data and with retrieval.

Rolls out a trained LeRobot policy (e.g. multi_task_dit) from LIBERO initial states it never saw and records
every action chunk it generates. For each chunk (first ``--horizon`` actions) it measures, in per-element
squared action units:

* ``nn_train``: distance to the nearest training chunk of the same task (does it output something it has seen?)
* ``knn_policy``: distance to the chunk the KNN policy retrieves for the same observation
* ``knn_top1``: distance to the chunk that followed the KNN policy's single nearest observation-neighbour
* ``obs_neighbour_overlap``: whether the training frame whose chunk is nearest in *action* space is among the KNN
  policy's k nearest frames in *observation* space (i.e. "it chose the action of an observation neighbour")

and the reference scale ``real_novelty``: the same nearest-training-chunk distance for real demonstration chunks
measured leave-one-demonstration-out, i.e. how close genuinely new demonstrations come to the others.

    uv run python -m lerobot_policy_knn.experiments.memorization --policy_path=outputs/dit_libero/checkpoints/last/pretrained_model \\
        --knn_config=outputs/knn/v2_39.json --init_states=0:2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..configuration_knn import KNNConfig
from .libero_pool import SUITES, _stack_obs
from .search import OUT, MemoryFactory


def chunk_table(frames, horizon: int) -> np.ndarray:
    """(N, horizon, A) action chunks following every dataset frame (held at the end of its episode)."""
    idx = np.arange(len(frames))[:, None] + np.arange(horizon)[None]
    idx = np.minimum(idx, frames.episode_end[:, None])
    return frames.action[idx]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--knn_config", required=True)
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--init_states", default="0:2")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--out", default=str(OUT / "memorization.json"))
    parser.add_argument("--device", default="cuda", help="device of the evaluated policy")
    parser.add_argument("--knn_device", default="cuda")
    parser.add_argument("--tasks", default=None, help="restrict to task ids, e.g. 0:2 (smoke tests)")
    args = parser.parse_args()
    dev = args.device
    os.environ.setdefault("MUJOCO_GL", "egl")

    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    env_cfg = LiberoEnvConfig(observation_height=256, observation_width=256)
    env_pre, env_post = env_cfg.get_env_processors()
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = Path(args.policy_path)
    cfg.device = dev
    if hasattr(cfg, "num_inference_steps"):
        cfg.num_inference_steps = args.num_inference_steps
        if hasattr(cfg, "noise_scheduler_type"):
            cfg.noise_scheduler_type = "DDIM"
    policy = make_policy(cfg, env_cfg=env_cfg).eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=args.policy_path, preprocessor_overrides={"device_processor": {"device": dev}}
    )

    kdev = args.knn_device
    factory = MemoryFactory(device=kdev)
    frames = factory.frames
    knn = factory.build(KNNConfig(device=kdev, **json.loads(Path(args.knn_config).read_text())))
    knn_chunks_all = torch.from_numpy(chunk_table(frames, args.horizon)).to(kdev)  # (N, H, A)
    task_rows = {t: np.nonzero(frames.task_index == t)[0] for t in np.unique(frames.task_index)}

    records = []
    lo, hi = (int(x) for x in args.init_states.split(":"))
    for suite_name in args.suites.split(","):
        suite = _get_suite(suite_name)
        task_range = range(len(suite.tasks))
        if args.tasks:
            t_lo, t_hi = (int(x) for x in args.tasks.split(":"))
            task_range = range(t_lo, t_hi)
        for task_id in task_range:
            for init in range(lo, hi):
                env = LiberoEnv(
                    suite, task_id, suite_name, obs_type="pixels_agent_pos", observation_height=256,
                    observation_width=256, episode_index=init,
                )  # fmt: skip
                obs, _ = env.reset(seed=1000 + init)
                policy.reset()
                pre.reset()
                tid = knn.config.task_names.index(env.task_description)
                rows = torch.from_numpy(task_rows[tid]).to(kdev)
                success = False
                for _step in range(TASK_SUITE_MAX_STEPS[suite_name]):
                    raw = preprocess_observation(_stack_obs([obs]))
                    raw["task"] = [env.task_description]
                    raw = env_pre(raw)
                    queue_before = (
                        policy.count_queued_actions() if hasattr(policy, "count_queued_actions") else 0
                    )
                    action = policy.select_action(pre(dict(raw)))
                    if queue_before == 0:
                        # a new chunk was generated: the returned action plus what is now queued
                        queued = []
                        for attr in ("_queues", "_action_queue"):
                            q = getattr(policy, attr, None)
                            if isinstance(q, dict) and ACTION in q:
                                queued = list(q[ACTION])
                            elif q is not None and not isinstance(q, dict):
                                queued = list(q)
                        chunk = torch.stack([action[0], *[a[0] for a in queued]])[: args.horizon]
                        chunk = post(chunk[None])[0] if chunk.ndim == 2 else chunk
                        chunk = chunk.float().to(kdev)
                        cands = knn_chunks_all[rows]
                        d_train = (cands - chunk[None, : cands.shape[1]]).pow(2).mean(dim=(1, 2))
                        nn_train, nn_idx = d_train.min(0)
                        nn_frame = int(rows[nn_idx])
                        query = knn.encode_observation(
                            {k: v.to(kdev) if torch.is_tensor(v) else v for k, v in raw.items()}
                        )
                        t_now = knn._as_device_tensor([_step])
                        krows, kw, _ = knn.retrieve(query, t_now, knn.task_ids([env.task_description]))
                        kchunk = knn.aggregate(krows, kw)[0, : chunk.shape[0]]
                        top1 = knn.gather_chunks(krows[:, :1], args.horizon)[0, 0, : chunk.shape[0]]
                        # yardstick: how far kNN's own averaged chunk is from its individual neighbours' chunks
                        neigh = knn.gather_chunks(krows, args.horizon)[0, :, : chunk.shape[0]]  # (k, H, A)
                        spread = (
                            (neigh[..., :-1] - kchunk[None, :, :-1]).pow(2).mean(dim=(1, 2)) * kw[0]
                        ).sum()
                        neigh_frames = set(knn.key_ptr[krows[0]].tolist())
                        neigh_eps = {int(frames.episode_index[f]) for f in neigh_frames}
                        motion = slice(0, chunk.shape[-1] - 1)
                        d_motion = (
                            (cands[..., motion] - chunk[None, : cands.shape[1], motion])
                            .pow(2)
                            .mean(dim=(1, 2))
                        )
                        records.append({
                            "suite": suite_name, "task_id": task_id, "init": init, "t": _step,
                            "nn_train": float(nn_train), "knn_policy": float((kchunk - chunk).pow(2).mean()),
                            "knn_top1": float((top1 - chunk).pow(2).mean()),
                            "nn_train_motion": float(d_motion.min()),
                            "knn_policy_motion": float((kchunk[:, motion] - chunk[:, motion]).pow(2).mean()),
                            "knn_top1_motion": float((top1[:, motion] - chunk[:, motion]).pow(2).mean()),
                            "knn_neighbour_spread_motion": float(spread),
                            "gripper_agree": float((torch.sign(kchunk[:, -1]) == torch.sign(chunk[:, -1])).float().mean()),
                            "obs_neighbour_overlap": nn_frame in neigh_frames,
                            "same_demo_as_neighbour": int(frames.episode_index[nn_frame]) in neigh_eps,
                        })  # fmt: skip
                    action = env_post({ACTION: post(action)})[ACTION]
                    obs, _, terminated, _, info = env.step(action[0].cpu().numpy())
                    if info.get("is_success") or terminated:
                        success = bool(info.get("is_success"))
                        break
                env.close()
                for r in records:
                    if r["suite"] == suite_name and r["task_id"] == task_id and r["init"] == init:
                        r["success"] = success
                print(suite_name, task_id, init, "success" if success else "fail", flush=True)

    # reference: how close real demonstration chunks come to other demos' chunks (leave-one-demo-out)
    rng = np.random.default_rng(0)
    real, real_motion = [], []
    for _t, rows_np in task_rows.items():
        pick = rng.choice(rows_np, min(64, len(rows_np)), replace=False)
        cands = knn_chunks_all[torch.from_numpy(rows_np).to(kdev)]
        eps = torch.from_numpy(frames.episode_index[rows_np]).to(kdev)
        for f in pick:
            d = (cands - knn_chunks_all[int(f)][None]).pow(2).mean(dim=(1, 2))
            d[eps == int(frames.episode_index[f])] = float("inf")
            real.append(float(d.min()))
            dm = (cands[..., :-1] - knn_chunks_all[int(f)][None, :, :-1]).pow(2).mean(dim=(1, 2))
            dm[eps == int(frames.episode_index[f])] = float("inf")
            real_motion.append(float(dm.min()))
    summary = {
        "n_chunks": len(records),
        "median_nn_train": float(np.median([r["nn_train"] for r in records])),
        "median_real_novelty": float(np.median(real)),
        "median_real_novelty_motion": float(np.median(real_motion)),
        "median_knn_policy": float(np.median([r["knn_policy"] for r in records])),
        "median_knn_top1": float(np.median([r["knn_top1"] for r in records])),
        "median_nn_train_motion": float(np.median([r["nn_train_motion"] for r in records])),
        "median_knn_policy_motion": float(np.median([r["knn_policy_motion"] for r in records])),
        "median_knn_top1_motion": float(np.median([r["knn_top1_motion"] for r in records])),
        "gripper_agreement": float(np.mean([r["gripper_agree"] for r in records])),
        "median_knn_neighbour_spread_motion": float(
            np.median([r["knn_neighbour_spread_motion"] for r in records])
        ),
        "obs_neighbour_overlap": float(np.mean([r["obs_neighbour_overlap"] for r in records])),
        "same_demo_as_neighbour": float(np.mean([r["same_demo_as_neighbour"] for r in records])),
        "success_rate": float(np.mean([r.get("success", False) for r in records])) if records else None,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "records": records, "real_novelty": real}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
