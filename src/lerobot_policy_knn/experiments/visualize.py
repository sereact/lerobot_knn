"""Look inside the retrieval: 2-D maps (PCA / t-SNE) of a task's memory with live rollouts on top, plus
galleries of what each query retrieved.

The memory keys and live queries are embedded in the policy's own weighted metric, i.e. the vector

    [sqrt(w_cam / s_cam) * f_cam for each camera,  sqrt(w_state / (s_state * D)) * state]

whose squared Euclidean distances are exactly the retrieval distances (without time/text terms), so
neighbourhoods in the maps are what the policy sees.

    uv run python -m lerobot_policy_knn.experiments.visualize --config=outputs/knn/best_v1.json \\
        --suite=libero_10 --task_id=0 --init_states=0:6
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..configuration_knn import KNNConfig
from ..modeling_knn import KNNPolicy, camera_key_name
from ..processor_knn import make_knn_pre_post_processors
from .search import MemoryFactory

OUT_DIR = Path("outputs/knn/viz")


def metric_space(policy: KNNPolicy, feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate modality features scaled so that squared L2 == the policy's retrieval distance."""
    cfg = policy.config
    scales = policy._scales()
    parts = []
    for cam in cfg.active_cameras:
        name = camera_key_name(cam)
        parts.append(feats[name].float() * (cfg.camera_weights[cam] / scales.get(name, 1.0)) ** 0.5)
    if cfg.state_weight > 0 and "key_state" in feats:
        s = feats["key_state"].float()
        parts.append(s * (cfg.state_weight / scales.get("key_state", 1.0) / s.shape[-1]) ** 0.5)
    return torch.cat(parts, -1)


def thumb(img: np.ndarray | torch.Tensor, size: int = 96) -> str:
    if isinstance(img, torch.Tensor):
        img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).resize((size, size), Image.BILINEAR).save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def rollout(policy: KNNPolicy, suite_name: str, task_id: int, init_state: int) -> dict:
    """One episode with the exact lerobot-eval observation path; logs every retrieval and its images."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    from .libero_pool import _stack_obs

    suite = _get_suite(suite_name)
    env = LiberoEnv(
        suite, task_id, suite_name, obs_type="pixels_agent_pos", observation_height=256,
        observation_width=256, episode_index=init_state,
    )  # fmt: skip
    env_pre, env_post = LiberoEnvConfig(observation_height=256, observation_width=256).get_env_processors()
    pre, post = make_knn_pre_post_processors(policy.config)
    policy.reset()
    policy.query_log = []
    obs, _ = env.reset(seed=1000 + init_state)
    images, success, steps = [], False, 0
    for step in range(1, TASK_SUITE_MAX_STEPS[suite_name] + 1):
        steps = step
        batch = preprocess_observation(_stack_obs([obs]))
        batch["task"] = [env.task_description]
        batch = pre(env_pre(batch))
        n_logged = len(policy.query_log)
        action = policy.select_action(batch)
        if len(policy.query_log) > n_logged:  # a retrieval happened at this step
            images.append({cam: batch[cam][0].cpu() for cam in policy.config.active_cameras})
        action = env_post({ACTION: post(action)})[ACTION]
        obs, _, terminated, _, info = env.step(action[0].numpy())
        if info.get("is_success") or terminated:
            success = bool(info.get("is_success"))
            break
    env.close()
    log, policy.query_log = policy.query_log, None
    return {"log": log, "images": images, "success": success, "steps": steps, "task": env.task_description}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--init_states", default="0:6")
    parser.add_argument("--max_memory_points", type=int, default=6000)
    parser.add_argument("--gallery_moments", type=int, default=6)
    parser.add_argument("--neighbours_shown", type=int, default=4)
    args = parser.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from lerobot.datasets import LeRobotDataset

    cfg = KNNConfig(device="cuda", **json.loads(Path(args.config).read_text()))
    factory = MemoryFactory()
    policy = factory.build(cfg)
    lo, hi = (int(x) for x in args.init_states.split(":"))
    episodes = [rollout(policy, args.suite, args.task_id, i) | {"init_state": i} for i in range(lo, hi)]
    task_name = episodes[0]["task"]
    tid = cfg.task_names.index(task_name)
    a, b = int(policy.task_offsets[tid]), int(policy.task_offsets[tid + 1])

    # memory of this task, subsampled for the projection
    rng = np.random.default_rng(0)
    mem_rows = np.arange(a, b)
    if len(mem_rows) > args.max_memory_points:
        mem_rows = np.sort(rng.choice(mem_rows, args.max_memory_points, replace=False))
    mem_feats = {
        name: getattr(policy, name)[mem_rows]
        for name in [camera_key_name(c) for c in cfg.active_cameras]
        + (["key_state"] if cfg.state_weight > 0 else [])
    }
    mem_x = metric_space(policy, mem_feats).cpu().numpy()
    frames = factory.frames
    ptr = policy.key_ptr[mem_rows].cpu().numpy()
    progress = frames.frame_index[ptr] / np.maximum(frames.episode_length[ptr] - 1, 1)

    q_x, q_meta = [], []
    for e, ep in enumerate(episodes):
        for entry in ep["log"]:
            q_x.append(
                metric_space(policy, {k: v.to(policy.device) for k, v in entry["query"].items()})
                .cpu()
                .numpy()
            )
            q_meta.append((e, int(entry["t"][0]), entry["rows"][0].numpy(), entry["dists"][0].numpy()))
    q_x = np.concatenate(q_x)
    joint = np.concatenate([mem_x, q_x])
    reduced = PCA(n_components=min(50, joint.shape[1])).fit_transform(joint)
    tsne = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(reduced)
    pca2 = reduced[:, :2]

    row_to_point = {int(r): i for i, r in enumerate(mem_rows)}
    n_mem = len(mem_rows)

    def pack(proj: np.ndarray) -> dict:
        mem = [[round(float(x), 3), round(float(y), 3), round(float(p), 3), int(frames.episode_index[q])]
               for (x, y), p, q in zip(proj[:n_mem], progress, ptr, strict=True)]  # fmt: skip
        queries, links = [], []
        for j, (e, t, rows, _d) in enumerate(q_meta):
            x, y = proj[n_mem + j]
            queries.append([round(float(x), 3), round(float(y), 3), t, e])
            for r in rows[: args.neighbours_shown]:
                if int(r) in row_to_point:
                    links.append([j, row_to_point[int(r)]])
        return {"memory": mem, "queries": queries, "links": links}

    # galleries: a few evenly spaced retrievals per episode, query frame + retrieved demo frames
    dataset = LeRobotDataset("lerobot/libero")
    galleries = []
    for ep in episodes:
        n = len(ep["log"])
        moments = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(args.gallery_moments, n))})
        items = []
        for m in moments:
            entry = ep["log"][m]
            rows = entry["rows"][0][: args.neighbours_shown]
            neighbours = []
            for r in rows.tolist():
                g = int(policy.key_ptr[r])
                item = dataset[g]
                neighbours.append({
                    "episode": int(frames.episode_index[g]),
                    "frame": int(frames.frame_index[g]),
                    "images": {c.rsplit(".", 1)[-1]: thumb(item[c]) for c in cfg.active_cameras},
                })  # fmt: skip
            items.append({
                "t": int(entry["t"][0]),
                "query": {c.rsplit(".", 1)[-1]: thumb(ep["images"][m][c]) for c in cfg.active_cameras},
                "dists": [round(float(d), 4) for d in entry["dists"][0][: args.neighbours_shown]],
                "neighbours": neighbours,
            })  # fmt: skip
        galleries.append(
            {"init_state": ep["init_state"], "success": ep["success"], "steps": ep["steps"], "moments": items}
        )

    out = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task": task_name,
        "config": json.loads(Path(args.config).read_text()),
        "episodes": [
            {"init_state": ep["init_state"], "success": ep["success"], "steps": ep["steps"]}
            for ep in episodes
        ],
        "tsne": pack(tsne),
        "pca": pack(pca2),
        "pca_explained": [
            round(float(v), 4) for v in PCA(n_components=2).fit(reduced).explained_variance_ratio_
        ],
        "galleries": galleries,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.suite}_{args.task_id}.json"
    path.write_text(json.dumps(out))
    print(path, [(ep["init_state"], ep["success"], ep["steps"]) for ep in episodes])


if __name__ == "__main__":
    main()
