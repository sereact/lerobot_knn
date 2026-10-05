"""Does a VLA's action head behave like kNN in its own embedding space?

Takes the chunks a VLA predicted during its own closed-loop rollouts (``vla_rollouts --record``) and, for every
embedding space in ``--spaces``, retrieves the ``k`` nearest demonstration frames of the same task and
averages the chunks that followed them (plain kNN: no state, time, history or mode seeking). Reported per
space and ``k`` (motion dimensions, first ``horizon`` actions):

* ``vla_r2``: fraction of the VLA's chunk variance (around its per-task mean) that the kNN answer explains
* ``vla_mse`` / ``vla_top1_mse`` / ``vla_gripper``: distance to the kNN / single-nearest-neighbour chunk
* ``demo_r2`` ...: the same kNN predicting *held-out demonstrations* (leave-one-episode-out), i.e. how good the
  space is at predicting the demonstrator rather than the VLA

plus two reference scales: the VLA's own resampling noise (a second flow-matching sample at the same
observation) and how close the VLA's chunks come to *any* training chunk of the task.

    uv run python -m lerobot_policy_knn.experiments.head_vs_knn outputs/vla/rollouts/vla_smolvla_libero.npz \\
        --vla=smolvla_libero --spaces=dinov2,resnet18_l3,vla:smolvla_base/grid2,vla:smolvla_libero/grid2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE

from ..configuration_knn import KNNConfig
from ..modeling_knn import camera_key_name
from ..vla import DATASET_CAMERAS, VLA_CAMERA, cache_tag, load_vla, resolve_spec
from .search import MemoryFactory

MOTION = slice(0, 6)
IMAGE_SPACES = {
    "dinov2": {"image_encoder": "dinov2_small", "image_pooling": "cls"},
    "resnet18_l3": {"image_encoder": "resnet18_l3", "image_pooling": "grid4"},
    "resnet18_l3_mean": {"image_encoder": "resnet18_l3", "image_pooling": "mean"},
}


def base_task(name: str, tasks) -> str:
    """The LIBERO task a (LIBERO-plus variant) instruction or name derives from: itself, else the longest task
    whose text it starts with or contains (variants append e.g. ``_moved_level5_sample4`` to the task name)."""
    if name in tasks:
        return name
    flat = name.replace("_", " ")
    hits = [t for t in tasks if t in flat]
    if not hits:
        raise KeyError(f"no LIBERO task in {name!r}")
    return max(hits, key=len)


def space_config(space: str, k: int, horizon: int) -> KNNConfig:
    """``<encoder space>[|<key_norm>]``, e.g. ``resnet18_l3``, ``vla:pi05_libero/grid2@L12|zscore``."""
    space, _, key_norm = space.partition("|")
    if space.startswith("vla:"):
        enc, pooling = space.rsplit("/", 1)
        per_camera = pooling == "img" or pooling.startswith("grid2")
        cams = list(DATASET_CAMERAS) if per_camera else [VLA_CAMERA]
        emb = {"image_encoder": enc, "image_pooling": pooling}
    else:
        cams = list(DATASET_CAMERAS)
        emb = dict(IMAGE_SPACES[space])
    return KNNConfig(
        device="cuda", **emb, key_norm=key_norm or "none", camera_weights=dict.fromkeys(cams, 1.0),
        state_weight=0.0, time_weight=0.0,
        task_conditioning="filter", k=k, temperature=0.1, one_per_episode=True, history_weight=0.0,
        chunk_size=horizon, n_action_steps=horizon, memory_stride=2,
    )  # fmt: skip


def r2(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """1 - SSE / SST with SST measured around each group's (task's) mean chunk."""
    sst = 0.0
    for g in np.unique(groups):
        m = groups == g
        sst += float(((y[m] - y[m].mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / max(sst, 1e-12)


def chunk_metrics(y: np.ndarray, p: np.ndarray, top1: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2(y[..., MOTION], p[..., MOTION], groups),
        "mse": float(((y[..., MOTION] - p[..., MOTION]) ** 2).mean()),
        "top1_mse": float(((y[..., MOTION] - top1[..., MOTION]) ** 2).mean()),
        "top1_r2": r2(y[..., MOTION], top1[..., MOTION], groups),
        "gripper": float((np.sign(y[..., 6]) == np.sign(p[..., 6])).mean()),
    }


@torch.no_grad()
def retrieve_chunks(policy, query: dict, t: torch.Tensor, task: torch.Tensor, horizon: int, exclude=None):
    """(kNN answer, nearest neighbour's chunk, distance to the nearest neighbour) per query."""
    rows, w, d = policy.retrieve(query, t, task, exclude_episode=exclude)
    pred = policy.aggregate(rows, w)[:, :horizon]
    top1 = policy.gather_chunks(rows[:, :1], horizon)[:, 0]
    return pred.float().cpu().numpy(), top1.float().cpu().numpy(), d[:, 0].float().cpu().numpy()


@torch.no_grad()
def embed_records(policy, rec: dict, batch: int = 48) -> dict[str, torch.Tensor]:
    """Keys of recorded observations; the batch halves whenever an embedder runs out of GPU memory (e.g. Evo-1's eager
    attention at 448 px)."""
    out: dict[str, list[torch.Tensor]] = {}
    n = len(rec["state"])
    s = 0
    while s < n:
        sl = slice(s, min(n, s + batch))
        b = {
            cam: torch.from_numpy(rec["images"][sl, i]).cuda().float() / 255.0
            for i, cam in enumerate(DATASET_CAMERAS)
        }
        b[OBS_STATE] = torch.from_numpy(rec["state"][sl]).float().cuda()
        b["task"] = [str(rec["task_names"][i]) for i in rec["task"][sl]]
        try:
            keys = policy.encode_observation(b)
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            continue
        for key, v in keys.items():
            out.setdefault(key, []).append(v)
        s = sl.stop
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def demo_queries(policy, factory, rows: np.ndarray) -> dict[str, torch.Tensor]:
    cfg = policy.config
    q = {}
    for cam in cfg.active_cameras:
        raw = factory.load_raw(cfg.image_encoder, cam, cfg.image_pooling, cfg.image_resolution)
        name = camera_key_name(cam)
        q[name] = policy.key_features(name, torch.as_tensor(np.asarray(raw[rows])).cuda())
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts")
    parser.add_argument("--vla", required=True, help="VLA that produced the rollouts (for the noise floor)")
    parser.add_argument("--spaces", required=True)
    parser.add_argument("--ks", default="1,4,8,16,32")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--train_split", default=None, help="split.json: memory = train, demo queries = heldout"
    )
    parser.add_argument("--demo_queries", type=int, default=4000)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--no_references", action="store_true", help="skip the VLA noise floor / nearest-chunk refs"
    )
    parser.add_argument(
        "--no_noise_floor", action="store_true", help="skip only the noise floor (it re-runs the VLA itself)"
    )
    args = parser.parse_args()
    horizon = args.horizon
    ks = [int(k) for k in args.ks.split(",")]
    data = np.load(args.rollouts)
    rec = {k: data[k] for k in data.files}
    # a VLA re-planning every n < horizon steps (VLA-JEPA: 7) recorded n-step chunks: compare that many actions
    horizon = min(horizon, int(rec["chunk"].shape[1]))
    # rollout prompts -> memory task ids
    factory = MemoryFactory()
    frames = factory.frames
    task_of = {name: i for i, name in enumerate(frames.task_names)}
    if (
        "category" in rec
    ):  # LIBERO-plus rollouts: the base task from each variant's name (prompts may be reworded)
        sample = json.loads(Path("outputs/libero_plus/sample.json").read_text())
        name_of = {(s["suite"], s["task_id"]): s["name"] for s in sample}
        names = [name_of[(str(su), int(ti))] for su, ti in zip(rec["suite"], rec["task_id"], strict=True)]
    else:
        names = [str(n) for n in rec["task_names"][rec["task"]]]
    rec_task = np.array([task_of[base_task(n, task_of)] for n in names])
    y_vla = rec["chunk"][:, :horizon]

    episodes = None
    demo_rows_pool = np.nonzero(frames.frame_index % 10 == 4)[0]
    if args.train_split:
        split = json.loads(Path(args.train_split).read_text())
        episodes = np.array(split["train"])
        demo_rows_pool = demo_rows_pool[np.isin(frames.episode_index[demo_rows_pool], split["heldout"])]
    rng = np.random.default_rng(0)
    demo_rows = np.sort(
        rng.choice(demo_rows_pool, min(args.demo_queries, len(demo_rows_pool)), replace=False)
    )
    gt_idx = np.minimum(demo_rows[:, None] + np.arange(horizon), frames.episode_end[demo_rows][:, None])
    y_demo = frames.action[gt_idx]
    demo_task = frames.task_index[demo_rows]

    result: dict = {"rollouts": args.rollouts, "vla": args.vla, "horizon": horizon, "n_chunks": len(y_vla),
                    "vla_success": float(np.mean(rec["success"])), "spaces": {}}  # fmt: skip
    per_record: dict[str, list[float]] = {}
    novelty: dict[str, list[float]] = {}  # top-1 distance to the memory / the typical leave-one-demo-out one
    for space in args.spaces.split(","):
        result["spaces"][space] = {}
        policy = factory.build(space_config(space, max(ks), horizon), episodes=episodes)
        q_vla = embed_records(policy, rec)
        q_demo = demo_queries(policy, factory, demo_rows)
        t_vla = torch.from_numpy(rec["t"]).cuda()
        task_vla = torch.from_numpy(rec_task).cuda()
        t_demo = torch.from_numpy(frames.frame_index[demo_rows]).cuda()
        task_demo = torch.from_numpy(demo_task).cuda()
        excl = torch.from_numpy(frames.episode_index[demo_rows]).cuda() if episodes is None else None
        for k in ks:
            policy.config.k = k
            p, top1, d1 = retrieve_chunks(policy, q_vla, t_vla, task_vla, horizon)
            pd, top1d, d1d = retrieve_chunks(policy, q_demo, t_demo, task_demo, horizon, exclude=excl)
            m = {f"vla_{a}": b for a, b in chunk_metrics(y_vla, p, top1, rec_task).items()}
            m.update({f"demo_{a}": b for a, b in chunk_metrics(y_demo, pd, top1d, demo_task).items()})
            result["spaces"][space][k] = m
            if k == 8:
                per_record[space] = (((y_vla[..., MOTION] - p[..., MOTION]) ** 2).mean((1, 2))).tolist()
                novelty[space] = (d1 / max(float(np.median(d1d)), 1e-9)).tolist()
            print(space, k, {a: round(b, 4) for a, b in m.items()}, flush=True)
        del policy, q_vla, q_demo
        factory._encoders.clear()
        torch.cuda.empty_cache()
    result["per_record_mse_k8"] = per_record
    result["per_record_novelty"] = novelty
    result["per_record"] = {k: [str(x) if k == "category" else x.item() for x in rec[k]]
                            for k in ("success", "episode", "category") if k in rec}  # fmt: skip

    out = Path(args.out or f"outputs/vla/head_vs_knn_{cache_tag(args.vla)}.json")
    if args.no_references:
        out.write_text(json.dumps(result, indent=1))
        print("->", out)
        return
    if not args.no_noise_floor:  # reference 1: the VLA's own sampling noise at the recorded observations
        policy, pre, post = load_vla(args.vla)
        rename = resolve_spec(args.vla).camera_rename

        def predict(b):
            policy.reset()
            return post(policy.predict_action_chunk(pre(b))).float().cpu().numpy()[:, :horizon]

        resampled = []
        for s in range(0, len(y_vla), 32):
            sl = slice(s, s + 32)
            b = {
                rename.get(cam, cam): torch.from_numpy(rec["images"][sl, i]).float() / 255.0
                for i, cam in enumerate(DATASET_CAMERAS)
            }
            b[OBS_STATE] = torch.from_numpy(rec["state"][sl]).float()
            b["task"] = [str(rec["task_names"][i]) for i in rec["task"][sl]]
            try:
                resampled.append(predict(b))
            except Exception:  # noqa: BLE001 - e.g. an invalid FAST token sequence: row by row, failures as NaN
                rows = []
                for j in range(len(b["task"])):
                    one = {k: (v[j : j + 1] if torch.is_tensor(v) else [v[j]]) for k, v in b.items()}
                    try:
                        rows.append(predict(one))
                    except Exception:  # noqa: BLE001
                        rows.append(np.full((1, horizon, y_vla.shape[-1]), np.nan, dtype=np.float32))
                resampled.append(np.concatenate(rows))
        y2 = np.concatenate(resampled)
        ok = ~np.isnan(y2).any(axis=(1, 2))
        result["noise_floor"] = {
            f"vla_{a}": b for a, b in chunk_metrics(y_vla[ok], y2[ok], y2[ok], rec_task[ok]).items()
        }
        result["noise_floor"]["rows_failed"] = int((~ok).sum())
        del policy
        torch.cuda.empty_cache()

    # reference 2: nearest training chunk of the same task (does the VLA output something it has seen?)
    mem_rows = np.nonzero(frames.frame_index % 2 == 0)[0]
    if episodes is not None:
        mem_rows = mem_rows[np.isin(frames.episode_index[mem_rows], episodes)]
    idx = np.minimum(mem_rows[:, None] + np.arange(horizon), frames.episode_end[mem_rows][:, None])
    table = torch.from_numpy(frames.action[idx][..., MOTION]).cuda()
    mem_task = torch.from_numpy(frames.task_index[mem_rows]).cuda()
    nn_vla = []
    for i in range(len(y_vla)):
        cand = table[mem_task == int(rec_task[i])]
        nn_vla.append(float((cand - torch.from_numpy(y_vla[i, :, MOTION]).cuda()).pow(2).mean((1, 2)).min()))
    nn_demo = []
    mem_ep = torch.from_numpy(frames.episode_index[mem_rows]).cuda()
    for i in range(min(1000, len(demo_rows))):
        sel = (mem_task == int(demo_task[i])) & (mem_ep != int(frames.episode_index[demo_rows[i]]))
        d = (table[sel] - torch.from_numpy(y_demo[i, :, MOTION]).cuda()).pow(2).mean((1, 2))
        nn_demo.append(float(d.min()))
    result["nearest_training_chunk"] = {
        "vla_median": float(np.median(nn_vla)),
        "demo_median": float(np.median(nn_demo)),
    }
    result["per_record_nearest_training_chunk"] = nn_vla
    print(
        "noise floor", result.get("noise_floor"), "\nnearest training chunk", result["nearest_training_chunk"]
    )
    out.write_text(json.dumps(result, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
