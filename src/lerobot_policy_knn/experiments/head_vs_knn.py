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
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from ..configuration_knn import KNNConfig
from ..features import FrameIndex
from ..modeling_knn import KNNPolicy, camera_key_name
from ..vla import DATASET_CAMERAS, VLA_CAMERA, cache_tag, load_vla, resolve_spec
from .common import GRIPPER, MOTION, PLUS_OUT, VLA_OUT, action_chunks, record_batch
from .search import MemoryFactory

IMAGE_SPACES = {
    "dinov2": {"image_encoder": "dinov2_small", "image_pooling": "cls"},
    "resnet18_l3": {"image_encoder": "resnet18_l3", "image_pooling": "grid4"},
    "resnet18_l3_mean": {"image_encoder": "resnet18_l3", "image_pooling": "mean"},
}


def base_task(name: str, tasks: Iterable[str]) -> str:
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
    """The plain lookup (no state, time, history or mode seeking) on a key space named
    ``<encoder space>[|<key_norm>]``, e.g. ``resnet18_l3``, ``vla:pi05_libero/grid2@L12|zscore``."""
    space, _, key_norm = space.partition("|")
    if space.startswith("vla:"):
        encoder, pooling = space.rsplit("/", 1)
        per_camera = pooling == "img" or pooling.startswith("grid2")
        cameras = list(DATASET_CAMERAS) if per_camera else [VLA_CAMERA]
        embedding = {"image_encoder": encoder, "image_pooling": pooling}
    else:
        cameras = list(DATASET_CAMERAS)
        embedding = dict(IMAGE_SPACES[space])
    return KNNConfig(
        device="cuda",
        **embedding,
        key_norm=key_norm or "none",
        camera_weights=dict.fromkeys(cameras, 1.0),
        state_weight=0.0,
        time_weight=0.0,
        task_conditioning="filter",
        k=k,
        temperature=0.1,
        one_per_episode=True,
        history_weight=0.0,
        chunk_size=horizon,
        n_action_steps=horizon,
        memory_stride=2,
    )


def r2(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """1 - SSE / SST with SST measured around each group's (task's) mean chunk."""
    sst = 0.0
    for g in np.unique(groups):
        m = groups == g
        sst += float(((y[m] - y[m].mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / max(sst, 1e-12)


def chunk_metrics(y: np.ndarray, p: np.ndarray, top1: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """How well chunks ``p`` (and the single nearest neighbour's, ``top1``) predict the chunks ``y``."""
    return {
        "r2": r2(y[..., MOTION], p[..., MOTION], groups),
        "mse": float(((y[..., MOTION] - p[..., MOTION]) ** 2).mean()),
        "top1_mse": float(((y[..., MOTION] - top1[..., MOTION]) ** 2).mean()),
        "top1_r2": r2(y[..., MOTION], top1[..., MOTION], groups),
        "gripper": float((np.sign(y[..., GRIPPER]) == np.sign(p[..., GRIPPER])).mean()),
    }


@torch.no_grad()
def retrieve_chunks(
    policy: KNNPolicy, query: dict, t: torch.Tensor, task: torch.Tensor, horizon: int, exclude=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kNN answer, nearest neighbour's chunk, distance to the nearest neighbour) per query."""
    rows, w, d = policy.retrieve(query, t, task, exclude_episode=exclude)
    pred = policy.aggregate(rows, w)[:, :horizon]
    top1 = policy.gather_chunks(rows[:, :1], horizon)[:, 0]
    return pred.float().cpu().numpy(), top1.float().cpu().numpy(), d[:, 0].float().cpu().numpy()


@torch.no_grad()
def embed_records(policy: KNNPolicy, rec: dict, batch: int = 48) -> dict[str, torch.Tensor]:
    """Keys of recorded observations; the batch halves whenever an embedder runs out of GPU memory (e.g. Evo-1's eager
    attention at 448 px)."""
    out: dict[str, list[torch.Tensor]] = {}
    n = len(rec["state"])
    start = 0
    while start < n:
        rows = slice(start, min(n, start + batch))
        try:
            keys = policy.encode_observation(record_batch(rec, rows, device="cuda"))
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            continue
        for name, value in keys.items():
            out.setdefault(name, []).append(value)
        start = rows.stop
    return {name: torch.cat(values) for name, values in out.items()}


@torch.no_grad()
def demo_queries(policy: KNNPolicy, factory: MemoryFactory, rows: np.ndarray) -> dict[str, torch.Tensor]:
    """Keys of demonstration frames ``rows``, from the feature cache."""
    cfg = policy.config
    queries = {}
    for cam in cfg.active_cameras:
        raw = factory.load_raw(cfg.image_encoder, cam, cfg.image_pooling, cfg.image_resolution)
        name = camera_key_name(cam)
        queries[name] = policy.key_features(name, torch.as_tensor(np.asarray(raw[rows])).cuda())
    return queries


def record_tasks(rec: dict, frames: FrameIndex) -> np.ndarray:
    """Memory task index of every recorded step."""
    task_of = {name: i for i, name in enumerate(frames.task_names)}
    if (
        "category" in rec
    ):  # LIBERO-plus rollouts: the base task from each variant's name (prompts may be reworded)
        sample = json.loads((PLUS_OUT / "sample.json").read_text())
        name_of = {(s["suite"], s["task_id"]): s["name"] for s in sample}
        names = [name_of[(str(su), int(ti))] for su, ti in zip(rec["suite"], rec["task_id"], strict=True)]
    else:
        names = [str(n) for n in rec["task_names"][rec["task"]]]
    return np.array([task_of[base_task(n, task_of)] for n in names])


def noise_floor(vla: str, rec: dict, y_vla: np.ndarray, rec_task: np.ndarray, batch: int = 32) -> dict:
    """Reference 1: the VLA's own sampling noise, a second sample at every recorded observation."""
    policy, pre, post = load_vla(vla)
    rename = resolve_spec(vla).camera_rename
    horizon = y_vla.shape[1]

    def predict(b: dict) -> np.ndarray:
        policy.reset()
        return post(policy.predict_action_chunk(pre(b))).float().cpu().numpy()[:, :horizon]

    resampled = []
    for s in range(0, len(y_vla), batch):
        b = {rename.get(k, k): v for k, v in record_batch(rec, slice(s, s + batch)).items()}
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
    out = {f"vla_{a}": b for a, b in chunk_metrics(y_vla[ok], y2[ok], y2[ok], rec_task[ok]).items()}
    out["rows_failed"] = int((~ok).sum())
    return out


def nearest_training_chunks(
    frames: FrameIndex,
    episodes: np.ndarray | None,
    y_vla: np.ndarray,
    rec_task: np.ndarray,
    y_demo: np.ndarray,
    demo_rows: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Reference 2: does the VLA output something it has seen? The motion MSE from every VLA chunk to the nearest
    training chunk of its task, and from (up to 1,000) held-out demo chunks to the nearest chunk of another demo."""
    horizon = y_vla.shape[1]
    memory = np.nonzero(frames.frame_index % 2 == 0)[0]
    if episodes is not None:
        memory = memory[np.isin(frames.episode_index[memory], episodes)]
    table = torch.from_numpy(action_chunks(frames, memory, horizon)[..., MOTION]).cuda()
    task = torch.from_numpy(frames.task_index[memory]).cuda()
    episode = torch.from_numpy(frames.episode_index[memory]).cuda()

    def nearest(chunk: np.ndarray, candidates: torch.Tensor) -> float:
        return float(
            (table[candidates] - torch.from_numpy(chunk[:, MOTION]).cuda()).pow(2).mean((1, 2)).min()
        )

    vla = [nearest(y_vla[i], task == int(rec_task[i])) for i in range(len(y_vla))]
    demo = []
    for i in range(min(1000, len(demo_rows))):
        row = demo_rows[i]
        other_demos = (task == int(frames.task_index[row])) & (episode != int(frames.episode_index[row]))
        demo.append(nearest(y_demo[i], other_demos))
    return vla, demo


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
    ks = [int(k) for k in args.ks.split(",")]
    rec = dict(np.load(args.rollouts))
    # a VLA re-planning every n < horizon steps (VLA-JEPA: 7) recorded n-step chunks: compare that many actions
    horizon = min(args.horizon, int(rec["chunk"].shape[1]))
    factory = MemoryFactory()
    frames = factory.frames
    rec_task = record_tasks(rec, frames)
    y_vla = rec["chunk"][:, :horizon]

    # held-out demonstration frames: what the same lookup predicts of the demonstrator
    episodes = None
    demo_pool = np.nonzero(frames.frame_index % 10 == 4)[0]
    if args.train_split:
        split = json.loads(Path(args.train_split).read_text())
        episodes = np.array(split["train"])
        demo_pool = demo_pool[np.isin(frames.episode_index[demo_pool], split["heldout"])]
    rng = np.random.default_rng(0)
    demo_rows = np.sort(rng.choice(demo_pool, min(args.demo_queries, len(demo_pool)), replace=False))
    y_demo = action_chunks(frames, demo_rows, horizon)
    demo_task = frames.task_index[demo_rows]

    result: dict = {
        "rollouts": args.rollouts,
        "vla": args.vla,
        "horizon": horizon,
        "n_chunks": len(y_vla),
        "vla_success": float(np.mean(rec["success"])),
        "spaces": {},
    }
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
        # with the whole dataset as memory, a demo query must not retrieve its own demonstration
        exclude = torch.from_numpy(frames.episode_index[demo_rows]).cuda() if episodes is None else None
        for k in ks:
            policy.config.k = k
            p, top1, d1 = retrieve_chunks(policy, q_vla, t_vla, task_vla, horizon)
            p_demo, top1_demo, d1_demo = retrieve_chunks(policy, q_demo, t_demo, task_demo, horizon, exclude)
            m = {f"vla_{a}": b for a, b in chunk_metrics(y_vla, p, top1, rec_task).items()}
            m.update({f"demo_{a}": b for a, b in chunk_metrics(y_demo, p_demo, top1_demo, demo_task).items()})
            result["spaces"][space][k] = m
            if k == 8:
                per_record[space] = (((y_vla[..., MOTION] - p[..., MOTION]) ** 2).mean((1, 2))).tolist()
                novelty[space] = (d1 / max(float(np.median(d1_demo)), 1e-9)).tolist()
            print(space, k, {a: round(b, 4) for a, b in m.items()}, flush=True)
        del policy, q_vla, q_demo
        factory.release_encoders()
    result["per_record_mse_k8"] = per_record
    result["per_record_novelty"] = novelty
    result["per_record"] = {
        k: [str(x) if k == "category" else x.item() for x in rec[k]]
        for k in ("success", "episode", "category")
        if k in rec
    }

    out = Path(args.out or VLA_OUT / f"head_vs_knn_{cache_tag(args.vla)}.json")
    if not args.no_references:
        if not args.no_noise_floor:
            result["noise_floor"] = noise_floor(args.vla, rec, y_vla, rec_task)
        nn_vla, nn_demo = nearest_training_chunks(frames, episodes, y_vla, rec_task, y_demo, demo_rows)
        result["nearest_training_chunk"] = {
            "vla_median": float(np.median(nn_vla)),
            "demo_median": float(np.median(nn_demo)),
        }
        result["per_record_nearest_training_chunk"] = nn_vla
        print(
            "noise floor",
            result.get("noise_floor"),
            "\nnearest training chunk",
            result["nearest_training_chunk"],
        )
    out.write_text(json.dumps(result, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
