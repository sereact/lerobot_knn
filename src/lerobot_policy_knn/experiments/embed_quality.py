"""How good is an embedding space as a lookup key? Offline action prediction on held-out demonstrations.

For query frames of the held-out episodes of ``outputs/vla/split.json`` (or leave-one-episode-out without a
split), retrieve the ``k`` nearest frames of the same task from the memory episodes and compare the averaged
chunk with the demonstrated one (R² of the motion dimensions around each task's mean chunk, gripper agreement).
Same plain kNN as ``head_vs_knn``.

    uv run python -m lerobot_policy_knn.experiments.embed_quality \\
        --spaces=resnet18_l3,vla:smolvla_base/img --train_split=outputs/vla/split.json --tag=ladder
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .head_vs_knn import chunk_metrics, demo_queries, retrieve_chunks, space_config
from .search import MemoryFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", required=True)
    parser.add_argument("--ks", default="1,8,32")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--queries", type=int, default=6000)
    parser.add_argument("--tag", default="embed_quality")
    parser.add_argument("--out", default="outputs/vla/embed_quality.jsonl")
    args = parser.parse_args()
    horizon = args.horizon
    factory = MemoryFactory()
    frames = factory.frames
    pool = np.nonzero(frames.frame_index % 10 == 4)[0]
    episodes = None
    if args.train_split:
        split = json.loads(Path(args.train_split).read_text())
        episodes = np.array(split["train"])
        pool = pool[np.isin(frames.episode_index[pool], split["heldout"])]
    rows = np.sort(np.random.default_rng(0).choice(pool, min(args.queries, len(pool)), replace=False))
    y = frames.action[np.minimum(rows[:, None] + np.arange(horizon), frames.episode_end[rows][:, None])]
    task = frames.task_index[rows]
    t = torch.from_numpy(frames.frame_index[rows]).cuda()
    task_t = torch.from_numpy(task).cuda()
    excl = torch.from_numpy(frames.episode_index[rows]).cuda() if episodes is None else None
    ks = [int(k) for k in args.ks.split(",")]
    for space in args.spaces.split(","):
        policy = factory.build(space_config(space, max(ks), horizon), episodes=episodes)
        q = demo_queries(policy, factory, rows)
        for k in ks:
            policy.config.k = k
            p, top1, _ = retrieve_chunks(policy, q, t, task_t, horizon, exclude=excl)
            m = chunk_metrics(y, p, top1, task)
            row = {"tag": args.tag, "space": space, "k": k, "n_queries": len(rows),
                   "split": args.train_split, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **m}  # fmt: skip
            with Path(args.out).open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(space, k, {a: round(b, 4) for a, b in m.items()}, flush=True)
        del policy, q
        factory._encoders.clear()
        factory.forget_raw()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
