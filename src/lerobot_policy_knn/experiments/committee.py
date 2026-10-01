"""A committee of KNN policies with different retrieval metrics: every step, average their actions.

    uv run python -m lerobot_policy_knn.experiments.committee outputs/knn/v2_39.json outputs/knn/v2_9.json \\
        --init_states=0:20 --workers=6
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ..configuration_knn import KNNConfig
from ..processor_knn import make_knn_pre_post_processors
from .libero_pool import SUITES, LiberoPool, make_jobs, summarize
from .search import OUT, MemoryFactory


class Committee:
    """Duck-types the KNN policy interface used by the evaluation pool."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights or [1.0 / len(members)] * len(members)

    def reset(self):
        for m in self.members:
            m.reset()

    def reset_envs(self, env_ids):
        for m in self.members:
            m.reset_envs(env_ids)

    def needs_observation(self, env_ids):
        needs = [m.needs_observation(env_ids) for m in self.members]
        return [any(col) for col in zip(*needs, strict=True)]

    @torch.no_grad()
    def select_action(self, batch, env_ids=None):
        actions = [m.select_action(batch, env_ids=env_ids).float() for m in self.members]
        out = sum(w * a for w, a in zip(self.weights, actions, strict=True))
        out[:, -1] = torch.where(out[:, -1] >= 0, 1.0, -1.0)
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--init_states", default="0:20")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="committee")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.init_states.split(":"))
    jobs = make_jobs(args.suites.split(","), init_states=range(lo, hi))
    factory = MemoryFactory()
    members = [
        factory.build(KNNConfig(device="cuda", **json.loads(Path(c).read_text()))) for c in args.configs
    ]
    pre, post = make_knn_pre_post_processors(members[0].config)
    policy = Committee(members)
    with LiberoPool(n_workers=args.workers) as pool:
        t0 = time.time()
        results = pool.evaluate(policy, pre, post, jobs, progress=False)
    summary = summarize(results)
    summary["eval_s"] = time.time() - t0
    row = {"tag": args.tag, "config_file": "+".join(args.configs), "init_states": args.init_states,
           **{k: v for k, v in summary.items() if "/" not in k}}  # fmt: skip
    with (OUT / "compare.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: round(v, 4) for k, v in summary.items() if "/" not in k}))


if __name__ == "__main__":
    main()
