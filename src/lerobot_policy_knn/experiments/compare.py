"""Evaluate several KNN configs (JSON files of KNNConfig overrides) on the same LIBERO episodes.

    uv run python -m lerobot_policy_knn.experiments.compare a.json b.json --init_states=40:50 --tag=ab1

Results (per suite / per task) are appended to outputs/knn/compare.jsonl with the tag, config, and seed slice.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..configuration_knn import KNNConfig
from .libero_pool import SUITES, LiberoPool, make_jobs
from .search import OUT, MemoryFactory, run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--init_states", default="40:50")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--tag", default="compare")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.init_states.split(":"))
    jobs = make_jobs(args.suites.split(","), init_states=range(lo, hi))
    factory = MemoryFactory()
    with LiberoPool(n_workers=args.workers) as pool:
        for path in args.configs:
            overrides = json.loads(Path(path).read_text())
            # harness-level options (not policy parameters) start with an underscore
            fraction = overrides.pop("_demo_fraction", None)
            episodes = factory.demo_subset(fraction) if fraction is not None else None
            split = overrides.pop("_train_split", None)  # e.g. outputs/vla/split.json: memory = its "train"
            if split is not None:
                episodes = np.array(json.loads(Path(split).read_text())["train"])
            policy_cfg = KNNConfig(device="cuda", **overrides)
            summary, results = run_eval(factory, pool, policy_cfg, jobs, episodes=episodes)
            if fraction is not None:
                overrides["_demo_fraction"] = fraction
            if split is not None:
                overrides["_train_split"] = split
            row = {
                "tag": args.tag,
                "config_file": path,
                "config": overrides,
                "init_states": args.init_states,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                **summary,
            }
            with (OUT / "compare.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            with (OUT / "compare_episodes.jsonl").open("a") as f:
                for r in results:
                    f.write(json.dumps({"tag": args.tag, "config_file": path, **r.__dict__}) + "\n")
            suites = " ".join(f"{s.split('_')[1]}={summary[s]:.3f}" for s in SUITES if s in summary)
            print(f"{path}: {summary['success']:.4f} ({suites}) {summary['eval_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
