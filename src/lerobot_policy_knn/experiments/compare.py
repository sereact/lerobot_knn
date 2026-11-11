"""Evaluate several KNN configs (JSON files of KNNConfig overrides) on the same LIBERO episodes.

    uv run python -m lerobot_policy_knn.experiments.compare a.json b.json --init_states=40:50 --tag=ab1

Results (per suite / per task) are appended to outputs/knn/compare.jsonl with the tag, config, and seed slice.

Keys of a config file that start with an underscore are options of this harness, not policy parameters:
``_demo_fraction`` keeps that fraction of every task's demonstrations in the memory, ``_train_split`` names a
``split.json`` whose ``train`` episodes are the memory.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..configuration_knn import KNNConfig
from .common import KNN_OUT, append_jsonl, parse_range, suite_scores
from .libero_pool import SUITES, LiberoPool, make_jobs
from .search import MemoryFactory, run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--init_states", default="40:50")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--tag", default="compare")
    args = parser.parse_args()
    jobs = make_jobs(args.suites.split(","), init_states=parse_range(args.init_states))
    factory = MemoryFactory()
    with LiberoPool(n_workers=args.workers) as pool:
        for path in args.configs:
            config = json.loads(Path(path).read_text())
            overrides = {k: v for k, v in config.items() if not k.startswith("_")}
            episodes = None
            if config.get("_demo_fraction") is not None:
                episodes = factory.demo_subset(config["_demo_fraction"])
            if config.get("_train_split") is not None:
                episodes = np.array(json.loads(Path(config["_train_split"]).read_text())["train"])
            summary, results = run_eval(
                factory, pool, KNNConfig(device="cuda", **overrides), jobs, episodes=episodes
            )
            row = {
                "tag": args.tag,
                "config_file": path,
                "config": config,
                "init_states": args.init_states,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                **summary,
            }
            append_jsonl(KNN_OUT / "compare.jsonl", [row])
            append_jsonl(
                KNN_OUT / "compare_episodes.jsonl",
                ({"tag": args.tag, "config_file": path, **vars(r)} for r in results),
            )
            print(
                f"{path}: {summary['success']:.4f} ({suite_scores(summary, SUITES)}) {summary['eval_s']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
