"""Select final configurations without optimiser's-curse bias.

The search ranked trials on LIBERO initial states 40-49. Here the best trials overall and per suite are
re-evaluated on held-out initial states (default 0-19) the search never saw, and the winners (one global
configuration, and the best configuration per suite) are written to ``outputs/knn/final_candidates.json``.

    uv run python -m lerobot_policy_knn.experiments.finalize --studies=knn_v1,knn_v2 --top=5 --top_per_suite=2
"""

import argparse
import json
from pathlib import Path

import optuna

from ..configuration_knn import KNNConfig
from .libero_pool import SUITES, LiberoPool, make_jobs
from .search import OUT, MemoryFactory, run_eval

STORAGE = f"sqlite:///{OUT / 'optuna.db'}"


def candidates(studies: list[str], top: int, top_per_suite: int, suites: list[str] = SUITES) -> list[dict]:
    done = []
    for name in studies:
        study = optuna.load_study(study_name=name, storage=STORAGE)
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and "config" in t.user_attrs:
                done.append(
                    {
                        "id": f"{name}#{t.number}",
                        "value": t.value,
                        "attrs": t.user_attrs,
                        "config": json.loads(t.user_attrs["config"]),
                    }
                )
    chosen: dict[str, dict] = {}
    for c in sorted(done, key=lambda c: -c["value"])[:top]:
        chosen[c["id"]] = c
    for suite in suites:
        for c in sorted(done, key=lambda c: -c["attrs"].get(suite, 0))[:top_per_suite]:
            chosen[c["id"]] = c
    return list(chosen.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", default="knn_v1,knn_v2")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--top_per_suite", type=int, default=2)
    parser.add_argument("--init_states", default="0:20")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--out", default=str(OUT / "final_candidates.json"))
    args = parser.parse_args()
    suites = args.suites.split(",")
    cands = candidates(args.studies.split(","), args.top, args.top_per_suite, suites)
    print(f"{len(cands)} candidates:", [c["id"] for c in cands], flush=True)
    lo, hi = (int(x) for x in args.init_states.split(":"))
    jobs = make_jobs(suites, init_states=range(lo, hi))
    factory = MemoryFactory()
    results = []
    with LiberoPool(n_workers=args.workers) as pool:
        for c in cands:
            summary, _ = run_eval(factory, pool, KNNConfig(device="cuda", **c["config"]), jobs)
            row = {"tag": "heldout", "config_file": c["id"], "config": c["config"], "init_states": args.init_states,
                   "search_value": c["value"], **summary}  # fmt: skip
            with (OUT / "compare.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            results.append(row)
            print(f"{c['id']}: held-out {summary['success']:.4f} (search {c['value']:.4f}) "
                  + " ".join(f"{s.split('_')[1]}={summary[s]:.3f}" for s in suites), flush=True)  # fmt: skip
    best = max(results, key=lambda r: r["success"])
    per_suite = {s: max(results, key=lambda r: r[s]) for s in suites}
    out = {
        "init_states": args.init_states,
        "global": {"id": best["config_file"], "config": best["config"], "heldout": best["success"]},
        "per_suite": {
            s: {"id": r["config_file"], "config": r["config"], "heldout": r[s]} for s, r in per_suite.items()
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {"global": out["global"]["id"], **{s: v["id"] for s, v in out["per_suite"].items()}}, indent=1
        )
    )


if __name__ == "__main__":
    main()
