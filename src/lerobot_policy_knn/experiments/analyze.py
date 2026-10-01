"""Summarise Optuna studies of the KNN policy: leaderboard, per-suite scores, parameter importances.

uv run python -m lerobot_policy_knn.experiments.analyze --study=knn_v1 [--top=10] [--csv=out.csv]
"""

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from .libero_pool import SUITES
from .search import OUT

STORAGE = f"sqlite:///{OUT / 'optuna.db'}"


def trials_frame(study_name: str, storage: str = STORAGE) -> pd.DataFrame:
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "state": t.state.name,
            "value": t.value,
            "duration_s": t.duration.total_seconds() if t.duration else None,
        }
        # pruned trials only have intermediate values
        if t.value is None and t.intermediate_values:
            row["partial_value"] = t.intermediate_values[max(t.intermediate_values)]
        row.update({f"p.{k}": v for k, v in t.params.items()})
        for suite in SUITES:
            if suite in t.user_attrs:
                row[suite] = t.user_attrs[suite]
        if "config" in t.user_attrs:
            row["config"] = t.user_attrs["config"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="knn_v1")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--importance", action="store_true")
    args = parser.parse_args()
    df = trials_frame(args.study)
    done = df[df.state == "COMPLETE"].sort_values("value", ascending=False)
    print(f"{args.study}: {len(df)} trials ({len(done)} complete, {(df.state == 'PRUNED').sum()} pruned)")
    cols = ["trial", "value", *[s for s in SUITES if s in done], *[c for c in done if c.startswith("p.")]]
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 20):
        print(done[cols].head(args.top).to_string(index=False))
    if args.importance and len(done) >= 10:
        study = optuna.load_study(study_name=args.study, storage=STORAGE)
        imp = optuna.importance.get_param_importances(study)
        print(json.dumps({k: round(v, 3) for k, v in imp.items()}, indent=1))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
