"""Gather every experiment of the LIBERO KNN study into one JSON document for the HTML report.

uv run python -m lerobot_policy_knn.experiments.report_data --out=outputs/knn/report/data.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import optuna

from .libero_pool import SUITES
from .search import OUT

STORAGE = f"sqlite:///{OUT / 'optuna.db'}"

# Published LIBERO success rates (%, spatial / object / goal / long), 500 episodes per suite in the
# original papers' own evaluation code. Sources: OpenVLA (Kim et al. 2024, Table 7) for Diffusion
# Policy (trained from scratch), Octo and OpenVLA fine-tuned; OpenVLA-OFT (Kim et al. 2025, Table 1)
# for pi0 and OpenVLA-OFT; LeRobot's LIBERO docs for pi0.5.
BASELINES = [
    {
        "name": "Diffusion Policy (from scratch)",
        "kind": "learned",
        "suites": [78.3, 92.5, 68.3, 50.5],
        "source": "OpenVLA paper",
    },
    {
        "name": "Octo (fine-tuned)",
        "kind": "learned",
        "suites": [78.9, 85.7, 84.6, 51.1],
        "source": "OpenVLA paper",
    },
    {
        "name": "OpenVLA 7B (fine-tuned)",
        "kind": "learned",
        "suites": [84.7, 88.4, 79.2, 53.7],
        "source": "OpenVLA paper",
    },
    {
        "name": "Multi-task DiT (diffusion, LeRobot)",
        "kind": "learned",
        "suites": [87.0, 98.2, 93.8, 83.2],
        "source": "LeRobot multi_task_dit docs",
    },
    {
        "name": "π0 (fine-tuned)",
        "kind": "learned",
        "suites": [96.8, 98.8, 95.8, 85.2],
        "source": "OpenVLA-OFT paper",
    },
    {
        "name": "OpenVLA-OFT",
        "kind": "learned",
        "suites": [97.6, 98.4, 97.9, 94.5],
        "source": "OpenVLA-OFT paper",
    },
    {
        "name": "π0.5 (LeRobot)",
        "kind": "learned",
        "suites": [97.0, 99.0, 98.0, 96.0],
        "source": "LeRobot LIBERO docs",
    },
]

KEY_PARAMS = [
    "image_encoder", "state_weight", "k", "temperature", "chunk_size", "temporal_ensemble", "ensemble_coeff",
    "n_action_steps", "memory_stride", "w_agentview", "w_wrist", "use_time", "one_per_episode", "gripper_mode",
    "use_consistency", "use_history", "norm_preserving", "use_action_mode", "action_mode_bandwidth",
    "distance_normalization", "task_conditioning", "text_weight",
]  # fmt: skip


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def study_rows(name: str) -> list[dict]:
    try:
        study = optuna.load_study(study_name=name, storage=STORAGE)
    except KeyError:
        return []
    rows = []
    for t in study.trials:
        params = t.params
        enc = params.get("image_encoder")
        row = {
            "n": t.number,
            "state": t.state.name,
            "value": _clean(t.value),
            "partial": _clean(t.intermediate_values[max(t.intermediate_values)])
            if t.intermediate_values
            else None,
            "minutes": round(t.duration.total_seconds() / 60, 2) if t.duration else None,
            "encoder": enc,
            "pooling": params.get(f"pooling_{enc}") if enc else None,
            "suites": [_clean(t.user_attrs.get(s)) for s in SUITES],
            "params": {k: _clean(params[k]) for k in KEY_PARAMS if k in params},
        }
        rows.append(row)
    return rows


def importances(name: str) -> dict[str, float]:
    try:
        study = optuna.load_study(study_name=name, storage=STORAGE)
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(done) < 10:
            return {}
        imp = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0)
        )
        return {k: round(v, 4) for k, v in imp.items()}
    except Exception:  # importance is best-effort
        return {}


# Human-readable description of the controlled experiments in compare.jsonl: tag -> (question, baseline
# config file, {config file: label}). Rows of other tags are shown with their file names.
EXPERIMENTS = {
    "codec_ab": (
        "Match the dataset's video compression on live frames?",
        "outputs/knn/best_v1.json",
        {
            "outputs/knn/best_v1.json": "best of study 1 (repeat runs)",
            "outputs/knn/best_v1_yuv420.json": "+ YUV 4:2:0 chroma subsampling of queries",
            "outputs/knn/best_v1_av1.json": "+ real SVT-AV1 round trip of queries",
        },
    ),
    "codec_ab_chunked": (
        "Same, open-loop chunked config (fewer queries)",
        "outputs/knn/trial1.json",
        {
            "outputs/knn/trial1.json": "16-step chunks, execute 8",
            "outputs/knn/trial1_av1.json": "+ real SVT-AV1 round trip of queries",
        },
    ),
    "mode_ab": (
        "Only average neighbours that agree? (relative kernel)",
        "outputs/knn/best_v1.json",
        {
            "outputs/knn/best_v1_mode0.1.json": "mode-seeking, bandwidth 0.1 × median spread",
            "outputs/knn/best_v1_mode0.3.json": "mode-seeking, bandwidth 0.3 × median spread",
            "outputs/knn/best_v1_mode1.json": "mode-seeking, bandwidth 1 × median spread",
            "outputs/knn/best_v1_mode0.3_ens0.5.json": "… and across ensembled chunks",
        },
    ),
    "absmode_ab": (
        "Only average neighbours that agree? (absolute kernel)",
        "outputs/knn/best_v1.json",
        {
            "outputs/knn/best_v1_absmode0.15.json": "mode-seeking, RMS action difference 0.15",
            "outputs/knn/best_v1_absmode0.3.json": "mode-seeking, RMS action difference 0.3",
        },
    ),
}


RETRY_LABELS = {
    "outputs/knn/retry/base.json": "final config, no retry",
    "outputs/knn/retry/r4_b20_tabu.json": "retry at 4× distance, rewind 20, avoid failed demos",
    "outputs/knn/retry/r4_b40_tabu.json": "retry at 4×, rewind 40, avoid failed demos",
    "outputs/knn/retry/r4_b20_notabu.json": "retry at 4×, rewind 20",
    "outputs/knn/retry/r5_b20_tabu.json": "retry at 5×, rewind 20, avoid failed demos",
    "outputs/knn/retry/r4_b40_tabu_max4.json": "retry at 4×, rewind 40, up to 4 retries, avoid failed demos",
    "outputs/knn/retry/r4_b40_notabu.json": "retry at 4×, rewind 40",
    "outputs/knn/retry/r3.5_b20_notabu.json": "retry at 3.5×, rewind 20",
    "outputs/knn/retry/r4_b20_notabu_max4.json": "retry at 4×, rewind 20, up to 4 retries",
    "outputs/knn/retry/r4_b20_notabu_stuck.json": "retry at 4× or when stuck 30 steps",
    "outputs/knn/retry/r4_b20_notabu_grip.json": "retry at 4× + gripper-finger distance term",
}
TTA_LABELS = {
    "outputs/knn/tta/base.json": "final config",
    "outputs/knn/tta/tta2_4px.json": "+ test-time augmentation, 2 shifted views (4 px)",
    "outputs/knn/tta/tta4_4px.json": "+ test-time augmentation, 4 shifted views (4 px)",
    "outputs/knn/tta/tta4_8px.json": "+ test-time augmentation, 4 shifted views (8 px)",
    "outputs/knn/tta/grip0.1.json": "+ gripper-finger distance term (0.1)",
    "outputs/knn/tta/grip0.3.json": "+ gripper-finger distance term (0.3)",
}
EXPERIMENTS.update({
    "retry_long": ("Backtrack and retry when lost? (LIBERO-Long, 200 episodes)", "outputs/knn/retry/base.json", RETRY_LABELS),
    "retry_long2": ("Backtrack and retry when lost? (LIBERO-Long, 400 episodes)", "outputs/knn/retry/base.json", RETRY_LABELS),
    "tta": ("Test-time augmentation / gripper term (held-out states 0–19)", "outputs/knn/tta/base.json", TTA_LABELS),
})  # fmt: skip


def _ablation_labels() -> dict[str, str]:
    path = OUT / "ablations" / "labels.json"
    return json.loads(path.read_text()) if path.exists() else {}


def annotate_compare(rows: list[dict]) -> list[dict]:
    """Attach group/label and the difference to the tag's baseline (mean over its repeats)."""
    EXPERIMENTS["ablation"] = (
        "What each component contributes (held-out states 20–29)",
        "outputs/knn/ablations/00_base.json",
        _ablation_labels(),
    )
    EXPERIMENTS["committee"] = (
        "Average two retrieval metrics? (held-out states 0–19)",
        "knn_v2#39",
        {"outputs/knn/v2_39.json+outputs/knn/v2_9.json": "committee of configs #39 and #9"},
    )
    baselines: dict[str, float] = {}
    for tag, (_q, base, _labels) in EXPERIMENTS.items():
        vals = [
            r["success"]
            for r in rows
            if r.get("tag") == tag and r.get("config_file") == base and "success" in r
        ]
        if not vals:  # the baseline may have been measured under another tag on the same episodes
            states = {r.get("init_states") for r in rows if r.get("tag") == tag} or {"40:50"}
            vals = [
                r["success"]
                for r in rows
                if r.get("config_file") == base and "success" in r and r.get("init_states") in states
            ]
        if vals:
            baselines[tag] = sum(vals) / len(vals)
    out = []
    for r in rows:
        tag = r.get("tag")
        if tag in EXPERIMENTS:
            question, base, labels = EXPERIMENTS[tag]
            r = {
                **r,
                "group": question,
                "label": labels.get(r["config_file"], r["config_file"].split("/")[-1]),
            }
            if tag in baselines and r["config_file"] != base:
                r["delta"] = r["success"] - baselines[tag]
        out.append(r)
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def official_evals() -> list[dict]:
    out = []
    for info in sorted((OUT / "eval_official").glob("*/eval_info.json")):
        d = json.loads(info.read_text())
        groups = {
            g: {"pc": v["pc_success"], "n": v["n_episodes"], "ci": v.get("pc_success_ci95")}
            for g, v in d["per_group"].items()
        }
        per_task = {f"{t['task_group']}/{t['task_id']}": t.get("pc_success") for t in d.get("per_task", [])}
        out.append({"name": info.parent.name, "overall": d["overall"]["pc_success"], "n": d["overall"]["n_episodes"],
                    "ci": d["overall"].get("pc_success_ci95"), "groups": groups, "per_task": per_task})  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT / "report" / "data.json"))
    parser.add_argument("--studies", default="knn_v1,knn_v2")
    args = parser.parse_args()
    studies = args.studies.split(",")
    viz = OUT / "viz"
    data = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "suites": list(SUITES),
        "baselines": BASELINES,
        "studies": {s: study_rows(s) for s in studies},
        "importance": {s: importances(s) for s in studies},
        "compare": annotate_compare(read_jsonl(OUT / "compare.jsonl")),
        "official": official_evals(),
        "domain_gap": json.loads((OUT / "domain_gap.json").read_text())
        if (OUT / "domain_gap.json").exists()
        else None,
        "codec_gap": json.loads((OUT / "codec_gap.json").read_text())
        if (OUT / "codec_gap.json").exists()
        else None,
        "disagreement": json.loads((OUT / "neighbour_disagreement.json").read_text())
        if (OUT / "neighbour_disagreement.json").exists()
        else None,
        "global_map": json.loads((viz / "global_map.json").read_text())
        if (viz / "global_map.json").exists()
        else None,
        "task_maps": {p.stem: json.loads(p.read_text()) for p in sorted(viz.glob("libero_*.json"))},
        "final": json.loads((OUT / "final.json").read_text()) if (OUT / "final.json").exists() else None,
        "memorization": json.loads((OUT / "memorization.json").read_text())
        if (OUT / "memorization.json").exists()
        else None,
        **(json.loads((OUT / "notes.json").read_text()) if (OUT / "notes.json").exists() else {}),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(args.out, round(Path(args.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
