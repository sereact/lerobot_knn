"""Write one-change-at-a-time variants of a base configuration (for experiments.compare).

    uv run python -m lerobot_policy_knn.experiments.ablations --base=outputs/knn/final_global.json \\
        --out_dir=outputs/knn/ablations
    uv run python -m lerobot_policy_knn.experiments.compare outputs/knn/ablations/*.json \\
        --init_states=20:30 --tag=ablation
"""

import argparse
import json
from pathlib import Path

AGENT_VIEW, WRIST = "observation.images.image", "observation.images.image2"


def variants(base: dict) -> dict[str, tuple[str, dict]]:
    """name -> (description, overrides)."""
    weights = base.get("camera_weights", {AGENT_VIEW: 1.0, WRIST: 1.0})
    agent_only = {AGENT_VIEW: weights.get(AGENT_VIEW, 1.0), WRIST: 0.0}
    wrist_only = {AGENT_VIEW: 0.0, WRIST: weights.get(WRIST, 1.0)}
    open_loop = {"temporal_ensemble_coeff": None, "n_action_steps": base.get("chunk_size", 8)}
    out: dict[str, tuple[str, dict]] = {
        "00_base": ("final configuration", {}),
        "no_language": ("no task conditioning (prompt ignored)", {"task_conditioning": "none"}),
        "no_state": ("images only (state weight 0)", {"state_weight": 0.0}),
        "no_wrist": ("agent-view camera only", {"camera_weights": agent_only}),
        "no_agentview": ("wrist camera only", {"camera_weights": wrist_only}),
        "k1": ("k = 1: replay the single nearest demo, no averaging", {"k": 1}),
        "no_ensemble": ("open-loop chunks (execute 8 of 8), no temporal ensembling", open_loop),
        "encoder_dinov2": (
            "encoder: DINOv2-S CLS",
            {"image_encoder": "dinov2_small", "image_pooling": "cls"},
        ),
        "encoder_pixels": ("encoder: raw 16×16 pixels", {"image_encoder": "pixels", "image_pooling": "p16"}),
        "demos_50": ("50% of the demonstrations", {"_demo_fraction": 0.5}),
        "demos_25": ("25% of the demonstrations", {"_demo_fraction": 0.25}),
        "demos_10": ("10% of the demonstrations (4–5 per task)", {"_demo_fraction": 0.1}),
    }
    if base.get("task_conditioning") == "soft":
        out["hard_filter"] = (
            "hard task filter instead of soft prompt distance",
            {"task_conditioning": "filter"},
        )
    else:
        out["soft_text"] = (
            "soft prompt distance instead of hard filter",
            {"task_conditioning": "soft", "text_weight": 0.5},
        )
    if base.get("action_mode_bandwidth", 0) > 0:
        out["no_mode"] = (
            "plain weighted mean (no mode-seeking)",
            {"action_mode_bandwidth": 0.0, "ensemble_mode_bandwidth": 0.0},
        )
    if base.get("history_weight", 0) > 0:
        out["no_history"] = ("no history matching", {"history_weight": 0.0})
    if base.get("time_weight", 0) > 0:
        out["no_time"] = ("no time feature", {"time_weight": 0.0})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out_dir", default="outputs/knn/ablations")
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {}
    for name, (description, overrides) in sorted(variants(base).items()):
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({**base, **overrides}, indent=1))
        labels[str(path)] = description
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print("\n".join(sorted(labels)))


if __name__ == "__main__":
    main()
