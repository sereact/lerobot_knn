"""Render the self-contained HTML report from report/template.html and the gathered data JSON.

uv run python -m lerobot_policy_knn.experiments.report_data
uv run python -m lerobot_policy_knn.experiments.build_report --out=outputs/knn/report/libero_by_lookup.html
"""

import argparse
import json
from pathlib import Path

from .search import OUT

TEMPLATE = Path(__file__).resolve().parents[3] / "report" / "template.html"

COMMANDS = """# environment (uv workspace; Python 3.12)
uv sync --locked --python 3.12 --extra libero --group knn

# embed lerobot/libero once and save a policy (features cached in outputs/knn/feature_cache)
uv run lerobot-knn-fit --repo_id=lerobot/libero --output_dir=outputs/knn/policies/final \\
    --policy.image_encoder=resnet18 --policy.image_pooling=grid2   # + the tuned knobs, see config.json

# official evaluation, all four suites, 50 episodes per task
MUJOCO_GL=egl uv run lerobot-eval --policy.path=outputs/knn/policies/final --env.type=libero \\
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \\
    --env.observation_height=256 --env.observation_width=256 \\
    --eval.n_episodes=50 --eval.batch_size=10

# hyper-parameter search and its dashboard
uv run python -m lerobot_policy_knn.experiments.search --study=knn_v2 --space=v2 --n_trials=90
uv run optuna-dashboard sqlite:///outputs/knn/optuna.db

# maps and galleries
uv run python -m lerobot_policy_knn.experiments.global_map --config=<config.json>
uv run python -m lerobot_policy_knn.experiments.visualize --config=<config.json> --suite=libero_10 --task_id=0"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(OUT / "report" / "data.json"))
    parser.add_argument("--out", default=str(OUT / "report" / "libero_by_lookup.html"))
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text())
    data.setdefault("commands", COMMANDS)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/", payload)
    Path(args.out).write_text(html)
    print(args.out, round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
