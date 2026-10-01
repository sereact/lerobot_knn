#!/usr/bin/env bash
# Reproduce lerobot/smolvla_libero (SmolVLA base -> LIBERO, 25k steps, batch 32, full fine-tune) with 10% of
# every task's demonstrations held out (outputs/vla/split.json) and a checkpoint every SAVE_FREQ steps, so the
# VLM's quality as a retrieval embedder can be tracked over fine-tuning.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CFG=outputs/vla/smolvla_libero_train_config.json
# the published config, minus fields the current TrainPipelineConfig no longer has
uv run python - "$CFG" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
cfg = json.load(open(hf_hub_download("lerobot/smolvla_libero", "train_config.json")))
for key in ("eval_freq", "delta_action_stats_num_samples"):
    cfg.pop(key, None)
json.dump(cfg, open(sys.argv[1], "w"), indent=1)
PY
EPISODES=$(uv run python -c "import json; print(json.load(open('outputs/vla/split.json'))['train'])" | tr -d ' ')
OUTPUT_DIR=${OUTPUT_DIR:-outputs/vla/smolvla_own}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run lerobot-train \
  --config_path="$CFG" \
  --dataset.episodes="$EPISODES" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_own \
  --steps="${STEPS:-25000}" \
  --save_freq="${SAVE_FREQ:-1000}" \
  --log_freq=100 \
  --num_workers="${NUM_WORKERS:-6}" \
  --batch_size="${BATCH_SIZE:-32}" \
  --policy.push_to_hub=false \
  --policy.repo_id=null \
  --wandb.enable=false \
  --resume=false "$@"
