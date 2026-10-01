#!/usr/bin/env bash
# Diffusion baseline for the kNN study: LeRobot's language-conditioned Multi-task DiT on the same demonstrations.
# Architecture, augmentations and optimiser follow docs/source/multi_task_dit.mdx ("Libero Results"); the budget
# is reduced (batch 64 x 60k steps instead of 320 x 100k) so it trains overnight on one shared RTX 5090.
set -euo pipefail
cd "$(dirname "$0")/../../.."

TFS='{"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.75,1.25]}},"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"rotation":{"type":"RandomRotation","kwargs":{"degrees":[-5,5]}},"translation":{"type":"RandomAffine","kwargs":{"degrees":0,"translate":[0.1,0.1]}}}'

exec uv run lerobot-train \
  --dataset.repo_id=lerobot/libero \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs="$TFS" \
  --policy.type=multi_task_dit \
  --policy.push_to_hub=false \
  --policy.use_amp=true \
  --policy.horizon=48 \
  --policy.n_obs_steps=2 \
  --policy.use_rope=true \
  --policy.use_positional_encoding=false \
  --policy.hidden_dim=768 \
  --policy.num_layers=8 \
  --policy.num_heads=12 \
  --policy.dropout=0.1 \
  --policy.timestep_embed_dim=256 \
  --policy.objective=diffusion \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_weight_decay=0 \
  --policy.scheduler_warmup_steps=0 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape=[256,256] \
  --policy.image_crop_is_random=true \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --policy.vision_encoder_lr_multiplier=0.1 \
  --policy.device=cuda \
  --output_dir="${OUTPUT_DIR:-outputs/dit_libero}" \
  --job_name=dit_libero \
  --wandb.enable=false \
  --num_workers="${NUM_WORKERS:-6}" \
  --save_freq="${SAVE_FREQ:-20000}" \
  --log_freq=100 \
  --steps="${STEPS:-60000}" \
  --batch_size="${BATCH_SIZE:-64}" \
  "$@"
