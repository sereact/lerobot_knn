# The k-NN policy: retrieval and configuration

```
distance(q, m) = sum over modalities   w_mod * ||q_mod - m_mod||^2 / scale_mod
```

| Modality | Embedding |
| --- | --- |
| each camera | a frozen image backbone (`resnet18`, `resnet18_l3`, `dinov2_small/base`, `dinov3_small/base`, `siglip2_base`, `clip_b16`, raw `pixels`) pooled (`cls`, `mean`, `grid2`, `grid4`) and unit-normalised; or a VLA's VLM (`vla:<name>`, below) |
| proprioception | end-effector position + rotation matrix (from LIBERO's axis-angle, avoiding its ±π wrap) + gripper, standardised |
| time | step index since the start of the episode |
| task | CLIP text embedding of the prompt: a hard filter (only the demos of the closest task) or a soft distance term |

`scale_mod` makes the modality weights comparable (`distance_normalization`: the median distance between random frames
of the same task, or the median nearest-neighbour distance to other demos). On top of plain retrieval:

- `k`, `one_per_episode` (at most one neighbour per demonstration) and `temperature` (softmax weights relative to the
  neighbours' mean distance; `k=1` is a pure replay)
- `chunk_size` / `n_action_steps` / `temporal_ensemble_coeff`: open-loop chunks or ACT-style temporal ensembling
- `task_conditioning` (`filter` / `soft` / `none`) and `text_weight`
- `action_mode_bandwidth` / `ensemble_mode_bandwidth`: mode seeking. Neighbours (or overlapping ensembled chunks)
  whose actions disagree with the dominant one by more than about this RMS are down-weighted instead of averaged into
  an action none of them took
- `history_weight` / `history_gap`: also match the observation `history_gap` steps ago against the frame that
  preceded each candidate by as much in its demonstration
- `retry_*`: rewind the last actions and retrieve again when the observation looks unseen or the arm is stuck
- `state_weight`, `time_weight`, per-camera `camera_weights`, `memory_stride`, `gripper_mode`
- `key_norm`: `zscore` standardises every key dimension with memory statistics (needed for VLM keys, whose residual
  streams are dominated by a few massive activations), `zscore_pca<n>` whitens, `file:<path>` applies a learnt key
- `relative_action_dims`: re-anchor the neighbours' absolute end-effector targets on the query's state
- searched by Optuna but unused in the final configuration: `consistency_weight` (trajectory stickiness),
  `norm_preserving`, `gripper_weight`, `query_tta` (shifted-view test-time augmentation)

**VLM keys.** `--policy.image_encoder=vla:<name>` runs the cameras, the prompt and (for π0.5) the state through a VLA's
VLM prefix (`vla.py`, `VLAEmbedder`) and keys on the hidden states over each camera's image tokens.
`image_pooling=grid2@L12` taps the residual stream after 12 LM layers on a 2×2 grid; use `key_norm=zscore`. The
registry (`vla.VLA_SPECS`) has each family before and after LIBERO fine-tuning: `pi05_base` / `pi05_libero`,
`pi0_base` / `pi0_libero`, `pi0fast_base` / `pi0fast_libero`, `smolvla_base` / `smolvla_libero`, `xvla_base` /
`xvla_libero`, `vla_jepa_pretrain` / `vla_jepa_libero`, `evo1_base` / `evo1_libero`, `molmoact2_base` /
`molmoact2_libero`, and `groot_base` (GR00T's frozen VLM). `<family>@<checkpoint dir or Hub repo>` works for other
checkpoints. The keys used in the study are `outputs/vla/configs/plain_*.json` (listed in
`outputs/vla/extra_keys.json`). `lerobot/pi05_base` and `lerobot/pi0_base` are float32 on the Hub;
`experiments/convert_bf16.py` writes the bfloat16 copies under `outputs/vla/` that the registry loads.

## Notes on fitting and evaluating

- The configuration of the full method is `outputs/knn/v2_39.json` (KNNConfig fields; pass them as
  `--policy.<field>=<value>`). Inference options can be overridden at evaluation time, e.g. `--policy.k=4`.
- `lerobot-knn-fit` caches the features under `outputs/knn/feature_cache`.
- `--env.observation_height/width=256` renders at the resolution of the `lerobot/libero` frames the memory was built
  from.
- With `lerobot-eval`, keep `--env.max_parallel_tasks=1`: the per-env action queues are keyed by batch position, which
  concurrent tasks would share.
- Python 3.12 is pinned in `.python-version`. draccus 0.11, used by every `lerobot-*` CLI, cannot parse `X | None`
  fields on 3.14.
- The extras are `experiments`, `vla` (SmolVLA, π0 / π0-FAST / π0.5, X-VLA), `vla_jepa`, `groot`, `molmoact2`, `evo1`
  and `all`. `uv sync` makes the environment match the given selection exactly, so pass every extra you need on each
  sync.
- Set `HF_HUB_DISABLE_XET=1` if Hub downloads stall.
