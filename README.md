# lerobot_policy_knn

A non-parametric **k-nearest-neighbour action-chunk retrieval** policy for LeRobot, packaged as an
out-of-tree policy plugin (`--policy.type=knn`, auto-discovered by `lerobot-eval`/`lerobot-train` through the
`lerobot_policy_` prefix).

Nothing is trained. `lerobot-knn-fit` embeds every demonstration frame with frozen encoders and stores the
embeddings as the policy's memory. At test time the policy embeds the observation, retrieves the `k` closest
frames among the demonstrations of the prompted task, and executes a weighted average of the action chunks
that followed them.

## Results on LIBERO

Official `lerobot-eval`, 50 episodes per task on all four suites (2000 episodes), `lerobot/libero` data:

| policy                                                  | spatial | object | goal | long |  average |
| ------------------------------------------------------- | ------: | -----: | ---: | ---: | -------: |
| **kNN, one configuration** (`outputs/knn/v2_39.json`)   |    88.0 |   99.8 | 97.6 | 81.4 | **91.7** |
| kNN, per-suite configuration (spatial uses `v2_9.json`) |    91.8 |   99.8 | 97.6 | 81.4 |     92.7 |
| Multi-task DiT (diffusion transformer, LeRobot docs)    |    87.0 |   98.2 | 93.8 | 83.2 |     90.6 |
| Diffusion Policy from scratch (OpenVLA paper)           |    78.3 |   92.5 | 68.3 | 50.5 |     72.4 |
| OpenVLA 7B fine-tuned (OpenVLA paper)                   |    84.7 |   88.4 | 79.2 | 53.7 |     76.5 |
| π0.5 (LeRobot docs)                                     |    97.0 |   99.0 | 98.0 | 96.0 |     97.5 |

The single configuration: frozen ImageNet ResNet‑18 layer‑3 features on a 4×4 grid for both cameras, a
soft CLIP prompt-distance term, a small state weight, history matching, k = 13 neighbours (one per demo)
with a sharp softmax, mode-seeking aggregation (neighbours whose chunks differ by more than ~0.35 RMS are
down-weighted) and ACT-style temporal ensembling of 8-step chunks. Hyper-parameters were chosen by Optuna on
LIBERO initial states 40–49 and re-validated on held-out states before the official run.

What mattered (ablations on held-out initial states): averaging over neighbours (k = 1 costs 11 points),
the wrist camera (−7.5), CNN rather than DINOv2 features (−7.5), the prompt (goal suite drops to 8 % without
it). What did not help: trajectory stickiness, matching the dataset's video compression on live frames,
test-time augmentation, backtracking/retry when the observation looks unseen, a gripper-finger term.
With 10 % of the demonstrations (4–5 per task) retrieval still reaches 78 %.

### Is a diffusion policy "kNN with extra steps"?

We also trained LeRobot's language-conditioned Multi-task DiT on the same data (documented LIBERO recipe, reduced
budget: 60k steps × batch 32, `scripts/train_dit_libero.sh`; it reaches 68.8 % — the documented full-budget run
reports 90.6 %) and compared its action chunks, at states from rollouts on unseen initial states, with the data and
with retrieval (`experiments/memorization.py`, 789 chunks):

| motion distance, median per element               |        |
| ------------------------------------------------- | -----: |
| DiT chunk → nearest training chunk                | 0.0016 |
| a real held-out demo's chunk → nearest other demo | 0.0018 |
| DiT chunk → kNN answer for the same observation   | 0.0109 |
| kNN's own neighbours → kNN answer                 | 0.0095 |

Its outputs sit exactly as close to the demonstrations as a genuinely new demonstration does, and as close to the
kNN answer as kNN's own neighbours are: at each state it produces one of the demonstrated behaviours around that
state.

## How a retrieval works

```
distance(q, m) = sum over modalities   w_mod * ||q_mod - m_mod||^2 / scale_mod
```

| modality       | embedding                                                                                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| each camera    | frozen image backbone (`resnet18`, `resnet18_l3`, `dinov2_small/base`, `dinov3_small/base`, `siglip2_base`, `clip_b16`, raw `pixels`), pooled (`cls`, `mean`, `grid2`, `grid4`), unit norm |
| proprioception | end-effector position + rotation matrix (from LIBERO's axis-angle, avoiding its ±π wrap) + gripper, standardised                                                                           |
| time           | step index since the start of the episode                                                                                                                                                  |
| task           | CLIP text embedding of the prompt: either a hard filter (only the demos of the closest task) or a soft distance term                                                                       |

`scale_mod` normalises each modality (`distance_normalization`: median distance between random frames of the same
task, or median nearest-neighbour distance to other demos), so modality weights are comparable. On top of plain
retrieval the policy offers:

- `one_per_episode`: at most one neighbour per demonstration (neighbours are distinct demos, not consecutive frames)
- `temperature`: softmax weights over neighbours, relative to their mean distance (`k=1` → pure lookup)
- `chunk_size` / `n_action_steps` / `temporal_ensemble_coeff`: open-loop chunk execution or ACT-style ensembling
- `task_conditioning` (`filter` / `soft` / `none`) and `text_weight`
- `action_mode_bandwidth` / `ensemble_mode_bandwidth`: mode-seeking aggregation — neighbours (or overlapping
  ensembled chunks) whose actions disagree with the dominant one by more than ~this RMS are down-weighted instead of
  being averaged into an action none of them took
- `history_weight` / `history_gap`: also match the observation `history_gap` steps ago against the frame that
  preceded each candidate by as much in its demonstration (sequence matching)
- `consistency_weight`: trajectory stickiness — discount frames that continue the previously matched demo
- `retry_*`: rewind the last actions and retrieve again when the observation looks unseen (distance > N × the task's
  typical leave-one-demo-out distance) or the arm is stuck
- `gripper_weight`, `time_weight`, `state_weight`, per-camera `camera_weights`
- `query_tta` (shifted-view test-time augmentation) and `query_transform` (e.g. `av1`, to imitate the dataset's
  video compression on live frames)
- `norm_preserving`, `gripper_mode`, `memory_stride`

## Install

```bash
uv sync --extra all           # the whole study: LIBERO, Optuna, and every VLA family used as a key or baseline
uv sync --extra experiments   # only the kNN policy, LIBERO evaluation and the hyper-parameter search
```

LeRobot is a plain dependency, unmodified, pinned in `pyproject.toml` to the GitHub commit the study ran on. That
commit is newer than the 0.6.1 release, which lacks GR00T N1.7, MolmoAct2, Evo-1 and VLA-JEPA. The extras are `vla`
(SmolVLA, π0 / π0-FAST / π0.5, X-VLA), `vla_jepa`, `groot`, `molmoact2`, `evo1` and `multi_task_dit`. `uv sync`
makes the environment match the given selection exactly, so pass every extra you need on each sync.

- Python 3.12 is pinned in `.python-version`. draccus 0.11, used by every `lerobot-*` CLI, cannot parse `X | None`
  fields on 3.14.
- Run everything from the repository root: the experiments read and write under `outputs/`.

**LIBERO-plus** cannot share this environment: the fork replaces the `libero` package and pins robosuite 1.4.1 /
mujoco 3.7.0 (see `docker/Dockerfile.benchmark.libero_plus`). `outputs/libero_plus/setup.sh` builds a separate venv
with the same LeRobot commit and this package. Run its commands with
`PYTHONPATH=outputs/libero_plus/LIBERO-plus LIBERO_CONFIG_PATH=outputs/libero_plus/libero_config MUJOCO_GL=egl`.

## Fit and evaluate

```bash
# embed lerobot/libero and save a policy (features are cached under outputs/knn/feature_cache)
lerobot-knn-fit --repo_id=lerobot/libero --output_dir=outputs/knn/policies/my_knn \
    --policy.image_encoder=resnet18 --policy.image_pooling=grid2 --policy.k=8

# standard LeRobot evaluation (inference knobs can be overridden: --policy.k=4 ...)
MUJOCO_GL=egl lerobot-eval --policy.path=outputs/knn/policies/my_knn --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.observation_height=256 --env.observation_width=256 \
    --eval.n_episodes=50 --eval.batch_size=10
```

`--env.observation_height/width=256` renders at the resolution of the `lerobot/libero` frames the memory was
built from. With `lerobot-eval`, keep `--env.max_parallel_tasks=1`: the per-env action queues are keyed by
batch position, which concurrent tasks would share.

## Hyper-parameter search (Optuna)

```bash
uv run python -m lerobot_policy_knn.experiments.search --study=knn_v1 --n_trials=120
uv run optuna-dashboard sqlite:///outputs/knn/optuna.db           # browse / plot all trials
uv run python -m lerobot_policy_knn.experiments.analyze --study=knn_v1 --importance
```

Trials are scored by closed-loop LIBERO success with `experiments/libero_pool.py`, a persistent pool of
simulator processes that reproduces `lerobot-eval`'s LIBERO semantics (same env, init states, processors,
step limits) but schedules episodes of all tasks asynchronously. The search uses LIBERO initial states
40–49 as a validation slice; final numbers are reported with `lerobot-eval`.
