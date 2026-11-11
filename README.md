# lerobot_policy_knn

A training-free **k-nearest-neighbour action-chunk retrieval** policy for LeRobot, packaged as an out-of-tree policy
plugin (`--policy.type=knn`, auto-discovered by `lerobot-eval` / `lerobot-train` through the `lerobot_policy_` prefix).

Nothing is trained. `lerobot-knn-fit` embeds every demonstration frame with frozen encoders and stores the embeddings
as the policy's memory. At test time the policy embeds the observation, retrieves the `k` closest demonstration
frames, and executes a weighted average of the action chunks that followed them. Every executed action is a convex
combination of demonstrated actions.

**We do not propose kNN for robot control**: a lookup cannot extrapolate beyond its memory. We use it as a probe,
"How Much of Your VLA is kNN", that answers three questions without training anything:

1. **How hard is the task?** What a lookup over generic image features already solves is retrieval, not
   generalisation.
2. **How good is a VLA's backbone, and what did fine-tuning do to it?** Key the same lookup on the VLA's own
   vision-language model (VLM), before and after fine-tuning, in closed loop, or offline with a neighbour metric.
3. **How good is its action head?** Its margin over a lookup on its own backbone.

Because the probe is a LeRobot policy, it is fitted from any LeRobot dataset with one command and evaluated with the
standard `lerobot-eval`: any benchmark LeRobot runs can run the test, and any VLA in LeRobot can serve as its key.
The experiment notes are in [`docs/experiments/`](docs/experiments/README.md), related work in
[`docs/literature_review.md`](docs/literature_review.md).

## Results

### LIBERO and LIBERO-plus

LIBERO: official protocol, initial states 0–49 of all 40 tasks (2,000 episodes); the two full-method kNN rows with
`lerobot-eval`, every other row in our evaluation pool (`experiments/libero_pool.py`) on the same initial states.
LIBERO-plus: a fixed sample of 420 of its perturbed variants (60 per perturbation), the same for every policy, never
trained on. Rows are ordered by LIBERO-plus. The VLAs are public LeRobot Hub LIBERO checkpoints, evaluated as
released; none was trained here.

| Policy | LIBERO | Spatial | Object | Goal | Long | LIBERO-plus | drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| π0.5 | **97.7** | 97.0 | 99.8 | 97.8 | 96.2 | **85.5** | −12.2 |
| MolmoAct2 | 97.5 | 97.4 | 99.4 | 97.6 | 95.6 | 82.6 | −14.9 |
| **kNN, full method** (`outputs/knn/v2_39.json`) | 91.7 | 88.0 | 99.8 | 97.6 | 81.4 | 82.4 | **−9.3** |
| kNN, full method + learnt key | 93.0 | 90.6 | 99.6 | 97.0 | 84.8 | 81.7 | −11.3 |
| GR00T N1.7 | 94.8 | 95.8 | 98.2 | 94.6 | 90.8 | 80.2 | −14.6 |
| kNN, plain (ImageNet ResNet-18) | 88.7 | 90.4 | 95.0 | 95.2 | 74.2 | 75.5 | −13.2 |
| VLA-JEPA | 97.2 | 95.4 | 100.0 | 97.4 | 95.8 | 74.0 | −23.1 |
| Evo-1 | 95.4 | 92.0 | 98.6 | 97.4 | 93.6 | 71.4 | −24.0 |
| X-VLA | 94.5 | 94.6 | 98.6 | 92.6 | 92.2 | 64.5 | −30.0 |
| π0 | 71.9 | 70.0 | 86.4 | 85.8 | 45.2 | 42.9 | −29.0 |
| SmolVLA | 82.2 | 78.8 | 92.2 | 88.0 | 69.6 | 27.9 | −54.3 |
| π0-FAST | 90.5 | 90.4 | 93.8 | 90.6 | 87.4 | 25.7 | −64.8 |

- **A training-free lookup is competitive**: 91.7% on LIBERO, rank 7 of 10 (the kNN full method and the nine VLAs).
  Under perturbation it degrades least (−9.3 points, against −12.2 for the most robust VLA and up to −64.8) and ranks
  3rd. Part of this is a favourable benchmark: LIBERO-plus scenes need the same actions as the clean demonstrations,
  so the answer is in memory whenever the key ignores the perturbation.
- **VLAs fail where the lookup fails**: when a VLA fails a LIBERO episode, the lookup fails the same episode in 20–46%
  of cases, against 8% overall, and per-task success correlates with the lookup's (median r = 0.74).
- **VLAs act like a lookup**: at the states a VLA visits, a lookup over the demonstrations (ResNet-18 key, k = 8)
  predicts its action chunk as well as it predicts a held-out human demonstration, or better (R² 0.90 for π0.5, 0.78
  for humans), and its chunks are closer to a training chunk than a new human demonstration is.
- **The weak VLAs key on the wording**: reworded instructions over the original scenes drop SmolVLA, π0-FAST and π0
  to 22, 33 and 12% (original wording: 82, 88, 68%).

### Backbone vs action head, before and after fine-tuning

A plain lookup keyed on each VLA's own VLM (one layer, z-scored, 2×2 grid per camera; the layer is chosen once per
family offline and kept for both checkpoints), before and after its LIBERO fine-tuning, next to the VLA itself:

| VLA | LIBERO: key before | key after | VLA | LIBERO-plus: key before | key after | VLA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| π0.5 | 85.4 | 81.8 | 97.7 | 60.5 | 65.7 | 85.5 |
| MolmoAct2 | 78.6 | 91.3 | 97.5 | 52.6 | 81.0 | 82.6 |
| GR00T N1.7 (frozen VLM) | 87.5 | 87.5 | 94.8 | 66.4 | 66.4 | 80.2 |
| VLA-JEPA | 84.0 | 83.0 | 97.2 | 49.3 | 65.7 | 74.0 |
| Evo-1 | 62.1 | 87.1 | 95.4 | 35.5 | **72.6** | 71.4 |
| X-VLA | 83.3 | 88.5 | 94.5 | 62.4 | **78.1** | 64.5 |
| π0 | 86.9 | 17.9 | 71.9 | 56.7 | 2.4 | 42.9 |
| SmolVLA | 76.3 | **86.5** | 82.2 | 45.0 | **69.3** | 27.9 |
| π0-FAST | 86.6 | **95.0** | 90.5 | 58.3 | **74.0** | 25.7 |

Bold: the lookup on the fine-tuned VLM beats the VLA.
- **Some action heads subtract**: 2 of 9 VLAs score below a lookup on their own fine-tuned VLM on LIBERO, 4 of 9 under
  perturbation, by up to 48 points. GR00T's LIBERO fine-tunes froze its VLM, so everything it learnt on LIBERO is in
  its action head.
- **Fine-tuning mostly makes the VLM a more robust key** (+5 to +37 points on LIBERO-plus for 7 of 8 VLAs) while
  clean success improves for only 5. It can also break it: π0's fine-tune collapses its key (86.9 → 17.9%).
- **What changes in the key** (`experiments/embedding_maps.py`: PCA and t-SNE of every key on the same 4,150 demo
  frames). Before fine-tuning the scene dominates every key: the suite explains 40–84% of the variance of the first two
  principal components, 1–43% after, lower for all 8 fine-tuned VLMs. The fine-tuned keys also track the phase of the
  task: the median progress gap to the nearest frame of another demonstration shrinks for 7 of 8 VLMs (Evo-1 0.031 →
  0.022). π0's fine-tuned key still tells the tasks apart (100% same-task neighbours on LIBERO-Goal) but loses the
  phase (0.028 → 0.052). Only the VLMs whose image tokens attend to the instruction (PaliGemma's bidirectional prefix
  in π0, π0.5 and π0-FAST; Florence-2's encoder in X-VLA) separate LIBERO-Goal's ten tasks, which share one scene:
  96–100% same-task neighbours, against 76–82% for the causal VLMs and 79% for ResNet-18.

### Scoring a backbone without rollouts

LAC@8 (local action consistency) = 1 − the action distance from a held-out demonstration frame to its 8 nearest frames
of other demonstrations / the distance to a random frame of the same task (`experiments/neighbor_metric.py`).
- LAC tracks the closed-loop success of a lookup with each key: Pearson r = 0.78, Spearman ρ = 0.61 over the 12 keys
  that never saw these demonstrations.
- It flags a broken backbone: π0's fine-tuned key has LAC 0.55, every other key 0.76–0.88. Fine-tuned VLMs score
  high on their own training demonstrations, so on them LAC also measures memorisation.
- As a loss it improves a key: one linear map per ResNet grid cell trained for LAC (`experiments/learn_key.py`) lifts
  the plain lookup 88.7 → 93.0% and the full method 91.7 → 93.0%.

### Ablations of the full method (official protocol, 2,000 episodes)

| Change | LIBERO | Δ |
| --- | ---: | ---: |
| none (full method, `lerobot-eval`) | 91.7 | |
| prompt ignored (Goal suite 8.8%) | 64.6 | −27.1 |
| raw 16×16 pixels as the key (Object suite 0%) | 57.1 | −34.6 |
| 10% / 25% / 50% of the demos | 77.4 / 82.6 / 88.5 | −14.3 / −9.1 / −3.2 |
| k = 1 (replay the nearest demo) | 78.6 | −13.1 |
| k = 2 / 4 / 8 / 24 / 32 (13 in the full method) | 85.2 / 89.3 / 89.8 / 92.7 / 91.7 | −6.5 / −2.4 / −1.9 / +1.0 / 0.0 |
| DINOv2-S instead of ResNet-18 | 83.4 | −8.3 |
| agent-view camera only / wrist camera only | 82.0 / 87.5 | −9.7 / −4.2 |
| plain weighted mean (no mode seeking) | 90.7 | −1.0 |
| no temporal ensembling / no history / no state / no time feature | 91.1 / 91.5 / 91.3 / 91.3 | −0.6 / −0.2 / −0.4 / −0.4 |
| hard task filter instead of the soft prompt distance | 91.9 | +0.2 |

The representation, the prompt and the amount of data do the work; every other extra moves success by about a point,
within the ±1.2-point 95% interval of 2,000 episodes. Retry (rewind and retrieve again when the scene looks unlike any
demonstration) does not help on LIBERO-Long (80.8% → 78.5–81.3%) and is off.

## How a retrieval works

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

## Install

```bash
uv sync --extra all           # the whole study: LIBERO, Optuna, and every VLA family used as a key or baseline
uv sync --extra experiments   # only the kNN policy, LIBERO evaluation and the hyper-parameter search
```

LeRobot is a plain dependency, unmodified, pinned in `pyproject.toml` to the GitHub commit the study ran on. That
commit is newer than the 0.6.1 release, which lacks GR00T N1.7, MolmoAct2, Evo-1 and VLA-JEPA. The extras are `vla`
(SmolVLA, π0 / π0-FAST / π0.5, X-VLA), `vla_jepa`, `groot`, `molmoact2` and `evo1`. `uv sync` makes the environment
match the given selection exactly, so pass every extra you need on each sync.

- Python 3.12 is pinned in `.python-version`. draccus 0.11, used by every `lerobot-*` CLI, cannot parse `X | None`
  fields on 3.14.
- Run everything from the repository root: the experiments read and write under `outputs/`.
- Set `HF_HUB_DISABLE_XET=1` if Hub downloads stall.

**LIBERO-plus** cannot share this environment: the fork replaces the `libero` package and pins robosuite 1.4.1 /
mujoco 3.7.0 (see LeRobot's `docker/Dockerfile.benchmark.libero_plus`). `outputs/libero_plus/setup.sh` builds a
separate venv with the same LeRobot commit and this package. Run its commands with
`PYTHONPATH=outputs/libero_plus/LIBERO-plus LIBERO_CONFIG_PATH=outputs/libero_plus/libero_config MUJOCO_GL=egl`.

## Fit and evaluate

```bash
# embed lerobot/libero and save a policy (features are cached under outputs/knn/feature_cache)
lerobot-knn-fit --repo_id=lerobot/libero --output_dir=outputs/knn/policies/my_knn \
    --policy.image_encoder=resnet18_l3 --policy.image_pooling=grid4 --policy.k=8

# standard LeRobot evaluation (inference knobs can be overridden: --policy.k=4 ...)
MUJOCO_GL=egl lerobot-eval --policy.path=outputs/knn/policies/my_knn --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.observation_height=256 --env.observation_width=256 \
    --eval.n_episodes=50 --eval.batch_size=10
```

The official configuration is `outputs/knn/v2_39.json` (KNNConfig fields; pass them as `--policy.<field>=<value>`).
`--env.observation_height/width=256` renders at the resolution of the `lerobot/libero` frames the memory was built
from. With `lerobot-eval`, keep `--env.max_parallel_tasks=1`: the per-env action queues are keyed by batch position,
which concurrent tasks would share.

## Reproducing the study

Every closed-loop run except the official `lerobot-eval` ones goes through `experiments/libero_pool.py`, a persistent
pool of simulator processes that reproduces `lerobot-eval`'s LIBERO semantics (same env, initial states, processors
and step limits) but schedules the episodes of all tasks asynchronously; episode `i` of a task uses initial state `i`.
Run each module as `uv run python -m lerobot_policy_knn.experiments.<module> --help` for its options.

| Step | Module | Writes |
| --- | --- | --- |
| hyper-parameter search on initial states 40–49 (`knn_v1`, `knn_v2`, `knn_v3_<suite>`) | `search --study=knn_v2 --space=v2` | `outputs/knn/optuna.db`; browse with `uv run optuna-dashboard sqlite:///outputs/knn/optuna.db` |
| re-score the best trials on held-out initial states 0–19 | `finalize` | `outputs/knn/final_candidates.json` |
| kNN configurations on the same episodes (`--init_states=0:50 --tag=full50` for the official protocol) | `compare <config.json> ...` | `outputs/knn/compare.jsonl`, `compare_episodes.jsonl` |
| one-change ablation configurations | `ablations` | `outputs/knn/ablations/*.json` |
| VLA closed loop (`--record` saves its chunks) | `vla_rollouts <name>` | `outputs/vla/results.jsonl`, `results_episodes.jsonl`, `rollouts/` |
| offline check of a VLA checkpoint on demo frames | `vla_sanity <name>` | `outputs/vla/hub_sanity.json` |
| VLM features of every memory frame | `vla_features <name>` | the feature cache |
| layer pilot and key choice per family | `embed_pilot`, `choose_keys` | `outputs/vla/embed_pilot.json`, `configs/plain_*.json`, `extra_keys.json` |
| does a lookup predict the VLA's chunks? | `head_vs_knn` | `outputs/vla/head_vs_knn_*.json` |
| LAC and its correlation with closed loop | `neighbor_metric --heldout`, `neighbor_metric --correlate` | `outputs/vla/neighbor_metric_heldout.json` |
| a key trained for LAC | `learn_key` | `outputs/vla/learned_key_resnet.pt` |
| PCA and t-SNE maps of every key, neighbour statistics (CPU) | `embedding_maps` | `outputs/vla/embedding_maps.json` |
| LIBERO-plus closed loop (separate venv) | `libero_plus_eval knn:<config.json> vla:<name> ...` | `outputs/libero_plus/results.jsonl`, `results_episodes.jsonl` |
| LIBERO-plus demos as queries; reworded instructions | `plus_neighbor_metric`, `plus_language` | `outputs/libero_plus/neighbor_metric_plus.json`, `language_sensitivity.json` |

The result files stay local (`outputs/` is ignored except configurations); `poster/data/poster_data.json` collects every
number the poster and paper use, `poster/data/embedding_maps.json` the 2-D maps.

## Poster and paper

```bash
make -C poster          # prepare_data.py -> make_figures.py -> poster/figures/, numbers.tex
make -C poster/corl2026 # the CoRL 2026 workshop poster -> poster.pdf (HTML + SVG, Chromium)
make -C paper           # the same scripts in paper mode -> paper.pdf (CoRL 2026 workshop template, pdfLaTeX)
```

No number in `paper.tex` is typed by hand: they come from the generated `numbers.tex` and tables. See
[`poster/README.md`](poster/README.md) and [`poster/corl2026/README.md`](poster/corl2026/README.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/lerobot_policy_knn/` | the plugin: `configuration_knn.py`, `modeling_knn.py`, `processor_knn.py`, `memory.py`, `features.py`, `encoders.py`, `fit.py` (`lerobot-knn-fit`), `vla.py` (VLM keys and VLA rollouts) |
| `src/lerobot_policy_knn/experiments/` | the study's evaluation and analysis modules (table above); `common.py` holds the paths and helpers they share |
| `outputs/knn/`, `outputs/vla/configs/` | tracked configurations: Optuna trials, `v2_39.json`, ablations, k sweep, retry, TTA, and the VLM keys |
| `outputs/libero_plus/` | `setup.sh` and `venv_constraints.txt` for the LIBERO-plus venv |
| `poster/`, `paper/` | the figure scripts and their data, the CoRL 2026 workshop poster (`poster/corl2026/`) and the workshop paper |
| `docs/` | experiment notes and literature review |
| `tests/` | `uv run pytest tests` |
