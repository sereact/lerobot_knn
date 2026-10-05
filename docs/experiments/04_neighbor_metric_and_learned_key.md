# 4. Scoring a backbone without rollouts, and training a key for it

## The metric: local action consistency (LAC)

A lookup only ever uses a key's *nearest* neighbours, so a key is scored on how alike the actions of close frames
are. Far-apart pairs, which a lookup never uses, don't count. The implementation is
`experiments/neighbor_metric.py`.

- **Queries** are frames with `frame_index % 10 == 4`, 120 per task. With `--heldout`, queries come only from the
  held-out 10% of demos in `outputs/vla/split.json`.
- **Candidates** are the memory frames (even `frame_index`) of the same task from **other** demos.
- *d_e* is the retrieval distance in the key space. *d_a* is the mean squared difference of the 10-step motion
  chunks that follow two frames.
- **LAC@k** = 1 − mean *d_a*(query, its *k* nearest in *d_e*) / mean *d_a*(query, a random candidate). A value of 1
  means the neighbours act identically; 0 means they are no better than a random frame of the task.
- For comparison:
  - `local_rho`: the Spearman correlation of *d_e* and *d_a* within the 64 nearest.
  - `global_rho`: the same over all pairs.

Every key is normalised exactly as the policy does it, through the shared `fit_key_norm`.

## Results (held-out queries; `outputs/vla/neighbor_metric_heldout.json`)

Closed-loop success is the plain lookup with that key on the official protocol (2,000 episodes;
`neighbor_metric --correlate --protocol full50`). "Fair" keys never saw these demos.

| Key | LAC@8 | Closed-loop plain lookup | Fair? |
|---|---|---|---|
| ResNet-18 grid | 0.790 | 88.7 | yes |
| ResNet-18 grid, **learnt key** | 0.821 | **93.0** | yes (trained on the other 90%) |
| DINOv2-S | 0.756 | 73.1 | yes |
| SmolVLA base / LIBERO grid | 0.769 / 0.840 | 76.3 / 86.5 | yes / no |
| π0.5 base / LIBERO L12 | 0.811 / 0.829 | 85.4 / 81.8 | yes / no |
| π0 base / LIBERO L12 | 0.809 / **0.548** | 86.9 / **17.9** | yes / no |
| π0-FAST base / LIBERO L12 | 0.811 / 0.880 | 86.6 / 95.0 | yes / no |
| X-VLA base / LIBERO L8 | 0.796 / 0.821 | 83.3 / 88.5 | yes / no |
| VLA-JEPA base / LIBERO L7 | 0.796 / 0.822 | 84.0 / 83.0 | yes / no |
| Evo-1 base / LIBERO L10 | 0.760 / 0.880 | 62.1 / 87.1 | yes / no |
| MolmoAct2 base / LIBERO L27 | 0.808 / 0.844 | 78.6 / 91.3 | yes / no |
| GR00T N1.7 frozen VLM L4 | 0.785 | 87.5 | yes |

The file also holds the π0.5 and SmolVLA recipe variants of 03 (other layers, mean pooling, PCA), which are left out of
the correlation (one key per backbone).

## Does it track closed-loop success?

| Keys | n | Pearson LAC@8 | Spearman LAC@8 | Spearman `global_rho` |
|---|---|---|---|---|
| never trained on LIBERO (the "fair" set, incl. the learnt key) | 12 | **0.78** | **0.61** | 0.37 |
| all, including the fine-tuned VLMs | 20 | 0.93 | 0.62 | 0.51 |

- **LAC tracks the closed-loop success of a lookup with each key**, but does not rank close keys reliably (ρ 0.61):
  most keys sit within a few hundredths of LAC and a few points of success. An earlier version on the 400 development
  episodes, without π0's keys, gave ρ 0.90 over 10 fair keys; on the official protocol with every family the rank
  correlation is weaker, so the claim is "tracks", not "ranks".
- **It flags a broken backbone.** π0's fine-tuned key has LAC 0.548, against 0.76–0.88 for every other key; its lookup
  scores 17.9%. Most of the all-keys Pearson correlation comes from that point.
- **Only close pairs matter.** The correlation over all pairs (`global_rho`) is weaker; `local_rho` is even negatively
  related.
- **Memorisation inflates offline metrics.** The fine-tuned VLMs were trained on every one of these demos, including
  our "held-out" ones, so their offline LAC is flattered: LAC rises with fine-tuning for every family except π0
  (e.g. π0.5 0.811 → 0.829) while π0.5's and VLA-JEPA's closed loop slightly drops. On fine-tuned VLMs LAC measures
  memorisation too. LIBERO-plus demos, which no checkpoint saw, remove that confound (05, `plus_neighbor_metric`).

## Training a key for LAC (`experiments/learn_key.py`)

- **Model.** A linear map shared by the 16 grid cells of each camera's ResNet-18 layer-3 features: standardised,
  256 → 128 per cell, initialised near identity.
- **Objective.** Soft-kNN regression. The action chunk of a frame is predicted from the softmax-weighted chunks of
  its key neighbours in *other* demos of the same task, with a learnt temperature. 3,000 Adam steps, batches of
  256 queries against 2,048 candidates.
- **Data.** Only the `train` episodes (90%) of `outputs/vla/split.json`.
- **Output.** Saved as `key_norm="file:outputs/vla/learned_key_resnet.pt"`: mean, std and a block-diagonal
  projection per camera. The unchanged kNN policy uses it (`outputs/vla/configs/plain_resnet18_l3_learned.json`,
  `full_resnet18_l3_learned.json`).

| | before | after |
|---|---|---|
| held-out LAC@8 (`learn_key`'s own evaluation; memory = train demos) | 0.800 | 0.821 |
| held-out kNN R² (same) | 0.779 | 0.803 |
| held-out LAC@8 (`neighbor_metric --heldout`) | 0.790 | 0.821 |
| **closed loop, plain lookup** (official protocol) | 88.7 | **93.0** |
| **closed loop, full method** (`v2_39` + learnt key, `lerobot-eval`) | 91.7 | **93.0** |
| LIBERO-plus, plain lookup / full method | 75.5 / 82.4 | 76.7 / 81.7 |

**Reading.** Optimising "close embeddings → close actions" directly improves the kNN in closed loop. The plain lookup
gains 4.3 points and overtakes the hand-tuned full method with none of its extras; the full method gains 1.3. Under
perturbation the learnt key helps the plain lookup a little and the full method not at all.
