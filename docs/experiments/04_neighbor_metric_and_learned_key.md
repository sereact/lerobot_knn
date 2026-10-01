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

| Key | LAC@8 | Closed-loop plain lookup | Saw these demos? |
|---|---|---|---|
| ResNet-18 grid | 0.790 | 87.3 | no |
| ResNet-18 grid, **learnt key** | **0.821** | **90.5** | trained on the other 90% |
| DINOv2-S | 0.756 | 74.5 | no |
| SmolVLA base grid / mean | 0.769 / 0.749 | 74.5 / 62.3 | no |
| SmolVLA LIBERO grid / mean | 0.840 / 0.834 | 85.3 / 77.8 | yes |
| π0.5 base last / L12 / L15 | 0.805 / 0.811 / 0.803 | 81.3 / 88.5 / 82.3 | no |
| π0.5 base L12 PCA / L12+15+18 PCA | 0.813 / 0.810 | 88.0 / 84.5 | no |
| π0.5 LIBERO last raw / last / L12 / L15 | 0.798 / 0.829 / 0.829 / 0.832 | 71.0 / 79.3 / 84.5 / 79.0 | yes |
| π0.5 LIBERO L12 PCA / L12+15+18 PCA | 0.813 / 0.824 | 79.8 / 80.5 | yes |

## Does it predict closed-loop success?

| Keys | n | Spearman LAC@8 | Pearson LAC@8 | Spearman `global_rho` |
|---|---|---|---|---|
| never trained on LIBERO (the "fair" set, incl. the learnt key on held-out queries) | 10 | **0.90** | 0.91 | 0.31 |
| all, including the fine-tuned VLMs | 18 | 0.33 | | −0.03 |

- **Only close pairs matter.** LAC ranks the fair keys by their closed-loop success. The correlation over all pairs
  (`global_rho`) does not; `local_rho` is even negatively related.
- **Memorisation inflates offline metrics.** The fine-tuned VLMs were trained on every one of these demos, including
  our "held-out" ones, so their offline LAC is flattered. They sit below the fair keys' trend line, which is why the
  correlation over all 18 keys is weak.
- **Before vs after fine-tuning, LAC goes up:**
  - SmolVLA 0.769 → 0.840;
  - π0.5 L12 0.811 → 0.829.

  For π0.5 that gain does not carry over to closed loop (88.5 → 84.5): the offline increase is the memorisation
  effect.

## Training a key for LAC (`experiments/learn_key.py`)

- **Model.** A linear map shared by the 16 grid cells of each camera's ResNet-18 layer-3 features: standardised,
  256 → 128 per cell, initialised near identity.
- **Objective.** Soft-kNN regression. The action chunk of a frame is predicted from the softmax-weighted chunks of
  its key neighbours in *other* demos of the same task, with a learnt temperature. 3,000 Adam steps, batches of
  256 queries against 2,048 candidates.
- **Data.** Only the `train` episodes (90%) of `outputs/vla/split.json`.
- **Output.** Saved as `key_norm="file:outputs/vla/learned_key_resnet.pt"`: mean, std and a block-diagonal
  projection per camera. The unchanged kNN policy uses it.

| | before | after |
|---|---|---|
| held-out LAC@8 (`learn_key`'s own evaluation; memory = train demos) | 0.800 | 0.821 |
| held-out kNN R² (same) | 0.779 | 0.803 |
| held-out LAC@8 (`neighbor_metric --heldout`) | 0.790 | 0.821 |
| **closed loop, plain lookup** | 87.3 | **90.5** |
| **closed loop, full method** (`v2_39` + learnt key) | 90.6 | **91.5** |

**Reading.** Optimising "close embeddings → close actions" directly improves the kNN in closed loop. The plain lookup
gains 3.2 points, above noise, and matches the hand-tuned full method with none of its extras. The full method gains
0.9, which is within noise.
