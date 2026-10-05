# 3. Backbone vs action expert: a VLA's VLM as a lookup key

## How a VLM becomes a key

`VLAEmbedder` (`vla.py`) runs only the **prefix** of a VLA through its VLM: images, prompt, and for π0.5 its
discretised state. It pools the hidden states, and the kNN uses them as retrieval keys
(`image_encoder="vla:<name>"`). Features for every memory frame are cached by `experiments/vla_features.py`.

- **Poolings.**
  - `img`: the mean of a camera's image tokens.
  - `grid2`: a 2×2 spatial grid of image tokens, per camera.
  - `lang`, `all`: means of the prompt tokens and of all prefix tokens.
  - `state`: SmolVLA's state token.
- **Layer taps.**
  - `grid2@L<n>`: the residual stream after *n* LM layers.
  - `grid2@vision`: the vision tokens entering the LM.
  - Multi-layer keys concatenate taps, e.g. `grid2@L12+L15+last`.
- **Key normalisation (`key_norm`).**
  - `none`: per-cell unit norm.
  - `zscore`: per-dimension standardisation, fitted on the memory.
  - `zscore_pca<n>`: a whitened PCA of the standardised key, seeded so it is reproducible.
  - `file:<path>`: a learnt projection (see 04).

  The implementation is `fit_key_norm` / `normalize_keys` in `memory.py`.
- **Choosing the layer.** `experiments/embed_pilot.py` predicts held-out demo chunks with a plain lookup (k = 8) for
  every tap and normalisation on 10 tasks; `experiments/choose_keys.py` takes the tap with the best mean held-out
  action R² over a family's checkpoints before and after LIBERO fine-tuning, so both use the same layer. The chosen
  keys (z-scored, 2×2 grid) are π0.5, π0 and π0-FAST layer 12, X-VLA layer 8 of its Florence-2 encoder, VLA-JEPA layer
  7, GR00T layer 4, Evo-1 layer 10, MolmoAct2 layer 27, and SmolVLA's last layer (`outputs/vla/extra_keys.json`,
  `outputs/vla/configs/plain_*.json`).

"Plain lookup" below means k = 8, the hard task filter, and none of the full method's extras.

## Key recipes (development episodes: initial states 20–29, 400 episodes)

**SmolVLA and generic encoders**

| Key | Success |
|---|---|
| ImageNet ResNet-18, grid (reference) | 87.3 |
| ImageNet ResNet-18, mean | 70.3 |
| DINOv2-S | 74.5 |
| SmolVLM2 (no robot data), img | 63.0 |
| SmolVLA base, img / grid | 62.3 / 74.5 |
| SmolVLA LIBERO fine-tune, img / grid | 77.8 / **85.3** |

**π0.5 (PaliGemma, 18 layers, 2,048 dims)**

| Key recipe | base (no LIBERO) | LeRobot LIBERO fine-tune |
|---|---|---|
| last layer, raw cosine | not run | 71.0 |
| last layer, z-scored | 81.3 | 79.3 |
| layer 15, z-scored | 82.3 | 79.0 |
| **layer 12, z-scored** | **88.5** | 84.5 |
| layer 12, whitened PCA-512 | 88.0 | 79.8 |
| layers 12+15+18, whitened PCA-512 | 84.5 | 80.5 |
| layers 12+15(+18), z-scored | ran out of RAM | ran out of RAM |

**Why VLM keys need care.** 8 of π0.5's 2,048 channels carry about 78% of the hidden-state energy ("massive
activations"), so plain cosine mostly compares outliers. Per-dimension z-scoring fixes that: 71.0 → 79.3 on the LIBERO
fine-tune. Upper-middle layers then key better than the last. Every VLM key in the study is z-scored.

**Several layers do not help.** π0.5's action expert attends to every backbone layer, so we tried combining layers:
offline on 10 tasks (equal-weight sums of per-layer distances, and Dirichlet-searched weights; LIBERO upper layers
combined reach R² 0.845, base 0.80) and in closed loop (a whitened PCA of concatenated layers). Neither beats layer 12
alone. The z-scored multi-layer keys (≥ 2 × 8,192 dims per camera) exhausted RAM and swap in the simulator pool. The
offline multi-layer sweep (`multilayer_pilot.py`) was removed in the cleanup; it is in commit `b4acfae`.

## Offline pilots (held-out action R², k = 8, 10 tasks; `outputs/vla/embed_pilot.json`)

- **SmolVLA.** Every tap gives the same R² (base 0.785–0.789; LIBERO 0.852–0.854). Its residual stream is dominated
  by the scaled vision embeddings.
- **π0.5 base.** About 0.79–0.80 at every depth.
- **π0.5 LIBERO fine-tune.** Rises with depth: vision 0.80, L3 0.78, L6 0.80, L9 0.83, L12 0.83, L15 0.84, last 0.825.
  Fine-tuning changes only the upper layers.
- **π0.** Base 0.73–0.80 at every tap from the vision tokens to the last layer; LIBERO fine-tune 0.39–0.48 at every
  tap.

An earlier leave-one-episode-out R² ladder (`embed_quality.py`, removed in the cleanup, in commit `b4acfae`) had much
lower resolution than closed loop (ResNet-18 0.784, DINOv2 0.750, SmolVLA base / LIBERO grid 0.771 / 0.852); the
neighbour metric (04) replaced it.

## Before vs after fine-tuning, and the action expert (official protocol)

A plain lookup keyed on each VLA's own VLM before and after its LIBERO fine-tuning, next to the VLA itself. LIBERO:
initial states 0–49 (2,000 episodes); LIBERO-plus: the 420-variant sample (05).

| VLA | LIBERO: key before | key after | VLA | action-expert gain | LIBERO-plus: key before | key after | VLA | gain |
|---|---|---|---|---|---|---|---|---|
| π0.5 | 85.4 | 81.8 | 97.7 | +15.9 | 60.5 | 65.7 | 85.5 | +19.8 |
| MolmoAct2 | 78.6 | 91.3 | 97.5 | +6.2 | 52.6 | 81.0 | 82.6 | +1.6 |
| GR00T N1.7 (frozen VLM) | 87.5 | 87.5 | 94.8 | +7.3 | 66.4 | 66.4 | 80.2 | +13.8 |
| VLA-JEPA | 84.0 | 83.0 | 97.2 | +14.2 | 49.3 | 65.7 | 74.0 | +8.3 |
| Evo-1 | 62.1 | 87.1 | 95.4 | +8.3 | 35.5 | 72.6 | 71.4 | **−1.2** |
| X-VLA | 83.3 | 88.5 | 94.5 | +6.0 | 62.4 | 78.1 | 64.5 | **−13.6** |
| π0 | 86.9 | 17.9 | 71.9 | +54.0 (−15.0 vs the key before) | 56.7 | 2.4 | 42.9 | +40.5 (−13.8) |
| SmolVLA | 76.3 | 86.5 | 82.2 | **−4.3** | 45.0 | 69.3 | 27.9 | **−41.4** |
| π0-FAST | 86.6 | 95.0 | 90.5 | **−4.5** | 58.3 | 74.0 | 25.7 | **−48.3** |

The action-expert gain is the VLA minus the lookup on its fine-tuned VLM.

- **Some action experts subtract.** 2 of 9 VLAs score below a lookup on their own fine-tuned VLM on LIBERO (SmolVLA,
  π0-FAST), 4 of 9 under perturbation (also Evo-1 and X-VLA), by up to 48 points. What LIBERO fine-tuning buys
  SmolVLA and π0-FAST is a better key, not a better controller.
- **Fine-tuning mostly makes the VLM a more robust key.** +5 to +37 points on LIBERO-plus for 7 of the 8 VLAs with a
  separate checkpoint before fine-tuning, while clean LIBERO success improves for only 5 (π0.5 and VLA-JEPA lose a
  little).
- **Fine-tuning can break the key.** π0's LIBERO fine-tune turns a strong key (86.9%) into a useless one (17.9%;
  2.4% on LIBERO-plus). Its action expert carries the damaged backbone to 71.9%, still below a lookup on the VLM it
  started from.
- **π0.5's and MolmoAct2's experts add the most where the lookup is weakest**, LIBERO-Long.

**π0's broken key in detail** (development episodes: 86.0 → 15.0).
- Offline, the same drop (held-out action R² on 4 tasks): 0.79–0.86 for the base VLM vs 0.40–0.73 for the fine-tune,
  and at every depth (pilot above). The fine-tune's layer-12 activations are about 100× larger (per-dimension std 419
  vs 2 for π0.5; max 21k vs 280 for the base).
- Not an embedding bug: online and cached embeddings of the same frames agree (cosine 1.000).

**SmolVLA's fine-tune rewrites the vision side.** Relative weight change base → LIBERO is 9.4% in the vision encoder,
24.3% in the connector, 18.2% in the action expert, and only 1.4% in the language model
(`outputs/vla/weight_change_smolvla.json`).

**GR00T N1.7 learns LIBERO only in its action head.** Its LIBERO checkpoints were trained with `tune_llm=False` and
`tune_visual=False`.
- All 493 backbone tensors of the Spatial checkpoint are bit-identical to the base `nvidia/GR00T-N1.7-3B`; the
  action-head tensors changed.
- The base's pretraining data (its `experiment_cfg`) are DROID and real robots, with no LIBERO, so its key never
  saw these demos.
- Size: backbone 1.52B parameters (Cosmos-Reason2 / Qwen3-VL, truncated to the 16 layers the head reads), action
  head 1.62B.
- One key covers all four per-suite checkpoints: the base network with a LIBERO checkpoint's saved processors
  (`groot_base`). Its features match the Spatial checkpoint's exactly.
- So the gap between GR00T and a lookup on that key is entirely what its action head learnt from the demos.

X-VLA's key is layer 8 of its Florence-2 multimodal encoder (agent view and prompt); its wrist view reaches the
policy as raw vision-tower tokens, which are that camera's key.

## What fine-tuning changes in the key: embedding maps (`embedding_maps`)

Every key on the same 4,150 demonstration frames (6 random demos per task, every 10th frame), computed as the plain
lookup computes it (memory z-score statistics from 20k random memory frames, unit norm per camera, cameras
concatenated). Maps: PCA (first two components) and t-SNE (perplexity 30, on the first 50 components); the poster and
the paper show a few VLMs before / after (`embedding_pairs.pdf`), the paper's appendix every key
(`embedding_tsne_progress.pdf`, `embedding_pca_suite.pdf`). Measured in the full key space:
- **same-task neighbours**: how often a frame's nearest neighbour from another demonstration is of the same task (all
  40 tasks; LIBERO-Goal, where the ten tasks share one scene and only the instruction differs);
- **phase gap**: median |progress difference| (progress = frame / episode length) to the nearest frame of another
  demonstration of the same task;
- **suite share**: the share of the variance of the 2-D PCA map that the suite (the scene) explains.

| Key | stage | same-task neighbours | on LIBERO-Goal | phase gap | suite share of PC1+2 |
|---|---|---|---|---|---|
| ResNet-18 grid | generic | 91% | 79% | 0.030 | 85% |
| DINOv2-S | generic | 86% | 78% | 0.031 | 70% |
| ResNet-18 grid (learnt key) | learnt | 94% | 81% | 0.026 | 69% |
| SmolVLA base grid | before | 93% | 79% | 0.031 | 69% |
| SmolVLA LIBERO grid | after | 90% | 79% | 0.021 | 1% |
| π0.5 base L12 | before | 98% | 100% | 0.027 | 40% |
| π0.5 LIBERO L12 | after | 96% | 99% | 0.026 | 5% |
| π0 base L12 | before | 99% | 100% | 0.028 | 58% |
| π0 LIBERO L12 | after | 100% | 100% | 0.052 | 24% |
| X-VLA base grid2@L8 | before | 98% | 96% | 0.030 | 77% |
| X-VLA LIBERO grid2@L8 | after | 96% | 99% | 0.024 | 23% |
| π0-FAST base grid2@L12 | before | 98% | 100% | 0.027 | 40% |
| π0-FAST LIBERO grid2@L12 | after | 99% | 99% | 0.024 | 16% |
| VLA-JEPA base grid2@L7 | before | 93% | 82% | 0.029 | 84% |
| VLA-JEPA LIBERO grid2@L7 | after | 84% | 76% | 0.026 | 43% |
| GR00T N1.7 frozen VLM grid2@L4 | before | 94% | 81% | 0.030 | 68% |
| Evo-1 base grid2@L10 | before | 93% | 79% | 0.031 | 71% |
| Evo-1 LIBERO grid2@L10 | after | 86% | 81% | 0.022 | 4% |
| MolmoAct2 base grid2@L27 | before | 94% | 81% | 0.028 | 67% |
| MolmoAct2 LIBERO grid2@L27 | after | 91% | 78% | 0.028 | 27% |

- **The scene stops dominating.** Before fine-tuning the suite explains 40–84% of the first two components' variance
  (ResNet-18 85%); after it 1–43%, lower for all 8 fine-tuned VLMs. This fits the fine-tuned keys being more robust
  lookup keys under LIBERO-plus's scene perturbations.
- **The key tracks the phase.** The phase gap shrinks for 7 of 8 VLMs (SmolVLA 0.031 → 0.021, Evo-1 0.031 → 0.022,
  X-VLA 0.030 → 0.024; MolmoAct2 barely moves). The t-SNE maps show it as long start-to-end trajectories.
- **π0's fine-tune keeps the task, loses the phase.** Its fine-tuned key finds a same-task neighbour 100% of the time,
  but its phase gap doubles (0.028 → 0.052) and its t-SNE map is one tight cluster per task. Its lookup collapses
  (86.9 → 17.9%) and its LAC is the lowest of all keys (04).
- **Only VLMs whose image tokens see the instruction separate LIBERO-Goal's tasks.** PaliGemma (π0, π0.5, π0-FAST)
  attends bidirectionally over the image and text prefix, and X-VLA's key is a Florence-2 encoder layer over image and
  prompt: 96–100% same-task neighbours on LIBERO-Goal. The other VLMs attend causally with the image tokens before the
  instruction: 76–82%, like ResNet-18 (79%).

## Same state, same action: does the head behave like a lookup?

At the states each VLA visited in recorded clean LIBERO rollouts (`vla_rollouts --record`), a plain lookup over the
demonstrations (ResNet-18 key, k = 8) predicted its chunk; R² against the VLA's own chunk (`experiments/head_vs_knn.py`,
`outputs/vla/head_vs_knn_*.json`). The same lookup predicts a held-out human demonstration with R² 0.78.

| VLA | chunks | R², lookup → VLA chunk | median distance to the nearest training chunk |
|---|---|---|---|
| MolmoAct2 | 1,257 | 0.93 | 0.0012 |
| VLA-JEPA | 1,750 | 0.93 (its human reference: 0.78) | 0.0009 (human: 0.0015) |
| π0.5 | 1,332 | 0.90 | 0.0012 |
| Evo-1 | 959 | 0.90 | 0.0015 |
| π0-FAST | 1,452 | 0.80 | 0.0018 |
| GR00T N1.7 (Spatial) | 307 | 0.76 | 0.0020 |
| SmolVLA | 1,571 | 0.66 | 0.0016 |
| π0 | 1,752 | 0.63 | 0.0014 |
| a held-out human demo | | 0.78 | 0.0020 |

X-VLA predicts absolute poses and is left out. A second sample of the VLA itself (noise floor) gives R² 0.99.

**Reading.** The strong VLAs' chunks are lookup-like state by state, at least as predictable from the demonstrations as
a new human demonstration, and every VLA's chunks are closer to some training chunk than a new demonstration is: they
are not new. Their advantage sits in the few states where no neighbour fits. In the VLA's own key space the lookup
is similar (SmolVLA: R² 0.79 in its LIBERO-tuned VLM at k = 8, 0.65 before fine-tuning; π0.5: 0.88). Off the
demonstrations, on LIBERO-plus, the robust VLAs stay lookup-like and the collapsing ones do not (05).
