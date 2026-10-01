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

"Plain lookup" below means k = 8, the hard task filter, and none of the full method's extras.

## Closed-loop lookup success on VLM keys (same 400 episodes)

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

**Why π0.5 needs care.** 8 of its 2,048 channels carry about 78% of the hidden-state energy ("massive activations"),
so plain cosine mostly compares outliers. Per-dimension z-scoring fixes that: 71.0 → 79.3 on the LIBERO fine-tune.
Upper-middle layers then key better than the last.

**Several layers do not help.** π0.5's action expert attends to every backbone layer, so we tried combining layers:
- offline, on 10 tasks: equal-weight sums of per-layer distances, and Dirichlet-searched weights;
- closed loop: a whitened PCA of concatenated layers.

Neither beats layer 12 alone. The z-scored multi-layer keys (≥ 2 × 8,192 dims per camera) exhausted RAM and swap
in the simulator pool.

## π0: fine-tuning broke its backbone as a key

- **Closed loop.** A plain lookup on π0's VLM at layer 12 (z-scored) scores **86.0%** with `lerobot/pi0_base`, but
  **15.0%** with its LeRobot LIBERO fine-tune (v044). π0 itself scores 74.8%.
- **Offline, the same drop** (held-out action R² on 4 tasks): 0.79–0.86 for the base VLM vs 0.40–0.73 for the
  fine-tune. The fine-tune's layer-12 activations are about 100× larger (per-dimension std 419 vs 2 for π0.5; max
  21k vs 280 for the base).
- **Not an embedding bug.** Online and cached embeddings of the same frames agree (cosine 1.000).
- **Reading.** LIBERO fine-tuning of π0 turned a strong key into a useless one, and its action expert carries the
  damaged backbone to 74.8%. That is still below a lookup on the VLM it started from.
- **Check queued (phase P2).** A 10-task layer pilot (vision tokens, L3–L15, last), to see whether another depth is
  still a good key.

## Offline pilots (held-out action R², k = 8)

These use 10 tasks and the `*_pilot` caches (`outputs/vla/embed_pilot.json`, `multilayer_pilot.json`):
- **SmolVLA.** Every tap gives the same R² (base 0.785–0.789; LIBERO 0.852–0.854). Its residual stream is dominated
  by the scaled vision embeddings.
- **π0.5 base.** About 0.79–0.80 at every depth.
- **π0.5 LIBERO fine-tune.** Rises with depth: vision 0.80, L3 0.78, L6 0.80, L9 0.83, L12 0.83, L15 0.84, last 0.825.
  Fine-tuning changes only the upper layers.
- **Multi-layer, offline.** LIBERO upper layers combined reach 0.845; base 0.80 (learnt weights).

The leave-one-episode-out ladder (`outputs/vla/embed_quality.jsonl`, k = 8):

| Key | R² |
|---|---|
| ResNet-18 | 0.784 |
| DINOv2 | 0.750 |
| SmolVLM2 img | 0.753 |
| SmolVLA base img / grid | 0.753 / 0.771 |
| SmolVLA LIBERO img / grid | 0.846 / 0.852 |
| π0.5 LIBERO grid (raw) | 0.795 |

Offline R² has much lower resolution than the closed-loop differences above; the neighbour metric (04) works better.

## Before vs after fine-tuning

- **SmolVLA.** Fine-tuning makes its VLM a much better key: 74.5 → 85.3.
  - It rewrites the vision encoder: relative weight change base → LIBERO is 9.4% in the vision encoder, 24.3% in
    the connector, 18.2% in the action expert, and only 1.4% in the language model (`weight_change_smolvla.json`).
- **π0.5.** Fine-tuning slightly *worsens* its key in closed loop: layer 12, 88.5 → 84.5. Only its upper layers
  change.

## The diagnostic plane: backbone vs action expert

The plain lookup is keyed on the VLA's own VLM (best recipe). The VLA's excess over it is what the action expert
adds. The poster draws this as one row per VLA (`diagnostic.pdf`). It shows the same 400 LIBERO episodes and, from
05, the same 420 LIBERO-plus variants.

| VLA | Lookup on its own VLM | VLA | Action-expert gain | LIBERO-plus: lookup / VLA / gain |
|---|---|---|---|---|
| π0.5 | 84.5 (layer 12) | 97.0 | **+12.5** | 65.7 / 85.5 / **+19.8** |
| X-VLA | 87.8 (Florence-2 encoder layer 8; base: 86.0) | 93.0 | **+5.3** | 73.6 / 64.5 / **−9.1** (base key: 68.3) |
| SmolVLA | 85.3 | 80.0 | **−5.3** | 69.3 / 27.9 / **−41.4** |
| π0 | 15.0 (layer 12, LIBERO fine-tune); 86.0 with its base VLM | 74.8 | +59.8 over its own VLM; −11.2 vs a lookup on its pre-fine-tune VLM | 2.4 / 42.9 / +40.5 (base key: 56.7) |
| π0-FAST, VLA-JEPA | phase W (keys: layer 12 / layer 7, chosen by the pilot) | 89.8 / 96.5 | running | running |
| GR00T N1.7 | phase Y (its frozen VLM, layer chosen by the pilot) | phase X3 | queued | queued |

**GR00T N1.7 learns LIBERO only in its action head.** Its LIBERO checkpoints were trained with `tune_llm=False` and
`tune_visual=False`.
- All 493 backbone tensors of the Spatial checkpoint are bit-identical to the base `nvidia/GR00T-N1.7-3B`; the
  action-head tensors changed.
- The base's pretraining data (its `experiment_cfg`) are DROID and real robots, with no LIBERO, so its key never
  saw these demos.
- Size: backbone 1.52B parameters (Cosmos-Reason2 / Qwen3-VL, truncated to the 16 layers the head reads), action
  head 1.62B.
- One key covers all four per-suite checkpoints: the base network with a LIBERO checkpoint's saved processors.
  Its features match the Spatial checkpoint's exactly.
- So the gap between GR00T and a lookup on that key is entirely what its action head learnt from the demos.

X-VLA's key is layer 8 of its Florence-2 multimodal encoder (agent view and prompt) z-scored and pooled to a 2×2
grid; its wrist view reaches the policy as raw vision-tower tokens, which are that camera's key. `choose_keys.py`
picked the layer with the best mean held-out action R² over the base and LIBERO checkpoints, so both use the same
layer. Fine-tuning barely changes it as a key (86.0 → 87.8).

**Reading.**
- Small VLAs are "secretly kNN". SmolVLA's learned action expert is *worse* than a lookup on its own VLM. What LIBERO
  fine-tuning buys SmolVLA is a better key.
- π0.5's action expert adds 12.5 points on top of its best VLM key, mostly on LIBERO-Long (97% vs 73% for the
  lookup on its VLM).

## Same state, same action: does the head behave like a lookup?

At the states each VLA visited in closed loop, we predicted its chunk with the lookup in several key spaces and
measured R² against the VLA's own chunk (`outputs/vla/head_vs_knn_*.json`). A second sample of the VLA itself
(noise floor) gives R² 0.99.

| VLA | Key space | R², k = 1 | R², k = 8 |
|---|---|---|---|
| SmolVLA (1,571 chunks) | its own LIBERO-tuned VLM (grid) | 0.65 | 0.79 |
| | its VLM before fine-tuning | 0.38 | 0.65 |
| | ResNet-18 | 0.44 | 0.66 |
| π0.5 (1,332 chunks) | its own LIBERO-tuned VLM (grid) | 0.81 | 0.88 |
| | ResNet-18 | 0.78 | 0.90 |

Median distance of a VLA chunk to the nearest *training* chunk: π0.5 0.0012, SmolVLA 0.0016. A new human demo is
0.0020 from the nearest other demo.

**Reading.** Even π0.5's chunks are lookup-like state by state, and closer to a training chunk than a new demo is. Its
advantage sits in the few states where no neighbour fits.
