# 1. The kNN policy on LIBERO

## Method

`KNNPolicy` (`modeling_knn.py`) is a LeRobot policy with no trainable parameters. "Fitting" means embedding the
demonstrations (`memory.py`, `lerobot-knn-fit`).

- **Memory.** Every 2nd frame of the 1,693 `lerobot/libero` demos (137k keys), each with the action chunk that
  followed it.
- **Key φ.** A frozen ImageNet ResNet-18, layer 3, pooled on a 4×4 grid, for both cameras. Cells are unit-normalised.
  Alternatives tried:
  - DINOv2-S;
  - raw 16×16 pixels;
  - the hidden states of a VLA's VLM (see 03).
- **Retrieve.** The *k* nearest frames of the current task, at most one per demo. The task comes from the instruction:
  a hard filter on the nearest task prompt, or a "soft" text distance. Retrieval takes about 0.5 ms on the GPU.
- **Act.** A softmax-weighted average of the neighbours' chunks.
  - The *plain lookup* does only this.
  - The **full method** adds:
    - a robot-state distance;
    - a prompt text distance;
    - a time-step feature;
    - history matching (neighbours that continue the previous retrieval);
    - action-mode seeking (keep only the chunks that agree);
    - temporal ensembling.

The final configuration is `outputs/knn/v2_39.json`: k = 13, chunk 8, 1 action per step, text weight 0.48,
state weight 0.034, mode bandwidth 0.35, temporal ensembling 0.42.

## Hyper-parameter search

Optuna studies ran on initial states 40–49 (`outputs/knn/optuna.db`), 193 trials in total, each over 400 episodes:

| Study | Trials |
|---|---|
| `knn_v1` | 29 |
| `knn_v2` | 24 |
| `knn_v3_<suite>`, per suite | 29–43 |

The candidates were then re-scored on held-out initial states 0–19:
- `knn_v2#39` 91.6%;
- `#9` 91.3%;
- `#40` 91.0%;
- the best `knn_v1` trial, 89.5%.

## Official result (`lerobot-eval`, initial states 0–49, 2,000 episodes)

| Configuration | Spatial | Object | Goal | Long | Mean |
|---|---|---|---|---|---|
| one configuration (`v2_39`) | 88.0 | 99.8 | 97.6 | 81.4 | **91.7** |
| per-suite (Spatial uses `v2_9`) | 91.8 | 99.8 | 97.6 | 81.4 | 92.65 |

Re-runs of the single configuration: 360×360 renders (the `lerobot-eval` default) 91.5%; evaluation seed 2000 92.3%.
95% Wilson intervals are in `outputs/knn/final.json`.

Published LIBERO baselines (mean of 4 suites), for comparison:

| Model | Mean |
|---|---|
| Diffusion Policy | 72.4 |
| Octo | 75.1 |
| OpenVLA (7B) | 76.5 |
| Multi-task DiT (LeRobot docs) | 90.6 |
| π0 | 94.1 |
| OpenVLA-OFT | 97.1 |
| π0.5 (LeRobot) | 97.5 |

## Ablations (same 400 episodes, full method = 90.6%)

| Change | Success |
|---|---|
| prompt ignored | 63.0 (Goal suite 8%, where only the instruction differs between tasks) |
| raw 16×16 pixels instead of ResNet | 57.3 |
| k = 1 (replay the nearest demo) | 79.8 |
| 10% / 25% / 50% of demos | 77.8 / 81.5 / 86.5 |
| DINOv2-S instead of ResNet-18 | 83.0 |
| wrist camera only / agent view only | 85.5 / 83.0 |
| plain weighted mean (no mode seeking) | 89.8 |
| no state / no time feature | 90.5 / 90.5 |
| hard task filter | 91.3 |
| no history matching | 91.8 |
| no temporal ensembling | 92.5 |

k sweep: 1 → 79.8, 2 → 84.5, 4 → 89.0, 13 → 90.6.

**Reading.** The representation, the prompt and the amount of data do the work. The extras each move success by
about a point, which is within noise on 400 episodes.

## Other A/B tests (all within noise, not adopted)

- **Training features from decoded video vs simulator renders** (`codec_ab`, states 40–49):
  - YUV420-matched features 92.5 vs 91.5–92.3;
  - AV1-decoded features 88.8 vs 90.0.

  The measured feature gap is in `outputs/knn/codec_gap.json`.
- **Test-time augmentation** (2–4 shifted crops, `tta`, states 0–19): 91.4–91.9 vs 92.0.
- **Retry strategies after a detected failure on LIBERO-Long** (`retry_long*`): 78.0–83.5 vs 80.5 (states 0–19), and
  78.5–81.3 vs 80.8 (states 0–39). No strategy is reliably better.
- **Domain gap** (`outputs/knn/domain_gap.json`):
  - Live simulator frames sit farther from the memory than dataset frames sit from other demos: the median
    nearest-neighbour distance ratio is 1.68 for the agent view and 1.19 for the wrist camera.
  - The robot state is closer (ratio 0.59).

## When does the lookup fail?

On LIBERO-Long, 30 episodes were diagnosed step by step (`outputs/knn/diagnose_libero_10.json`):
- no successful episode ever strayed more than **3.9×** the typical distance from its nearest demo frame;
- all 4 failures went beyond **6.0×**, e.g. after a missed grasp left the scene in a state no demo visits.

A lookup cannot extrapolate beyond its memory.

## A diffusion transformer behaves like kNN

We trained LeRobot's multi-task DiT on the same demos with the documented recipe (CLIP ViT-B/16 + CLIP text, 8-layer
DiT), but at a reduced budget: 60k steps at batch 32, against the docs' 100k at batch 320. It reaches 68.8%
(68.0 / 79.4 / 67.2 / 60.6), so it is undertrained; the docs' 90.6% run is the fair diffusion reference.

We recorded all **789** chunks it generated from unseen initial states (`outputs/knn/memorization.json`). Median
squared difference of 8-step motion chunks:

| Comparison | Median |
|---|---|
| DiT chunk → nearest training chunk | 0.0016 |
| new human demo → nearest other demo | 0.0018 |
| DiT chunk → kNN answer at the same state | 0.0109 |
| kNN's own neighbours → kNN answer | 0.0095 |

Gripper agreement between the DiT and kNN: 83%.

**Reading.**
- Its outputs are not new: they are as close to some training chunk as a genuinely new demonstration is to the other
  demos.
- At a given state, the DiT is as far from the kNN answer as kNN's own neighbours are. So it outputs one of the
  behaviours demonstrated around that state.
