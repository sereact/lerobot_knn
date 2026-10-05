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
    - history matching (the previous observation against the frame before each candidate);
    - action-mode seeking (keep only the chunks that agree);
    - temporal ensembling.

  Every executed action stays a convex combination of demonstrated actions.

The final configuration is `outputs/knn/v2_39.json`: k = 13, chunk 8, 1 action per step, text weight 0.48,
state weight 0.034, mode bandwidth 0.35, temporal ensembling 0.42.

## Hyper-parameter search

Optuna studies (`experiments/search.py`) ran on initial states 40–49 (`outputs/knn/optuna.db`), 193 trials in total,
each over 400 episodes:

| Study | Search space | Trials |
|---|---|---|
| `knn_v1` | `v1` | 29 |
| `knn_v2` (the final configuration's study) | `v2` | 24 |
| `knn_v3_<suite>`, per suite | `v3` | 29–43 |

In `knn_v2`, the number of neighbours and *at most one per demonstration* explain 57% and 25% of the variance
(fANOVA); most other settings move success by about a point.

`experiments/finalize.py` re-scored the candidates on held-out initial states 0–19:
- `knn_v2#39` 91.6%;
- `#9` 91.3%;
- `#40` 91.0%;
- the best `knn_v1` trial, 89.5%.

## Official result (`lerobot-eval`, initial states 0–49, 2,000 episodes)

| Configuration | Spatial | Object | Goal | Long | Mean |
|---|---|---|---|---|---|
| one configuration (`v2_39`) | 88.0 | 99.8 | 97.6 | 81.4 | **91.7** |
| per-suite (Spatial uses `v2_9`) | 91.8 | 99.8 | 97.6 | 81.4 | 92.65 |
| `v2_39` with the learnt key (04) | 90.6 | 99.6 | 97.0 | 84.8 | 93.0 |

Re-runs of the single configuration: 360×360 renders (the `lerobot-eval` default) 91.5%; evaluation seed 2000 92.3%
(`outputs/knn/eval_official/`).

Published LIBERO baselines (mean of 4 suites), for context:

| Model | Mean |
|---|---|
| Diffusion Policy | 72.4 |
| OpenVLA (7B) | 76.5 |
| Multi-task DiT (LeRobot docs) | 90.6 |
| OpenVLA-OFT | 97.1 |
| π0.5 (LeRobot) | 97.5 |

## Ablations (official protocol, 2,000 episodes)

One change to the full method at a time (`experiments/ablations.py` wrote the configurations, `outputs/knn/ablations/`
and `outputs/knn/ksweep/`; `experiments/compare.py --init_states=0:50`, tags `full50_ablation` and `full50_ksweep`).

| Change | Mean | Spatial | Object | Goal | Long | Δ |
|---|---|---|---|---|---|---|
| none (full method, `lerobot-eval`) | 91.7 | 88.0 | 99.8 | 97.6 | 81.4 | |
| prompt ignored | 64.6 | 77.6 | 99.6 | 8.8 | 72.4 | −27.1 |
| raw 16×16 pixels | 57.15 | 79.0 | 0.0 | 91.0 | 58.6 | −34.6 |
| 10% / 25% / 50% of demos | 77.4 / 82.6 / 88.5 | | | | | −14.3 / −9.1 / −3.2 |
| k = 1 (replay the nearest demo) | 78.6 | 77.2 | 90.6 | 86.2 | 60.4 | −13.1 |
| DINOv2-S instead of ResNet-18 | 83.35 | 79.8 | 97.6 | 95.0 | 61.0 | −8.3 |
| agent-view camera only (no wrist) | 82.05 | 78.0 | 92.8 | 93.6 | 63.8 | −9.7 |
| wrist camera only (no agent view) | 87.45 | 87.0 | 98.8 | 92.8 | 71.2 | −4.2 |
| plain weighted mean (no mode seeking) | 90.7 | 88.4 | 100.0 | 96.6 | 77.8 | −1.0 |
| no temporal ensembling | 91.1 | 92.0 | 98.0 | 94.2 | 80.2 | −0.6 |
| no proprioceptive state / no time feature | 91.3 / 91.3 | | | | | −0.4 / −0.4 |
| no history matching | 91.5 | 89.6 | 98.8 | 97.6 | 79.8 | −0.2 |
| hard task filter | 91.9 | 88.4 | 100.0 | 97.4 | 81.8 | +0.2 |

k sweep: 1 → 78.6, 2 → 85.2, 4 → 89.35, 8 → 89.8, 13 (final) → 91.7, 24 → 92.65, 32 → 91.7.

**Reading.**
- The representation, the prompt and the amount of data do the work. The extras each move success by about a point,
  within the ±1.2-point 95% interval of 2,000 episodes.
- k = 24 scores slightly above the tuned k = 13, within that interval.
- Without the prompt the Goal suite collapses (8.8%): its tasks share one scene and differ only in the instruction.
- Raw pixels fail all of LIBERO-Object: at 16×16 its ten tasks share a scene of small objects.

The same ablations on the 400 development episodes (initial states 20–29, full method 90.6%) gave the same ordering
(prompt 63.0, pixels 57.3, k = 1 79.8, 10% of demos 77.8, DINOv2 83.0).

## Other A/B tests (all within noise, not adopted)

- **Retry after a detected failure on LIBERO-Long** (`retry_*` options, `outputs/knn/retry/`): 78.0–83.5 vs 80.5
  (states 0–19), and 78.5–81.3 vs 80.8 (states 0–39). No strategy is reliably better, so retry is off.
- **Test-time augmentation** (2–4 shifted crops, `query_tta`, `outputs/knn/tta/`, states 0–19): 91.4–91.9 vs 92.0.
- **A committee** of `v2_39` and `v2_9` averaging their actions: 92.0 on states 0–19, no better than `v2_39` alone.
- **Matching the dataset's video compression on live frames** (`codec_ab`, states 40–49): YUV420-matched features
  92.5 vs 91.5–92.3; AV1-decoded features 88.8 vs 90.0.
- **Domain gap.** Live simulator frames sit farther from the memory than dataset frames sit from other demos: the
  median nearest-neighbour distance ratio is 1.68 for the agent view and 1.19 for the wrist camera; the robot state is
  closer (0.59).

The code of the last two studies (the `query_transform` option, `codec_gap.py`, `domain_gap.py`) and of the committee
was removed in the repository cleanup; it is in git history (commit `b4acfae`).

## When does the lookup fail?

On LIBERO-Long, 30 episodes were diagnosed step by step (development run, `diagnose.py`, removed in the cleanup; in
commit `b4acfae`):
- no successful episode ever strayed more than **3.9×** the typical distance from its nearest demo frame;
- all 4 failures went beyond **6.0×**, e.g. after a missed grasp left the scene in a state no demo visits.

A lookup cannot extrapolate beyond its memory.

## Side study: a diffusion transformer behaves like kNN

Before the study moved to public VLA checkpoints, we trained LeRobot's multi-task DiT on the same demos with the
documented recipe at a reduced budget (60k steps at batch 32 against the docs' 100k at batch 320). It reached 68.8%,
so it was undertrained. On the 789 chunks it generated from unseen initial states, the median squared difference of
8-step motion chunks was:

| Comparison | Median |
|---|---|
| DiT chunk → nearest training chunk | 0.0016 |
| new human demo → nearest other demo | 0.0018 |
| DiT chunk → kNN answer at the same state | 0.0109 |
| kNN's own neighbours → kNN answer | 0.0095 |

Its outputs were as close to some training chunk as a new demonstration is to the other demos, and as far from the
kNN answer as kNN's own neighbours are. The same measurement on the nine VLAs (03, `head_vs_knn`) replaced it; the
DiT training script and `memorization.py` were removed in the cleanup and are in commit `b4acfae`.
