# 5. Off the demonstrations: LIBERO-plus

[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) rebuilds the four LIBERO suites as about 10k task variants in
seven perturbation categories:
- camera viewpoint;
- robot initial state;
- language (LLM-rewritten instructions);
- light;
- background texture;
- sensor noise (motion, Gaussian, zoom, fog and glass blur, applied to the agent view);
- object layout.

The benchmark protocol is one trial per variant, from its first initial state. Every policy was trained on plain
LIBERO only.

## Setup

- **Separate venv.** `outputs/libero_plus/venv` is built by `outputs/libero_plus/setup.sh`: the fork at commit
  `4976dc3`, robosuite 1.4.1, mujoco 3.7.0, and 9.5 GB of assets. Torch, transformers and lerobot versions are
  identical to the main venv.
- **Every command needs** `PYTHONPATH=outputs/libero_plus/LIBERO-plus` (the fork is a namespace package),
  `LIBERO_CONFIG_PATH=outputs/libero_plus/libero_config` and `MUJOCO_GL=egl`.
- **Environment.** LeRobot's `LiberoEnv(is_libero_plus=True)`, driven by the same pool (`LiberoPool(libero_plus=True)`).
  X-VLA gets absolute end-effector control through `LiberoPool.set_control_mode`.
- **Sample.** `outputs/libero_plus/sample.json`: 15 random variants per (suite × category), seed 0. That gives
  **420 variants**, 60 per category, the same for every policy. The evaluator is `experiments/libero_plus_eval.py`.
- **Checks before the runs.**
  - Every category builds, resets and steps.
  - The camera, light and background variants change the agent-view image a lot.
  - Robot-start-pose variants move the arm by up to 0.32 rad per joint and 5.8 cm at the end effector (level 5), so
    they are not overwritten by the initial-state file.
- **Fork patch.** The fork used `np.float_`, which NumPy 2 removed, so every fog variant crashed. We patched it to
  `np.float64` in our local copy and tested all five noise functions. Glass blur costs 0.5–0.8 s per step, which is
  slow but fine.
- **Language variants.** The kNN maps an unseen instruction to the nearest training prompt (CLIP text). That picks
  the right task for 81.5% of all LIBERO-plus rewrites and 85% of those in our sample; on Spatial, where tasks differ
  by one spatial phrase, only 61% (`outputs/libero_plus/language_mapping.json`).

## Results

LIBERO-plus success on the 420 variants, and per perturbation (60 variants each), next to clean LIBERO on the official
protocol (2,000 episodes). Rows are ordered by LIBERO-plus (`outputs/libero_plus/results.jsonl`).

| Policy | clean LIBERO | LIBERO-plus | drop | Camera | Robot | Language | Light | Background | Noise | Layout |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5 | 97.7 | **85.5** | −12.2 | 68 | 78 | 92 | **100** | 93 | 83 | **83** |
| MolmoAct2 | 97.5 | 82.6 | −14.9 | 73 | 67 | 87 | 95 | **100** | 75 | 82 |
| kNN, full method | 91.7 | 82.4 | **−9.3** | 73 | 58 | **98** | 92 | 97 | **92** | 67 |
| kNN, full method + learnt key | 93.0 | 81.7 | −11.3 | **88** | 58 | 90 | 90 | 82 | 88 | 75 |
| GR00T N1.7 | 94.8 | 80.2 | −14.6 | 63 | 63 | 87 | 95 | 97 | 85 | 72 |
| kNN, plain (ImageNet ResNet-18) | 88.7 | 75.5 | −13.2 | 68 | 48 | 77 | 92 | 85 | 88 | 70 |
| VLA-JEPA | 97.2 | 74.0 | −23.1 | 55 | 58 | 88 | 92 | 93 | 55 | 77 |
| Evo-1 | 95.4 | 71.4 | −24.0 | 53 | 48 | 82 | 88 | 92 | 73 | 63 |
| X-VLA | 94.5 | 64.5 | −30.0 | 7 | **83** | 77 | 93 | 93 | 42 | 57 |
| π0 | 71.9 | 42.9 | −29.0 | 67 | 27 | 12 | 43 | 58 | 52 | 42 |
| SmolVLA | 82.2 | 27.9 | −54.3 | 25 | 13 | 22 | 43 | 45 | 10 | 37 |
| π0-FAST | 90.5 | 25.7 | −64.8 | 10 | 15 | 33 | 43 | 37 | 0 | 42 |

With 60 variants per category the standard error is about 5–6 pp, so treat single cells cautiously; the overall
numbers (420 episodes) are good to about ±3.5 pp (95%). Bold: best per column.

## Observations

- **The lookup degrades least of all policies** (−9.3 points, against −12.2 for π0.5, the most robust VLA, and up to
  −64.8). It ranks 3rd of 10 and is the best policy on two of the four suites (Object and Goal). Part of this is a
  favourable benchmark: LIBERO-plus scenes need the same actions as the clean demonstrations (its own demos replay
  them), so the right answer is in memory whenever the key ignores the perturbation.
- **π0.5 is ahead where the scene itself changes:** robot start pose 78 vs 58, layout 83 vs 67, light 100 vs 92.
- **The lookup is ahead where only appearance or wording changes:** camera 73 vs 68, noise 92 vs 83, language 98 vs
  92, background 97 vs 93. The wrist camera (unaffected by camera and noise perturbations), the robot state and the
  prompt's nearest task anchor it. Changed robot initial states and object layouts hurt the lookup most.
- **X-VLA collapses under camera changes (7%) and sensor noise (42%)** while staying strong on robot start pose (83).
- **SmolVLA and π0-FAST collapse in every category** (27.9% and 25.7%), including the language-only variants (22% and
  33%), whose images are unperturbed. π0-FAST gets 0 of 60 under sensor noise.
- **Environment control (phase N2): the collapses are real.** The language variants keep LIBERO's original scene, so
  with their *original* instruction they are clean LIBERO inside the LIBERO-plus environment (robosuite 1.4.1 /
  mujoco 3.7.0 instead of 1.4.0 / 3.8.1). Every policy scores near its clean LIBERO success there (60 episodes each,
  rows tagged `libero_plus_original_instruction`):

  | Policy | clean LIBERO | control (original instruction) | same scenes, reworded instruction |
  |---|---|---|---|
  | π0.5 | 97.7 | 100.0 | 92 |
  | X-VLA | 94.5 | 91.7 | 77 |
  | π0-FAST | 90.5 | 88.3 | 33 |
  | kNN, plain | 88.7 | 91.7 | 77 |
  | SmolVLA | 82.2 | 81.7 | 22 |
  | π0 | 71.9 | 68.3 | 12 |

- **The weaker VLAs key on the wording, not the meaning.** On identical scenes, rewording the instruction takes
  SmolVLA from 82% to 22%, π0-FAST from 88% to 33% and π0 from 68% to 12%. π0.5 (100 → 92) and the full kNN (98% on
  reworded instructions, thanks to its soft text distance plus visual retrieval) barely notice.
- **Not comparable with the published leaderboard.** That leaderboard (e.g. π0 53.6, π0-FAST 61.6, OpenVLA-OFT 69.6)
  used openpi / OpenVLA checkpoints and the authors' own pipeline. Ours are LeRobot checkpoints in LeRobot's LIBERO
  environment on a 420-variant sample.

## How hard is LIBERO-plus? Episode overlap with a training-free lookup

Same 420 variants, each VLA against the plain ImageNet lookup (75.5%):

| VLA | Success | Successes the lookup also has | Only the VLA solves | Only the lookup solves | VLA success where the lookup fails |
|---|---|---|---|---|---|
| π0.5 | 85.5 | 80% | 72 | 30 | 70% |
| MolmoAct2 | 82.6 | 79% | 72 | 42 | 70% |
| GR00T N1.7 | 80.2 | 82% | 59 | 39 | 57% |
| VLA-JEPA | 74.0 | 81% | 59 | 65 | 57% |
| Evo-1 | 71.4 | 82% | 54 | 71 | 52% |
| X-VLA | 64.5 | 77% | 61 | 107 | 59% |
| π0 | 42.9 | 87% | 23 | 160 | 22% |
| SmolVLA | 27.9 | 85% | 18 | 218 | 17% |
| π0-FAST | 25.7 | 89% | 12 | 221 | 12% |

- Against the full kNN, π0.5 wins 45 variants and loses 32 (paired sign test p = 0.17; tied); MolmoAct2 wins 46 and
  loses 45. Every other VLA loses more variants to the full kNN than it wins.
- The weak VLAs' successes are almost a subset of the lookup's.

## Lookups keyed on each VLA's backbone, under perturbation

Plain lookups on LIBERO-plus, by key (clean LIBERO, official protocol, in brackets). The before/after-fine-tuning
table with the VLAs themselves is in 03.

| Key | LIBERO-plus |
|---|---|
| MolmoAct2 LIBERO L27 | 81.0 (91.3) |
| X-VLA LIBERO L8 | 78.1 (88.5) |
| ResNet-18, learnt key | 76.7 (93.0) |
| ResNet-18 | 75.5 (88.7) |
| π0-FAST LIBERO L12 | 74.0 (95.0) |
| Evo-1 LIBERO L10 | 72.6 (87.1) |
| SmolVLA LIBERO | 69.3 (86.5) |
| GR00T N1.7 frozen VLM L4 | 66.4 (87.5) |
| π0.5 LIBERO L12 / VLA-JEPA LIBERO L7 | 65.7 (81.8 / 83.0) |
| X-VLA base L8 | 62.4 (83.3) |
| π0.5 base L12 | 60.5 (85.4) |
| π0-FAST base L12 | 58.3 (86.6) |
| π0 base L12 | 56.7 (86.9) |
| MolmoAct2 base L27 | 52.6 (78.6) |
| VLA-JEPA base L7 | 49.3 (84.0) |
| SmolVLA base | 45.0 (76.3) |
| DINOv2 | 43.3 (73.1) |
| Evo-1 base L10 | 35.5 (62.1) |
| π0 LIBERO L12 | 2.4 (17.9) |

Reading:
- A VLM key is less robust than the ImageNet key before fine-tuning: every base VLM key drops 21–35 points from clean
  LIBERO, against 13 for ResNet-18. Sensor noise and camera changes hurt them most.
- LIBERO fine-tuning makes the keys more robust for 7 of 8 families (+5 to +37 points) and breaks π0's (56.7 → 2.4).
- A lookup on SmolVLA's own fine-tuned VLM is 2.5× as robust as SmolVLA itself; on π0-FAST's, 2.9×.

## Does the VLA still act like a lookup off the demonstrations? (phase T, `head_vs_knn`)

Recorded LIBERO-plus rollouts of each VLA (84 variants each), and R² of a plain lookup in the VLA's own key space (and
on ResNet-18) predicting its chunks (`outputs/libero_plus/head_vs_knn_*.json`):

| VLA | success of the recorded rollouts | R², own key space | R², ResNet-18 |
|---|---|---|---|
| π0.5 | 74% | 0.85 | 0.80 |
| VLA-JEPA | 57% | 0.68 | 0.59 |
| π0 (base key) | 30% | 0.20 | 0.22 |
| SmolVLA | 15% | 0.08 | −0.14 |
| π0-FAST | 16% | −0.01 | −0.01 |

X-VLA predicts absolute poses, so its R² (−0.72) is not comparable. **The robust VLAs stay lookup-like off the
demonstrations; the collapsing ones stop resembling anything their own neighbours did.**

## Offline: LIBERO-plus demos as queries (phase U, `plus_neighbor_metric`)

LIBERO-plus demos replay clean demos' exact actions in perturbed scenes, so every perturbed frame has a clean twin.
**Invariance**: does the key's nearest memory frame land on that twin? **LAC** among the other demos. 1,600 perturbed
frames from 644 demos; no checkpoint saw them, so fine-tuned keys are not flattered by memorisation
(`outputs/libero_plus/neighbor_metric_plus.json`).

| Key | finds its clean twin | LAC@8 (other demos) | closed-loop LIBERO-plus |
|---|---|---|---|
| ResNet-18 | 0.86 | 0.759 | 75.5 |
| ResNet-18, learnt key | 0.81 | 0.796 | 76.7 |
| X-VLA LIBERO / base | 0.80 / 0.66 | 0.836 / 0.741 | 78.1 / 62.4 |
| π0.5 LIBERO / base L12 | 0.74 / 0.57 | 0.815 / 0.771 | 65.7 / 60.5 |
| π0-FAST LIBERO / base | 0.69 / 0.58 | 0.831 / 0.772 | 74.0 / 58.3 |
| Evo-1 LIBERO / base | 0.67 / 0.50 | 0.847 / 0.681 | 72.6 / 35.5 |
| GR00T N1.7 frozen VLM | 0.67 | 0.729 | 66.4 |
| VLA-JEPA LIBERO / base | 0.66 / 0.52 | 0.784 / 0.706 | 65.7 / 49.3 |
| SmolVLA LIBERO / base | 0.65 / 0.55 | 0.822 / 0.679 | 69.3 / 45.0 |
| DINOv2 | 0.58 | 0.658 | 43.3 |
| π0 base / LIBERO L12 | 0.52 / 0.14 | 0.738 / 0.239 | 56.7 / 2.4 |

Over all 18 keys, **including the fine-tuned ones**, the offline metrics predict LIBERO-plus closed-loop success:
Spearman 0.83 for LAC@8 and 0.91 for the twin hit rate (Pearson 0.92 and 0.93). MolmoAct2's keys were not run.

## Rewording: backbone vs action expert (phase U, `plus_language`, 400 clean frames)

Clean demo frames of each task, paired with the original and with a random rewording of their instruction
(`outputs/libero_plus/language_sensitivity.json`).

| Model | backbone key shift | lookup task accuracy, original → reworded (no task filter) | action change under rewording (fraction of a task change) |
|---|---|---|---|
| ResNet-18 (text-free control) | 0 | 0.94 → 0.94 | – |
| SmolVLA | 0.0001 | 0.93 → 0.93 | **0.41** |
| π0.5 | 0.14 (base VLM: 1.20) | 1.00 → 0.95 | 0.08 |
| π0 | 2.18 (base VLM: 1.09) | 1.00 → **0.49** | 0.24 |
| π0-FAST | 0.59 (base VLM: 1.25) | 1.00 → 0.98 | **0.86** |
| VLA-JEPA | 0 | 0.87 → 0.87 | 0.04 |
| X-VLA | 0.10 (base: 0.16) | 1.00 → 0.95 | – |
| GR00T N1.7, Evo-1 | 0 | 0.94 → 0.94 | – |

Reading:
- SmolVLA's image tokens never see the text (nor VLA-JEPA's, GR00T's or Evo-1's at the tapped layer), so SmolVLA's
  sensitivity to wording sits entirely in the action expert.
- π0's fine-tuned backbone itself keys on the wording: a reworded prompt sends its key to the wrong task half the time.
- π0-FAST's backbone barely moves, but its actions change almost as much as for a different task: the wording
  sensitivity is in its action decoding.
- π0.5 is robust in both parts; its LIBERO fine-tune even made its key less sensitive to wording (1.20 → 0.14).
