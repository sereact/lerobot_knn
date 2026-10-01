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

## Results so far

| Policy | clean LIBERO* | LIBERO-plus | drop | Camera | Robot | Language | Light | Background | Noise | Layout |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5 (LeRobot) | 97.0 | **85.5** | −11.5 | 68 | 78 | 92 | 100 | 93 | 83 | 83 |
| kNN, full method | 90.6 | **82.4** | −8.2 | 73 | 58 | 98 | 92 | 97 | 92 | 67 |
| kNN, plain (ImageNet ResNet-18) | 87.3 | 75.5 | −11.8 | 68 | 48 | 77 | 92 | 85 | 88 | 70 |
| X-VLA | 93.0 | 64.5 | −28.5 | **7** | 83 | 77 | 93 | 93 | 42 | 57 |
| SmolVLA | 80.0 | 27.9 | −52.1 | 25 | 13 | 22 | 43 | 45 | 10 | 37 |
| π0-FAST | 89.8 | 25.7 | −64.1 | 10 | 15 | 33 | 43 | 37 | 0 | 42 |
| π0 | 74.8 | 42.9 | −31.9 | 67 | 27 | 12 | 43 | 58 | 52 | 42 |
| kNN, plain (π0.5 base VLM, layer 12) | 88.5 | 60.5 | −28.0 | 45 | 57 | 73 | 78 | 67 | 37 | 67 |

\*The same 400 clean LIBERO episodes as everywhere else. With 60 variants per category the standard error is about
5–6 pp, so treat single cells cautiously; the overall numbers (420 episodes) are good to about ±2 pp.

## Observations so far

- **The perturbations cost π0.5 about as much as they cost a training-free lookup** (−11.5 vs −11.8 for the plain
  lookup and −8.2 for the full method). π0.5's lead over the full kNN shrinks from 6.4 to 3.1 points.
- **π0.5 is ahead where the scene itself changes:** robot start pose 78 vs 58, layout 83 vs 67, light 100 vs 92.
- **The lookup is ahead where only appearance or wording changes:** camera 73 vs 68, noise 92 vs 83, language 98 vs
  92, background 97 vs 93. The wrist camera (unaffected by camera and noise perturbations), the robot state and the
  prompt's nearest task anchor it.
- **A VLM key is much less robust than the ImageNet key.** The plain lookup on π0.5's base VLM drops 28 points
  (88.5 → 60.5), against 11.8 for the same lookup on ResNet-18. Sensor noise (37%) and camera (45%) hurt it most.
- **X-VLA collapses under camera changes (7%) and sensor noise (42%)** while staying strong on robot start pose (83).
  It looks far more dependent on the exact agent-view image than the others.
- **SmolVLA and π0-FAST collapse in every category** (27.9% and 25.7%), including the language-only variants (22% and
  33%), whose images are unperturbed. π0-FAST gets 0 of 60 under sensor noise.
- **Environment control (phase N2): the collapses are real.** The language variants keep LIBERO's original scene, so
  with their *original* instruction they are clean LIBERO inside the LIBERO-plus environment (robosuite 1.4.1 /
  mujoco 3.7.0 instead of 1.4.0 / 3.8.1). Every policy scores near its clean LIBERO success there (60 episodes each,
  rows tagged `libero_plus_original_instruction`):

  | Policy | clean LIBERO | control (original instruction) | same scenes, reworded instruction |
  |---|---|---|---|
  | π0.5 | 97.0 | 100.0 | 92 |
  | X-VLA | 93.0 | 91.7 | 77 |
  | π0-FAST | 89.8 | 88.3 | 33 |
  | kNN, plain | 87.3 | 91.7 | 77 |
  | SmolVLA | 80.0 | 81.7 | 22 |
  | π0 | 74.8 | 68.3 | 12 |

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
| X-VLA | 64.5 | 77% | 61 | 107 | 59% |
| π0 | 42.9 | 87% | 23 | 160 | 22% |
| SmolVLA | 27.9 | 85% | 18 | 218 | 17% |
| π0-FAST | 25.7 | 89% | 12 | 221 | 12% |

- Against the full kNN, π0.5's edge shrinks to 45 variants won vs 32 lost (paired sign test p = 0.17; tied).
- The weak VLAs' successes are almost a subset of the lookup's.

## Analyses of VLA behaviour on LIBERO-plus (no LIBERO-plus training; LIBERO checkpoints and the clean memory)

| Analysis | What it measures | Phase |
|---|---|---|
| Lookups keyed on each VLA's own VLM (SmolVLA, π0.5, π0; before and after LIBERO fine-tuning), the learnt key, DINOv2 | the backbone-vs-action-expert plane under perturbations; which keys are robust | S |
| Recorded VLA rollouts (84 variants each) → `head_vs_knn` | per perturbation and outcome: does a lookup in the VLA's own space still reproduce its chunk; how far observations stray from the nearest clean demo frame; are its chunks still copies of training chunks when it fails (replaying the wrong demo)? | T |
| `plus_neighbor_metric` | LIBERO-plus demos replay clean demos' exact actions in perturbed scenes, so every perturbed frame has a clean twin. **Invariance**: does the key's nearest memory frame land on that twin? **LAC** among the other demos; correlation with LIBERO-plus closed-loop success | U |
| `plus_language` | original vs reworded instruction on the same clean frames. **Backbone**: key shift and task accuracy of a lookup without task filter. **Action expert**: chunk change relative to its sampling noise and to a real task change | U |

First results:
- **Invariance (1,600 perturbed frames from 644 demos).** ResNet-18 finds the clean twin for 86% of perturbed frames,
  DINOv2 for 58%. Leave-source-out LAC@8 is 0.759 and 0.658 (clean held-out: 0.790 and 0.756).
- **Language test (80 frames).** SmolVLA's VLM key does not move at all when the instruction is reworded (image
  tokens do not attend to the text). Its action expert's chunk still changes a third as much as for a real task
  change (rewording effect 0.33). The wording sensitivity lives in the action expert.

## Results of the LIBERO-plus behaviour analyses (2026-10-01)

### Lookups keyed on each VLA's backbone (phase S): the diagnostic plane under perturbations

| VLA | clean: VLA / lookup on its own VLM | LIBERO-plus: VLA / lookup on its own VLM | LIBERO-plus action-expert gain |
|---|---|---|---|
| π0.5 | 97.0 / 84.5 | 85.5 / 65.7 | **+19.8** |
| SmolVLA | 80.0 / 85.3 | 27.9 / 69.3 | **−41.4** |
| π0 | 74.8 / 15.0 (base VLM: 86.0) | 42.9 / 2.4 (base VLM: 56.7) | +40.5 over its own VLM; −13.8 vs its base VLM |
| VLA-JEPA (new) | 96.5 / – | 74.0 / – | – |

Plain lookups on LIBERO-plus, by key (clean LIBERO in brackets):

| Key | LIBERO-plus |
|---|---|
| ResNet-18, learnt key | 76.7 (90.5) |
| ResNet-18 | 75.5 (87.3) |
| SmolVLA LIBERO | 69.3 (85.3) |
| π0.5 LIBERO L12 | 65.7 (84.5) |
| π0.5 base L12 | 60.5 (88.5) |
| π0 base L12 | 56.7 (86.0) |
| SmolVLA base | 45.0 (74.5) |
| DINOv2 | 43.3 (74.5) |
| π0 LIBERO L12 | 2.4 (15.0) |

Reading:
- A lookup on SmolVLA's own VLM is 2.5× as robust as SmolVLA itself. LIBERO fine-tuning made SmolVLA's and π0.5's
  keys more robust (SmolVLA 45.0 → 69.3, π0.5 60.5 → 65.7) and broke π0's (56.7 → 2.4).
- **π0's broken key is real at every depth** (phase P2, 10-task pilot, held-out action R²): base 0.73–0.80 at every
  tap from the vision tokens to the last layer; LIBERO fine-tune 0.39–0.48 at every tap.

### Offline: LIBERO-plus demos as queries (phase U, `plus_neighbor_metric`)

| Key | finds its clean twin | LAC@8 (other demos) | closed-loop LIBERO-plus |
|---|---|---|---|
| ResNet-18 | 0.86 | 0.759 | 75.5 |
| ResNet-18, learnt key | 0.81 | 0.796 | 76.7 |
| SmolVLA LIBERO | 0.65 | 0.822 | 69.3 |
| π0.5 LIBERO L12 | 0.74 | 0.815 | 65.7 |
| π0.5 base L12 | 0.57 | 0.771 | 60.5 |
| π0 base L12 | 0.52 | 0.738 | 56.7 |
| SmolVLA base | 0.55 | 0.679 | 45.0 |
| DINOv2 | 0.58 | 0.658 | 43.3 |
| π0 LIBERO L12 | 0.14 | 0.239 | 2.4 |

Over all 9 keys, **including the fine-tuned ones** (no memorisation: none saw these demos), the offline metrics predict
LIBERO-plus closed-loop success: Spearman 0.82 for LAC@8 and 0.85 for the twin hit rate (Pearson 0.95 for both).

### Rewording: backbone vs action expert (phase U, `plus_language`, 400 clean frames)

| Model | backbone key shift | lookup task accuracy, original → reworded (no task filter) | action change under rewording (fraction of a task change) |
|---|---|---|---|
| ResNet-18 (text-free control) | 0 | 0.94 → 0.94 | – |
| SmolVLA | 0.0001 | 0.93 → 0.93 | **0.41** |
| π0.5 | 0.14 (base VLM: 1.20) | 1.00 → 0.95 | 0.08 |
| π0 | 2.18 (base VLM: 1.09) | 1.00 → **0.49** | 0.24 |
| π0-FAST | – | – | **0.86** |

Reading:
- SmolVLA's image tokens never see the text, so its sensitivity to wording sits entirely in the action expert.
- π0's fine-tuned backbone itself keys on the wording: a reworded prompt sends its key to the wrong task half the time.
- π0-FAST's actions change almost as much as for a different task.
- π0.5 is robust in both parts; its LIBERO fine-tune even made its key less sensitive to wording (1.20 → 0.14).
