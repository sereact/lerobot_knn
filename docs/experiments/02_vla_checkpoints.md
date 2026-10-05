# 2. VLA checkpoints, and how hard LIBERO really is

The study uses **one working LIBERO fine-tune per VLA family**, all public Hub checkpoints evaluated as released (no
fine-tuning on this machine). Where possible the family's weights from *before* LIBERO fine-tuning are also used, for
the before/after backbone comparison in 03.

Final numbers use the official protocol: initial states 0–49 of every task (2,000 episodes), run with
`experiments/vla_rollouts.py --init_states=0:50 --tag=full50`. Results are in `outputs/vla/results.jsonl` and, per
episode, `results_episodes.jsonl`. Development runs used the 400 episodes of initial states 20–29.

## The checkpoints

| Family | Size | Before LIBERO | After LIBERO (evaluated) | Spatial | Object | Goal | Long | **Ours** | Reported |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 | 3B | `lerobot/pi05_base` | `lerobot/pi05_libero_finetuned_v044` | 97.0 | 99.8 | 97.8 | 96.2 | **97.7** | 97.5 (LeRobot docs) |
| MolmoAct2 | | `allenai/MolmoAct2` | `allenai/MolmoAct2-LIBERO-LeRobot` | 97.4 | 99.4 | 97.6 | 95.6 | **97.5** | 98.25 (LeRobot docs) |
| VLA-JEPA | 3B | `lerobot/VLA-JEPA-Pretrain` (DROID) | `lerobot/VLA-JEPA-LIBERO` | 95.4 | 100.0 | 97.4 | 95.8 | **97.2** | 96.5 (LeRobot docs) |
| Evo-1 | 0.8B | its VLM as published: `OpenGVLab/InternVL3-1B-hf` (first 14 layers) | `zuoxingdong/evo1_libero` | 92.0 | 98.6 | 97.4 | 93.6 | **95.4** (256 px renders) | 96.65 (LeRobot docs) |
| GR00T N1.7 | 3.1B (VLM 1.5B + head 1.6B) | `nvidia/GR00T-N1.7-3B` (the LIBERO fine-tunes froze its VLM) | `nvidia/gr00t17-lerobot-libero_{spatial,object,goal,10}-640`, one per suite | 95.8 | 98.2 | 94.6 | 90.8 | **94.8** | 96.5 (95 / 100 / 98 / 93, LeRobot docs) |
| X-VLA | 0.9B | `lerobot/xvla-base` | `lerobot/xvla-libero` | 94.6 | 98.6 | 92.6 | 92.2 | **94.5** | 93 (LeRobot docs) |
| π0-FAST | 3B | `lerobot/pi0fast-base` | `lerobot/pi0fast-libero` | 90.4 | 93.8 | 90.6 | 87.4 | **90.5** | 82.5 (LeRobot docs) |
| SmolVLA | 0.45B | `lerobot/smolvla_base` | `lerobot/smolvla_libero` | 78.8 | 92.2 | 88.0 | 69.6 | **82.2** | 87.3 (SmolVLA paper) |
| π0 | 3B | `lerobot/pi0_base` | `lerobot/pi0_libero_finetuned_v044` | 70.0 | 86.4 | 85.8 | 45.2 | **71.9** | none from LeRobot (openpi's own π0: 94.2) |

Notes on running them:
- **X-VLA** needs its own LIBERO processors (a 6D-rotation → axis-angle action conversion) and **absolute**
  end-effector control. With the default relative control it scored 0 of 66.
- **π0-FAST** can emit an invalid FAST token sequence; such rows fall back to a hold-still chunk. It never happened
  in the final runs.
- **SmolVLA** re-plans every 10 actions. Executing the full 50-step chunk gives 64.3% (development episodes).
- **VLA-JEPA** re-plans every 7 actions (its LeRobot default).
- **Evo-1** re-plans every 14 actions. Its LeRobot evaluation renders at 448×448, but LIBERO-plus's fog noise assumes
  256 px renders, so the study uses 256 px renders on both benchmarks (development episodes: 96.3% at 256 px, 94.8% at
  448 px). Its LIBERO checkpoint fully fine-tuned the VLM (stage 2, `finetune_vlm=True`); "before" is that VLM as
  published.
- **MolmoAct2**: the wrist camera is renamed to `wrist_image`, bf16, `train_mode_vlm="fft"` (a full fine-tune).
  The policy builds its network from the HF checkpoint in `checkpoint_path` (`allenai/MolmoAct2-LIBERO`), then loads
  the LeRobot weights.
- **GR00T N1.7**: one checkpoint per suite, merged into one entry; 8 of 16 actions run per re-plan, as in NVIDIA's
  LIBERO wrapper. Loading pitfall: see below.
- **X-VLA** sees observations in its own LIBERO convention: only the agent view rotated, and a 20-D state with a 6-D
  rotation (`vla.XVLAFromDatasetFormat`). From 2026-10-01 11:24 to 2026-10-02 the standard convention was used by
  mistake. Its closed loop dropped to 0% and its backbone keys saw a rotated wrist view. The rows from that window
  are in `outputs/*/invalid_rows.jsonl` and were redone (phase AA).
- **π0** is the weakest of the working checkpoints, mostly on LIBERO-Long (45%). It is the only LeRobot π0 LIBERO
  fine-tune that loads and runs correctly; `lerobot/pi0_libero` ships without normaliser statistics.

## Checks and loading pitfalls

Every new checkpoint first passes `experiments/vla_sanity.py`. It runs the rollout code on 48 demo frames and requires
the predicted 8-step motion to be closer than half the error of a shuffled pairing. X-VLA predicts absolute poses,
so the check doesn't apply to it.

**GR00T loading.** The LIBERO checkpoints are LeRobot checkpoints: weights plus serialized processors that carry the
normalisation statistics.
- Building the policy the raw-checkpoint way (`--policy.base_model_path=<repo>`) initialises the network randomly
  from that repo without an error. It failed the sanity check: MSE 0.51 vs 0.51 shuffled, gripper sign 49%.
- They now load like every other LeRobot checkpoint: `from_pretrained` plus the saved processors. The saved
  `base_model_path` is a path on NVIDIA's machine, so it is pointed at the repo itself.
- The Spatial checkpoint then reproduces its demos: MSE 0.0013 vs 0.198 shuffled, gripper sign 98%.

Dependencies: `uv sync --extra vla_jepa` (and `groot`, `molmoact2`, `evo1`). The GR00T backbone
`nvidia/Cosmos-Reason2-2B` is gated.

## How hard is LIBERO? Episode-level overlap with a training-free lookup

Every VLA was compared episode by episode with the **plain lookup over ImageNet ResNet-18 features** (88.7%) on the
official 2,000 episodes (per-episode results in `poster/data/poster_data.json`, `libero_full`).

| VLA | Success | Successes the lookup also has | Only the VLA solves | Only the lookup solves | VLA success where the lookup fails |
|---|---|---|---|---|---|
| π0.5 | 97.7 | 89% | 213 | 33 | 94% |
| MolmoAct2 | 97.5 | 90% | 202 | 26 | 89% |
| VLA-JEPA | 97.2 | 90% | 201 | 32 | 89% |
| Evo-1 | 95.4 | 90% | 191 | 57 | 85% |
| GR00T N1.7 | 94.8 | 90% | 193 | 70 | 85% |
| X-VLA | 94.5 | 89% | 208 | 92 | 92% |
| π0-FAST | 90.5 | 90% | 184 | 147 | 81% |
| SmolVLA | 82.2 | 91% | 140 | 271 | 62% |
| π0 | 71.9 | 93% | 95 | 432 | 42% |

Against the full kNN method (91.7%), the overlap is 92–96%.

Failures coincide: when a VLA fails an episode, the full lookup fails the same episode in 20–46% of cases, against 8%
overall (2.4–5.5× chance). Per-task success of each VLA correlates with the lookup's (median Pearson r = 0.74 over the
nine VLAs; the task map in the paper's appendix shows the same tasks are hard for every policy).

Per task:
- **28 of 40** tasks are solved in ≥90% of episodes by the plain ImageNet lookup (8 in all 50);
- with the full method, **30 of 40** (16 in all 50);
- the hardest tasks for the lookup are LIBERO-Long tasks 0, 6 and 8 and Spatial task 5 (full method: Spatial 5 32%,
  Long 6 56%).

**Reading.**
- Most of what any VLA achieves on LIBERO is also achieved by retrieval with generic features.
- Strong and weak VLAs differ on the episodes retrieval fails: π0.5 still solves 94% of them, π0 42%.
- The weak ones also lose many episodes the lookup solves (SmolVLA 271, π0 432).
- A lookup over generic features is therefore a cheap measure of a benchmark's difficulty, and of how much of a
  reported success is plain retrieval of the demonstrations.
