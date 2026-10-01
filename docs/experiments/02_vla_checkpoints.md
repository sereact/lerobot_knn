# 2. VLA checkpoints, and how hard LIBERO really is

The study uses **one working LIBERO fine-tune per VLA family**, all public Hub checkpoints (no fine-tuning on this
machine). Where possible the family's weights from *before* LIBERO fine-tuning are also used, for the
before/after backbone comparison in 03.

All runs used the same 400 episodes (initial states 20–29), `experiments/vla_rollouts.py` and 10 actions per
re-plan. Results are in `outputs/vla/results.jsonl` and, per episode, `results_episodes.jsonl`.

## The checkpoints

| Family | Size | Before LIBERO | After LIBERO (evaluated) | Spatial | Object | Goal | Long | **Ours** | Reported |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 | 3B | `lerobot/pi05_base` | `lerobot/pi05_libero_finetuned_v044` | 96 | 99 | 96 | 97 | **97.0** | 97.5 (LeRobot docs) |
| X-VLA | 0.9B | `lerobot/xvla-base` | `lerobot/xvla-libero` | 92 | 99 | 91 | 90 | **93.0** | 93 (LeRobot docs) |
| π0-FAST | 3B | `lerobot/pi0fast-base` | `lerobot/pi0fast-libero` | 92 | 95 | 92 | 80 | **89.8** | 82.5 (LeRobot docs) |
| SmolVLA | 0.45B | `lerobot/smolvla_base` | `lerobot/smolvla_libero` | 76 | 90 | 88 | 66 | **80.0** | 87.3 (SmolVLA paper) |
| π0 | 3B | `lerobot/pi0_base` | `lerobot/pi0_libero_finetuned_v044` | 75 | 87 | 89 | 48 | **74.8** | none from LeRobot (openpi's own π0: 94.2) |
| VLA-JEPA | 3B | `lerobot/VLA-JEPA-Pretrain` (DROID) | `lerobot/VLA-JEPA-LIBERO` | 94 | 100 | 99 | 93 | **96.5** | 96.5 (LeRobot docs) |
| Evo-1 | 0.8B | none (VLM: InternVL3-1B) | `zuoxingdong/evo1_libero` | 91 | 98 | 99 | 91 | **94.8** | 96.65 (LeRobot docs) |

Notes on running them:
- **X-VLA** needs its own LIBERO processors (a 6D-rotation → axis-angle action conversion) and **absolute**
  end-effector control. With the default relative control it scored 0 of 66.
- **π0-FAST** can emit an invalid FAST token sequence; such rows fall back to a hold-still chunk. It never happened
  in the final run.
- **SmolVLA** re-plans every 10 actions. Executing the full 50-step chunk gives 64.3%.
- **VLA-JEPA** re-plans every 7 actions (its LeRobot default).
- **Evo-1** renders at 448×448 and re-plans every 14 actions, as in its LeRobot evaluation. LIBERO-plus's fog noise
  assumes 256 px renders, so its LIBERO-plus run (and a matching clean run) use 256 px renders.
- **π0** is the weakest of the working checkpoints, mostly on LIBERO-Long (48%). It is the only LeRobot π0 LIBERO
  fine-tune that loads and runs correctly; `lerobot/pi0_libero` ships without normaliser statistics.

## Still running (phase X3)

| Family | Before LIBERO | After LIBERO | Reported LIBERO | Evaluation notes |
|---|---|---|---|---|
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | `nvidia/gr00t17-lerobot-libero_{spatial,object,goal,10}-640`, one per suite (12 GB each) | 96.5 (95 / 100 / 98 / 93, LeRobot docs) | per-suite checkpoints, merged into one entry; 8 of 16 actions per re-plan (NVIDIA's LIBERO wrapper) |
| MolmoAct2 | `allenai/MolmoAct2` | `allenai/MolmoAct2-LIBERO-LeRobot` | 98.25 (LeRobot docs) | wrist camera renamed, bf16, `train_mode_vlm="fft"` |

Every new checkpoint first passes `experiments/vla_sanity.py`. It runs the rollout code on 48 demo frames and requires
the predicted 8-step motion to be closer than half the error of a shuffled pairing.

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

Every VLA was compared episode by episode with the **plain lookup over ImageNet ResNet-18 features** (87.3%) on the
same 400 episodes (`episode_overlap` in `poster/data/poster_data.json`).

| VLA | Success | Successes the lookup also has | Only the VLA solves | Only the lookup solves | VLA success where the lookup fails |
|---|---|---|---|---|---|
| π0.5 | 97.0 | 88% | 45 | 6 | 88% |
| X-VLA | 93.0 | 88% | 43 | 20 | 84% |
| π0-FAST | 89.8 | 91% | 34 | 24 | 67% |
| SmolVLA | 80.0 | 92% | 26 | 55 | 51% |
| π0 | 74.8 | 94% | 19 | 69 | 37% |
| VLA-JEPA | 96.5 | 89% | 43 | 6 | 84% |
| Evo-1 | 94.8 | 89% | 41 | 11 | 80% |

Against the full kNN method (90.6–90.8% on these episodes), the overlap is 92–95%.

Per task:
- **31 of 40** tasks are solved in ≥90% of episodes by the plain ImageNet lookup (19 in all 10);
- with the full method, **35 of 40** (21 in all 10);
- the tasks that are hard for the lookup are three LIBERO-Long tasks (task ids 6, 4 and 8, plain lookup 30–40%)
  and Spatial task 5.

**Reading.**
- Most of what any VLA achieves on LIBERO is also achieved by retrieval with generic features.
- Strong and weak VLAs differ on the episodes retrieval fails: π0.5 still solves 88% of them, π0 37%.
- The weak ones also lose many episodes the lookup solves (SmolVLA 55, π0 69).
- A lookup over generic features is therefore a cheap measure of a benchmark's difficulty, and of how much of a
  reported success is plain retrieval of the demonstrations.
