# "Your VLA is Secretly kNN": experiment notes

The notes cover the experiments behind the `lerobot_policy_knn` plugin and the workshop poster
(`poster/`): what was run, how, what came out, and what is still running or planned.
Numbers are copied from the result files named in each section. The poster is regenerated from those same files
by `make` in the poster directory.

## The question

Is a vision-language-action model (VLA) effectively an **embedding** (its VLM, the *backbone*) plus a
**nearest-neighbour lookup** over the demonstrations it was trained on? A kNN lookup needs no training, so it
also works as a **probe**:

1. **How hard is a task or benchmark?** Measure what a lookup over generic image features already solves, and on
   which episodes it fails.
2. **How good is a VLA's backbone?** Run the same lookup keyed on the VLA's own VLM, in closed loop. Offline, measure
   how often close embeddings have close actions (the *local action consistency*, LAC).
3. **How good is its action expert?** Measure what the VLA achieves beyond a lookup on its own VLM.

## Files

| File | Contents |
|---|---|
| [01_knn_policy.md](01_knn_policy.md) | The kNN policy, hyper-parameter search, official LIBERO result, ablations, failure analysis, DiT comparison |
| [02_vla_checkpoints.md](02_vla_checkpoints.md) | The five VLAs (one working LIBERO fine-tune per family) and candidates to add; episode-level overlap with the lookup (task hardness) |
| [03_backbone_vs_action_expert.md](03_backbone_vs_action_expert.md) | VLM hidden states as lookup keys (SmolVLA, π0.5), key recipes, before/after fine-tuning, the diagnostic plane, head vs kNN |
| [04_neighbor_metric_and_learned_key.md](04_neighbor_metric_and_learned_key.md) | The LAC metric and its correlation with closed-loop success; a ResNet key trained to maximise it |
| [05_libero_plus.md](05_libero_plus.md) | LIBERO-trained policies on the perturbed LIBERO-plus benchmark |
| [06_planned_and_open.md](06_planned_and_open.md) | What is still running, planned experiments, open questions, known bugs |

## Headline results (as of 2026-09-30)

| Finding | Number |
|---|---|
| kNN lookup on LIBERO, official `lerobot-eval`, 2,000 episodes, one configuration, no training | **91.7%** (Multi-task DiT per the LeRobot docs: 90.6%; π0.5: 97.5%) |
| LIBERO tasks solved in ≥90% of episodes by a *plain* lookup over ImageNet ResNet-18 features | **31 / 40** (35 / 40 with the full method) |
| Share of each VLA's successes that the plain ImageNet lookup also achieves (same 400 episodes) | **88–94%** |
| Action-expert gain = VLA − plain lookup on its own VLM | SmolVLA **−5.3**; π0.5 **+12.5** (π0 pending) |
| Offline LAC@8 vs closed-loop lookup success, over the 10 keys never trained on LIBERO | Spearman **ρ = 0.90** (0.33 with the fine-tuned VLM keys, n = 18) |
| ResNet key trained to maximise LAC | plain lookup 87.3 → **90.5%**, full method 90.6 → **91.5%** |
| LIBERO-plus (420 perturbed variants) | π0.5 85.5%, kNN full 82.4%, plain kNN 75.5%, X-VLA 64.5%, π0 42.9%, SmolVLA 27.9%, π0-FAST 25.7%; an environment control confirms the collapses are caused by the perturbations (reworded instructions: SmolVLA 82 → 22%, π0-FAST 88 → 33%) |

## Protocol shared by all closed-loop comparisons

- **Benchmark.**
  - LIBERO has 4 suites × 10 tasks, about 42 demos per task and 2 cameras (agent view + wrist).
  - The dataset is `lerobot/libero`: 1,693 episodes at 20 fps.
- **Same 400 episodes for every comparison.** Initial states 20–29 of every task, 10 per task. Every policy in the
  comparison tables was run on exactly these episodes, so rows are paired.
  - Hyper-parameters were searched on initial states 40–49 and selected on 0–19.
  - The official headline uses `lerobot-eval` on 0–49 (2,000 episodes).
- **Runner.** `experiments/libero_pool.py` runs 10 parallel simulator processes with one inference thread. It
  follows the exact `lerobot-eval` observation→action path: same env/policy processors, same per-suite step limits
  (280 / 280 / 300 / 520).
- **VLA runs.** VLAs run through `VLARollout` (`vla.py`) with `n_action_steps = 10`, unless noted otherwise.
- **Statistical resolution.** With 400 episodes the standard error is about 1.5 pp at 90% success and 2 pp at 80%.
  Differences under about 4 pp between two rows are within noise.
- **Hardware.** One RTX 5090, capped at 400 W. At about 570 W the GPU dropped off the bus (Xid 79) twice; see
  `outputs/GPU_XID79_ISSUE.md`. One heavy GPU job runs at a time, and the machine is shared.

## Where things live

| What | Path |
|---|---|
| Plugin code | `src/lerobot_policy_knn/` (`vla.py`, `memory.py`, `modeling_knn.py`, `experiments/*.py`) |
| kNN results | `outputs/knn/compare.jsonl`, `compare_episodes.jsonl`, `final.json`, `optuna.db`, `report/` |
| VLA results | `outputs/vla/results.jsonl`, `results_episodes.jsonl`, `neighbor_metric*.json`, `hub_sanity.json`, `head_vs_knn_*.json` |
| Run scripts and logs | `outputs/vla/phase_*.sh`, `outputs/vla/logs/` |
| LIBERO-plus | `outputs/libero_plus/` (venv, fork, `sample.json`, `results*.jsonl`, `logs/eval.log`) |
| Poster | `poster/` (`make` rebuilds data, figures, `numbers.tex` and the PDF) |
| Earlier report page | https://claude.ai/artifact/LgoGXHYTYmsriRZZ12rzYe |

Nothing has been committed yet.
