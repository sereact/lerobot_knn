# "Your VLA is Secretly kNN": experiment notes

The notes cover the experiments behind the `lerobot_policy_knn` plugin, the A1 poster (`poster/`) and the CoRL 2026
workshop paper (`paper/`): what was run, how, and what came out. Numbers are copied from the result files named in
each section. The poster and the paper are regenerated from those same files by `make` in their directories.

## The question

Is a vision-language-action model (VLA) effectively an **embedding** (its VLM, the *backbone*) plus a
**nearest-neighbour lookup** over the demonstrations it was trained on? A kNN lookup needs no training, so it works
as a **probe**. It is not a proposal for robot control: a lookup cannot extrapolate beyond its memory.

1. **How hard is a task or benchmark?** Measure what a lookup over generic image features already solves, and on
   which episodes it fails.
2. **How good is a VLA's backbone, and what did fine-tuning do to it?** Run the same lookup keyed on the VLA's own VLM,
   before and after fine-tuning, in closed loop. Offline, measure how often close embeddings have close actions (the
   *local action consistency*, LAC).
3. **How good is its action head?** Measure what the VLA achieves beyond a lookup on its own VLM.

## Files

| File | Contents |
|---|---|
| [01_knn_policy.md](01_knn_policy.md) | The kNN policy, hyper-parameter search, official LIBERO result, ablations, side studies |
| [02_vla_checkpoints.md](02_vla_checkpoints.md) | The nine VLAs (one working LIBERO fine-tune per family), loading pitfalls, episode-level overlap with the lookup (task hardness) |
| [03_backbone_vs_action_expert.md](03_backbone_vs_action_expert.md) | VLM hidden states as lookup keys, key recipes, before/after fine-tuning, backbone vs action head, embedding maps, head vs kNN |
| [04_neighbor_metric_and_learned_key.md](04_neighbor_metric_and_learned_key.md) | The LAC metric and how it tracks closed-loop success; a ResNet key trained to maximise it |
| [05_libero_plus.md](05_libero_plus.md) | LIBERO-trained policies on the perturbed LIBERO-plus benchmark |
| [06_planned_and_open.md](06_planned_and_open.md) | Status, open questions, known issues |

## Headline results (final)

LIBERO numbers use the official protocol (initial states 0–49, 2,000 episodes); LIBERO-plus numbers use the fixed
420-variant sample.

| Finding | Number |
|---|---|
| kNN lookup on LIBERO, official `lerobot-eval`, one configuration, no training | **91.7%**, rank 7 of 10 (π0.5: 97.7%) |
| The same on LIBERO-plus | **82.4%**, the smallest drop of all policies (−9.3; most robust VLA −12.2, worst −64.8), rank 3 of 10 |
| When a VLA fails a LIBERO episode, the lookup fails it too | in 20–46% of cases, against 8% overall (2.4–5.5× chance); per-task success correlates with the lookup's (median r = 0.74) |
| A lookup predicts a VLA's action chunks at the states it visits (ResNet-18 key, k = 8) | R² 0.90 for π0.5, as well as or better than for a held-out human demo (0.78) |
| Lookup on the VLA's own fine-tuned VLM beats the VLA | 2 of 9 on LIBERO, 4 of 9 on LIBERO-plus, by up to 48 points |
| Fine-tuning and the VLM as a key | +5 to +37 points on LIBERO-plus for 7 of 8 VLAs; clean success up for 5; π0's key collapses (86.9 → 17.9%) |
| Offline LAC@8 vs closed-loop lookup success, 12 keys never trained on LIBERO | tracks it: Pearson r = 0.78, Spearman ρ = 0.61; π0's broken key LAC 0.55 vs 0.76–0.88 for every other key |
| ResNet key trained to maximise LAC | plain lookup 88.7 → **93.0%**, full method 91.7 → **93.0%** |
| Reworded instructions over the original LIBERO-plus scenes | SmolVLA 82 → 22%, π0-FAST 88 → 33%, π0 68 → 12% |

The full results table (12 policies × LIBERO and LIBERO-plus) and the backbone table are in the top-level
[README](../../README.md).

## Protocols

- **Benchmark.**
  - LIBERO has 4 suites × 10 tasks, about 42 demos per task and 2 cameras (agent view + wrist).
  - The dataset is `lerobot/libero`: 1,693 episodes at 20 fps.
- **Official protocol (final numbers).** Initial states 0–49 of every task, 2,000 episodes. The two full-method kNN
  configurations ran with `lerobot-eval` (`outputs/knn/eval_official/*/eval_info.json`); everything else ran in the
  pool on the same initial states, tagged `full50` (kNN rows in `outputs/knn/compare.jsonl`, VLA rows in
  `outputs/vla/results.jsonl`). The 95% interval around 92% success is about ±1.2 pp.
- **Development protocol.** During the study, comparisons used the same 400 episodes for every policy: initial
  states 20–29, 10 per task. Several tables below (key recipes, offline pilots) only exist on these episodes; they
  are marked. With 400 episodes the standard error is about 1.5 pp at 90% success and 2 pp at 80%, so differences
  under about 4 pp are within noise.
  - Hyper-parameters were searched on initial states 40–49 and selected on 0–19.
- **Runner.** `experiments/libero_pool.py` runs 10 parallel simulator processes with one inference thread. It
  follows the exact `lerobot-eval` observation→action path: same env/policy processors, same per-suite step limits
  (280 / 280 / 300 / 520); episode `i` of a task uses initial state `i`.
- **VLA runs.** VLAs run through `VLARollout` (`vla.py`) with `n_action_steps = 10`, unless their LeRobot settings
  say otherwise (02).
- **Hardware.** One RTX 5090, capped at 400 W. At about 570 W the GPU dropped off the bus (Xid 79) twice. One heavy
  GPU job ran at a time; the machine is shared.

## Where things live

| What | Path |
|---|---|
| Plugin code | `src/lerobot_policy_knn/` (`modeling_knn.py`, `memory.py`, `vla.py`, `experiments/*.py`) |
| Tracked configurations | `outputs/knn/*.json`, `outputs/knn/{ablations,ksweep,retry,tta}/`, `outputs/vla/configs/`, `outputs/vla/extra_keys.json`, `outputs/vla/split.json` |
| kNN results (local) | `outputs/knn/compare.jsonl`, `compare_episodes.jsonl`, `eval_official/`, `optuna.db` |
| VLA results (local) | `outputs/vla/results.jsonl`, `results_episodes.jsonl`, `neighbor_metric*.json`, `hub_sanity.json`, `head_vs_knn_*.json` |
| LIBERO-plus (local) | `outputs/libero_plus/` (venv, fork, `sample.json`, `results*.jsonl`) |
| Everything the poster and paper show | `poster/data/poster_data.json` (tracked), built by `poster/scripts/prepare_data.py` |
| Poster, paper | `poster/`, `paper/` (`make` rebuilds data, figures, `numbers.tex` and the PDF) |

Result files under `outputs/` are not in git; only the configurations are.
