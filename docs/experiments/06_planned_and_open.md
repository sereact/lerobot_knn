# 6. Running, planned, and open

## Running now

Phases N–V are done; their results are in 02–05. On 2026-10-01 the study moved from the LeRobot checkout
(`plugins/lerobot_policy_knn` + `outputs/`) to this standalone repo. Phases W, X3 and Y were stopped at a step
boundary, and `outputs/vla/phase_z.sh` runs what they had left:

| Part | What | Result goes to |
|---|---|---|
| W, done before the move | X-VLA keys (base 86.0 → LIBERO 87.8 clean; 68.3 → 73.6 LIBERO-plus); π0-FAST base key on LIBERO (87.5) | `outputs/knn/compare.jsonl`, `outputs/libero_plus/` |
| W, rest | π0-FAST base on LIBERO-plus; π0-FAST LIBERO and both VLA-JEPA keys (LIBERO and LIBERO-plus); X-VLA and VLA-JEPA LIBERO-plus recordings with `head_vs_knn` | as above |
| X3 | MolmoAct2, Evo-1 at 256 px renders, GR00T N1.7 per suite: closed loop on the 400 episodes and LIBERO-plus, each after passing `vla_sanity` (MolmoAct2 and GR00T Spatial already pass) | `outputs/vla/results.jsonl`, `outputs/libero_plus/results.jsonl` |
| Y | GR00T's frozen VLM as a key (layer pilot, LIBERO and LIBERO-plus lookups), then `neighbor_metric`, `plus_neighbor_metric` and `plus_language` over every key | as W |

## Why the DiT was not reproduced, and the replacement

- **The budget.** The LeRobot docs' recipe is 100k steps at batch 320 (32M samples). Our run, 60k steps at batch 32
  (1.9M samples), took 4 h 42 min on the 5090. The full recipe would take about 3 days of GPU time at that rate, and
  batch 320 with CLIP fine-tuning does not fit in 32 GB without gradient accumulation. It was cut short before the
  "no training on this machine" decision.
- **No official checkpoint.** The docs' recipe sets `--policy.push_to_hub=false`.
- **Closest public checkpoint.** `Gen1113/dit-libero-all`: the same `multi_task_dit` policy (CLIP ViT-B/16, 8 layers,
  diffusion objective) trained on `HuggingFaceVLA/libero` for 300k steps at batch 64. That is 19.2M samples, 10× ours
  and 60% of the docs'; its horizon is 32 against the docs' 48. It publishes no results, so phase Q measures it.

## Planned next

1. **Poster.** Add MolmoAct2, Evo-1 and GR00T to the episode-overlap chart and the LIBERO-plus heatmap. Add π0-FAST,
   VLA-JEPA and GR00T to the diagnostic. The middle column is nearly full: two more rows per chart need about 3 cm.
2. **Backbone keys for MolmoAct2 and Evo-1** (not started). MolmoAct2's action expert reads every VLM layer, and its
   multi-crop image tokens need their own embedder path. Evo-1 (InternVL3-1B) has no checkpoint from before LIBERO.
3. **`head_vs_knn` for GR00T, MolmoAct2 and Evo-1** on recorded LIBERO-plus rollouts (needs `--record_per_cell` runs).
4. **Tighter LIBERO-plus numbers.** Consider 30 variants per cell (840 episodes), giving about ±4 pp per category.

## Hyper-parameter search: is it enough?

- **Trials so far:** 193 in total, each over 400 episodes. The single-configuration studies had 29 (`knn_v1`) and
  24 (`knn_v2`); the four per-suite studies (`knn_v3_*`) had 29–43 each.
- **Why more would buy little at this episode count:**
  - The best held-out configurations sit within 1 pp of each other (91.6 / 91.3 / 91.0 / 90.6 on 800 episodes).
  - Every extra in the ablation moves success by about a point.
  - The standard error of one 400-episode trial is about 1.4 pp.

  More trials at the same budget would mostly fit noise (the winner's curse).
- **A larger search is only useful with more episodes per trial**, e.g. 800–2,000 with a successive-halving pruner.
  At about 20–40 min per trial, 100 more trials is a day or two of GPU time.
- **The learnt key already gave more than tuning did:** the plain lookup went 87.3 → 90.5 with no extras.

## Possible extensions (not started)

- **Per-task hardness beyond LIBERO.** Apply the same lookup probe (success of a generic-feature lookup, plus the
  episode overlap) to other LeRobot benchmarks, e.g. MetaWorld, RoboCasa, VLABench, to measure how
  retrieval-solvable each is.

## Open questions

- Why does π0.5's key get *worse* in closed loop after LIBERO fine-tuning (88.5 → 84.5 at layer 12), while its offline
  LAC rises?
  - Our reading: memorisation of the training demos (the offline gain) vs worse generalisation to new initial states.
  - A memorisation-free LAC (item 5) would test it.
- What exactly does π0.5's action expert add over a lookup? It is concentrated on LIBERO-Long and on the episodes
  where retrieval fails (it solves 88% of them). Is that recovery from off-demo states, or composition across demos?
- The weaker VLAs fail on reworded instructions over unchanged scenes. Is that the language model's grounding, or overfitting of the action expert to 40 fixed instruction strings? Comparing their VLM keys on original vs reworded prompts would tell.

## Known issues and bugs found along the way

| Where | Issue | Status |
|---|---|---|
| `PI0Policy.from_pretrained` | a strict load that meets legacy normaliser buffers prints a warning and returns the model **without its weights** (seen with `lerobot/pi0_libero_base`) | worth a LeRobot issue |
| `lerobot/pi05_libero_base`, `lerobot/pi0_libero(_base)` | no normaliser statistics shipped; for π0.5 also a prompt (state tokens) and chunk length (50 vs 10) that differ from PI's training | not used in the study; worth reporting on the Hub |
| `lerobot-eval --env.max_parallel_tasks > 1` | shared tokenizer step (`self.transition`) and shared policy action queue across threads | confirmed in the code; the user is preparing a fix; our runs are unaffected |
| LIBERO-plus fork | `np.float_` removed in NumPy 2 crashes fog-noise variants | patched locally |
| Z-scored multi-layer π0.5 keys | exhausted RAM and swap; simulator workers hung | use `zscore_pca512` |
| RTX 5090 | Xid 79 (falls off the bus) at about 570 W sustained | capped at 400 W, stable since |
| HF Hub | Xet transfers failed repeatedly | `HF_HUB_DISABLE_XET=1` |
| GR00T N1.7 LIBERO checkpoints | they are LeRobot checkpoints; built the raw-checkpoint way (`--policy.base_model_path=<repo>`, as the docs' eval command does), the network loads **without its weights** (random init, only "MISSING" key warnings) and the processors without their statistics | load with `from_pretrained` and the saved processors; worth checking the docs' eval command against these checkpoints |

## Dropped from the study

- Other checkpoints of the same families:
  - community and HF SmolVLA fine-tunes;
  - PI's own π0.5 and π0 LIBERO conversions;
  - `lerobot/pi05-libero`.

  One working checkpoint per family is kept.
- Our own SmolVLA fine-tune and its dose-response study: stopped at the user's request, since the work should use
  Hub checkpoints only.
