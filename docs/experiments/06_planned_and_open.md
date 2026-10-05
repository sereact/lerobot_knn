# 6. Status, open questions, and known issues

## Status (2026-10-05)

Every experiment phase has finished (the last were AA–AI: the X-VLA re-runs, the official-protocol evaluations of every
VLA and VLM key, the ablations on 2,000 episodes, and the LIBERO-plus runs of the new families). Their results are in
01–05, the poster and the paper. On 2026-10-01 the study moved from the LeRobot checkout to this standalone repo; LeRobot
itself is an unmodified, pinned dependency.

- **Official protocol everywhere on LIBERO** (initial states 0–49, 2,000 episodes), including the ablations and every
  VLM key.
- **LIBERO-plus stays a 420-variant sample** (60 per perturbation, about ±3.5 pp overall). The full benchmark (10,030
  variants) would cost about 24× the GPU time per policy.
- **No VLA was fine-tuned here**: every checkpoint is a public Hub release.

## Hyper-parameter search: is it enough?

- **Trials:** 193 in total, each over 400 episodes. The single-configuration studies had 29 (`knn_v1`) and 24
  (`knn_v2`); the four per-suite studies (`knn_v3_*`) had 29–43 each.
- **Why more would buy little at this episode count:**
  - The best held-out configurations sit within 1 pp of each other (91.6 / 91.3 / 91.0 / 90.6 on 800 episodes).
  - Every extra in the ablation moves success by about a point.
  - The standard error of one 400-episode trial is about 1.4 pp.

  More trials at the same budget would mostly fit noise (the winner's curse). A larger search is only useful with
  more episodes per trial (800–2,000 with a successive-halving pruner).
- **The learnt key gave more than tuning did:** the plain lookup went 88.7 → 93.0% with no extras.

## Possible extensions

- **Other benchmarks.** Apply the same probe (success of a generic-feature lookup, the episode overlap, VLM keys before
  and after fine-tuning) to other LeRobot benchmarks, e.g. MetaWorld, RoboCasa, VLABench, and to LIBERO-PRO alongside
  LIBERO-plus, to measure how retrieval-solvable each is.
- **Alternative keys.** ALT's contrastive key and VINN's BYOL key (see the literature review).
- **Interpretability.** Relate LAC to the sparse-autoencoder "memorisation features" reported inside the same VLAs.

## Open questions

- Why does π0.5's key get slightly *worse* in closed loop after LIBERO fine-tuning (85.4 → 81.8 at layer 12), while its
  offline LAC rises? Our reading: memorisation of the training demos (the offline gain) vs worse generalisation to new
  initial states. On LIBERO-plus, where no checkpoint saw the demos, the fine-tuned key is the better one (60.5 → 65.7).
- What exactly does π0.5's action expert add over a lookup? It is concentrated on LIBERO-Long and on the episodes
  where retrieval fails (it solves 94% of them). Is that recovery from off-demo states, or composition across demos?
- The weaker VLAs fail on reworded instructions over unchanged scenes. For SmolVLA and π0-FAST the backbone key barely
  moves under rewording, so the sensitivity sits in the action expert; for π0 the fine-tuned backbone itself keys on
  the wording (05). Is that overfitting to 40 fixed instruction strings?

## Known issues and bugs found along the way

| Where | Issue | Status |
|---|---|---|
| `PI0Policy.from_pretrained` | a strict load that meets legacy normaliser buffers prints a warning and returns the model **without its weights** (seen with `lerobot/pi0_libero_base`) | worth a LeRobot issue |
| `lerobot/pi05_libero_base`, `lerobot/pi0_libero(_base)` | no normaliser statistics shipped; for π0.5 also a prompt (state tokens) and chunk length (50 vs 10) that differ from PI's training | not used in the study; worth reporting on the Hub |
| `lerobot-eval --env.max_parallel_tasks > 1` | shared tokenizer step (`self.transition`) and shared policy action queue across threads | confirmed in the code; our runs are unaffected |
| LIBERO-plus fork | `np.float_` removed in NumPy 2 crashes fog-noise variants | patched locally |
| Z-scored multi-layer π0.5 keys | exhausted RAM and swap; simulator workers hung | use `zscore_pca512` |
| RTX 5090 | Xid 79 (falls off the bus) at about 570 W sustained | capped at 400 W, stable since |
| HF Hub | Xet transfers failed repeatedly | `HF_HUB_DISABLE_XET=1` |
| GR00T N1.7 LIBERO checkpoints | they are LeRobot checkpoints; built the raw-checkpoint way (`--policy.base_model_path=<repo>`, as the docs' eval command does), the network loads **without its weights** (random init, only "MISSING" key warnings) and the processors without their statistics | load with `from_pretrained` and the saved processors; worth checking the docs' eval command against these checkpoints |
| X-VLA observation convention | from 2026-10-01 11:24 to 2026-10-02 the standard LIBERO convention was used instead of X-VLA's own | rows moved to `outputs/*/invalid_rows.jsonl`, redone (02) |

## Dropped from the study

- Other checkpoints of the same families: community and HF SmolVLA fine-tunes; PI's own π0.5 and π0 LIBERO
  conversions (with openpi's inputs and statistics); `lerobot/pi05-libero`. One working checkpoint per family is kept;
  their registry entries were removed in the cleanup (in commit `b4acfae`).
- Our own SmolVLA fine-tune and its dose-response study: stopped at the user's request, since the work should use Hub
  checkpoints only.
- Our own multi-task DiT (01, side study): an undertrained reduced-budget run (68.8%); the full recipe would have taken
  about 3 days of GPU time and there is no official checkpoint. The VLA measurements replaced it.
