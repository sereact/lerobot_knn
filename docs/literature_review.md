# Literature review: "Your VLA is Secretly kNN"

What other work says about retrieval policies, memorisation in VLAs, what fine-tuning does to the VLM backbone, VLA
interpretability, and offline evaluation, and where our probe sits. Collected 2026-10-03; every entry was checked
against its arXiv or venue page. Entries cited in `paper/refs.bib` are marked **[cited]**.

## Our position in one paragraph

Prior work uses retrieval *as a policy* or to *select data*, shows with perturbation benchmarks *that* VLAs memorise,
measures VLM forgetting with *vision-language* benchmarks (VQA, POPE), and interprets VLA internals with probes and
sparse autoencoders. We turn a training-free kNN lookup into a *measuring instrument* and use it for three questions:
- how much of a benchmark is retrieval (our lookup: 91.7% on official LIBERO, no training);
- how good a VLA's backbone is *as an action-relevant key*, before and after fine-tuning;
- what its action head adds over that key.

We also add LAC, an offline neighbour metric that tracks a key's closed-loop success (Pearson r 0.78, Spearman
ρ 0.61 over 12 keys that never saw the demos) and flags a broken backbone. We found no prior training-free
retrieval policy evaluated on the full LIBERO protocol, and no prior work that scores backbones and action heads
separately by closed-loop success.

## 1. Retrieval and non-parametric policies

| Work | What it does | Relation to us |
|---|---|---|
| **VINN**, Pari et al., RSS 2022 **[cited]** | kNN over self-supervised (BYOL) visual features for visual imitation; decouples representation from behaviour. | The closest ancestor of our policy. We add the action-chunk machinery (mode seeking, temporal ensembling, history), reach VLA-level LIBERO success, and use the lookup as a probe rather than a method. |
| **Behavior Retrieval**, Du et al., RSS 2023 **[cited]** | Retrieves relevant data from unlabelled play data for few-shot imitation. | Retrieval for data selection, not as the policy. |
| **STRAP**, Memmel et al., ICLR 2025 (arXiv 2412.15182) **[cited]** | Sub-trajectory retrieval with vision foundation models and dynamic time warping, then trains on the retrieved data. | Same: retrieval feeds training. |
| **REGENT**, Sridhar, Dutta, Jayaraman, Lee, ICLR 2025 oral (arXiv 2412.04759) **[cited]** | Semi-parametric generalist agent: a transformer over a query plus retrieved neighbours, which adapts to new environments in-context. | Its "Retrieve and Play" baseline, which replays the action of the closest demonstration state, "performs on-par or better than state-of-the-art generalist agents" (JAT/Gato). That is our message in a different setting. |
| **RICL**, same authors, CoRL 2025 (arXiv 2508.02062) **[cited]** | Post-trains a VLA (π0-FAST) to use retrieved demos in context; the final action interpolates the retrieved and generated actions by distance. | Retrieval inside a VLA. We ask the reverse question: how much of a VLA already behaves like retrieval. |
| **Retrieval-VLA**, Zhang et al., CVPR 2026 Findings | Training-free in-context adaptation of VLAs by retrieval (PDF not accessible, from the title and venue). | Training-free retrieval around a VLA. |
| **ALT: Demystifying diffusion policies**, He, Liu, Sznaier Camps, Sartoretti, Schwager, arXiv 2505.05787 (2025) **[cited]** | Argues diffusion policies memorise an action lookup table in the sparse-data regime, and proposes ALT, a lookup over a contrastive image encoder. | The closest in spirit. They study diffusion policies on their own tasks; we study nine large VLAs on a public benchmark, quantify how lookup-like the VLAs are, and separate backbone from action head. |
| **Closed-form diffusion policies (CFDP)**, Mishra & Manchester, arXiv 2606.01238 (2026) **[cited]** | A training-free diffusion policy with the score derived in closed form from the demonstrations; millisecond inference. | Another training-free policy over the demos, essentially kernel smoothing. Supports "the memory is the model". |
| **kNN-LM**, Khandelwal et al., ICLR 2020 **[cited]** | "Generalization through memorization": interpolating a language model with a kNN over its own hidden states improves perplexity. | The language-model analogue of keying a lookup on a VLA's internal states. |

## 2. Memorisation and robustness of VLAs

| Work | What it shows | Relation |
|---|---|---|
| **LIBERO**, Liu et al., NeurIPS 2023 D&B **[cited]** | The benchmark. | We measure how much of it is retrieval. |
| **LIBERO-Plus**, Fei et al., arXiv 2510.13626 (2025) **[cited]** | Seven perturbation dimensions. Success drops from about 95% to below 30% under camera and robot-state shifts, and models "tend to ignore language instructions completely". | Our robustness benchmark. Our language finding differs: the weak VLAs (SmolVLA, π0-FAST, π0) collapse on *reworded* instructions, so they key on the exact wording. |
| **LIBERO-PRO**, Zhou et al., arXiv 2510.03827 (2025) **[cited]** | Perturbs objects, initial states, instructions and environments. VLAs rely on "rote memorization of action sequences and environment layouts". | The same conclusion from a different angle. We *quantify* it with a probe that needs no new benchmark, and show which part (backbone or action head) the success comes from. |
| **Affordance Field Intervention**, Xu et al., arXiv 2512.07472 (2025) **[cited]** | VLAs (π0, π0.5) fall into "memory traps": they replay memorised trajectories instead of adapting (LIBERO-Pro). | Consistent with our "VLAs act like a lookup" finding. |
| **What can RL bring to VLA generalization?**, arXiv 2505.19789 (2025) | RL fine-tuning vs supervised fine-tuning for out-of-distribution generalisation. | Background. |
| **Much Ado About Noising**, Pan et al., arXiv 2512.01809 (2025) **[cited]** | Generative control policies do not win by capturing multi-modality; a two-step regression policy matches flow policies. | Another case of a simple policy matching complex ones on behaviour-cloning benchmarks. |

## 3. What VLA fine-tuning does to the VLM backbone

| Work | What it shows | Relation |
|---|---|---|
| **Knowledge Insulating VLAs**, Driess et al., NeurIPS 2025 (arXiv 2505.23705) **[cited]** | Naively training an action expert with the VLM harms training speed and knowledge transfer. Insulating the backbone (stop-gradient, FAST co-training) generalises better; this is π0.5's recipe. | We measure the backbone after fine-tuning *as an action key*. π0.5's key changes little on LIBERO (85.4 → 81.8% on the official protocol), while π0 collapses as a key (86.9 → 17.9%). |
| **Preserving pretrained representations**, Grover et al., arXiv 2509.11417 (2025) **[cited]** | Fine-tuning disrupts pretrained representations. A frozen extra vision encoder, string action tokens and co-training improve out-of-distribution visuals. | Same concern. We find that LIBERO fine-tuning mostly makes the VLM a *more robust* lookup key, which complicates the pure-forgetting picture. |
| **VLM4VLA**, Zhang et al., arXiv 2601.03309 (2026) **[cited]** | Converts many VLMs into VLAs with a minimal adapter. General VLM competence transfers poorly; control-relevant supervision of the vision encoder helps. | Ours is a training-free version of the same question: how good is a VLM as an action-relevant representation? |
| **TwinBrainVLA** (2601.14133), **MAPS** (2511.19878), **ReVLA** (2409.15250), **Anchor-Align** (2607.13429) | Catastrophic forgetting of the VLM during VLA training, measured with VQA/POPE or depth probes, and methods to prevent it. | These measure forgetting of *vision-language* skills; we measure the backbone's *action relevance* by closed-loop success, before and after fine-tuning. |
| **Massive activations**, Sun et al., COLM 2024 **[cited]** | A few huge channels dominate LLM residual streams. | Why we standardise VLM keys per dimension. |

## 4. Interpretability of VLAs

| Work | What it shows | Relation |
|---|---|---|
| **Mechanistic interpretability for steering VLAs**, Häon et al., arXiv 2509.00328 (2025) **[cited]** | Semantic directions (speed, direction) in π0 / OpenVLA feed-forward layers, usable for steering (LIBERO, UR5). | Internal analysis by probing; we analyse by retrieval. |
| **Sparse autoencoders in VLAs**, Swann, McGranahan, Buurmeijer, Kennedy, Schwager, arXiv 2603.19183 (2026) **[cited]** | Sparse-autoencoder features are either general motion primitives or *episode-specific memorisations* (LIBERO, DROID). | Direct evidence of memorisation features inside VLAs; from the same group as ALT. |
| **Not all features are created equal**, Grant, Zhao, Wang, arXiv 2603.19233 (2026) **[cited]** | Across six VLAs (80M–7B, 394k episodes) the visual pathway dominates. Language matters only when the scene does not specify the task. In multi-pathway VLAs (π0.5, SmolVLA, GR00T) the expert encodes motor programs and the VLM goal semantics. | Matches our split: the lookup on the VLM is the backbone's contribution, the rest is the expert's. |
| **VLA-Trace**, Shi et al., arXiv 2605.30117 (2026) | Kernel alignment and attention interventions on π0.5 / OpenVLA: strong visually grounded trajectories, weak semantics. | Background. |

## 5. Representations for robotics and offline evaluation

| Work | What it shows | Relation |
|---|---|---|
| **R3M**, Nair et al., CoRL 2022 (arXiv 2203.12601); **CortexBench / VC-1**, Majumdar et al., NeurIPS 2023 (arXiv 2303.18240) | Pretrained visual representations for robotics, evaluated by training policies on top; no representation dominates everywhere. | Our VLM keys are evaluated without training a policy, by lookup and by LAC. |
| **robomimic**, Mandlekar et al., CoRL 2021 **[cited]** | Policies are trained on surrogate losses, so validation loss does not say which checkpoints are good; they must be tried on the robot. | Motivates an offline metric that *does* track closed-loop success (LAC: Pearson r 0.78, Spearman ρ 0.61, 12 keys). |

## Open gaps our work fills

1. **A lookup baseline on full LIBERO.** No published work reports a training-free lookup on the official protocol. Ours is 91.7%, rank 7 of 10 (the full kNN method and nine public VLAs), and has the smallest drop under perturbation (rank 3 of 10 on LIBERO-plus).
2. **Backbone vs action head by closed-loop success.** Prior work separates them by probing or ablation, not by a policy that uses only the backbone.
3. **Before vs after fine-tuning as an action key.** Forgetting studies use VQA; we use action relevance. Fine-tuning mostly *improves* the key's robustness (+5 to +37 points on LIBERO-plus for 7 of 8 VLAs) but can destroy it (π0).
4. **An offline metric validated against closed loop.** LAC tracks the closed-loop success of a lookup with each key
   without rollouts (r 0.78, ρ 0.61) and flags π0's broken fine-tuned key (LAC 0.55 vs 0.76–0.88).

## Possible extensions suggested by the literature

- Run the probe on LIBERO-PRO alongside LIBERO-plus; both perturbation benchmarks are public.
- Compare with ALT's contrastive key, and with VINN's BYOL key, as alternative keys.
- Relate LAC to sparse-autoencoder "memorisation features" (Swann et al.) inside the same VLAs.
