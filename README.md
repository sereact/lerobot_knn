# How Much of Your VLA is kNN

[![arXiv](https://img.shields.io/badge/arXiv-2611.13160-b31b1b.svg)](https://arxiv.org/abs/2611.13160)
[![CoRL 2026 Workshop](https://img.shields.io/badge/CoRL%202026-G2A%2FTAG%20workshop-blue.svg)](https://actiongapworkshop.github.io)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

![A k-NN policy retrieves the nearest demonstration frames and executes their action chunks](assets/pipeline.png "The k-NN retrieval policy")

> **How Much of Your VLA is kNN**\
> Denis Grachev, Damien LaRocque, Omar Moured\
> Paper: https://arxiv.org/abs/2611.13160

This repo contains the source code of our paper _How Much of Your VLA is kNN_, presented at the [CoRL 2026 Workshop on G2A: From Generative Models to Robot Actions](https://actiongapworkshop.github.io).
Rollouts and interactive results are available on the [project page](https://sereact.github.io/lerobot_knn).

## Source Code

The k-NN policy is an out-of-tree [LeRobot](https://github.com/huggingface/lerobot) [policy plugin](https://huggingface.co/docs/lerobot/bring_your_own_policies), available as `--policy.type=knn`.
The `lerobot-knn-fit` command builds its memory from any LeRobot dataset, and the standard `lerobot-eval` command can be used to evaluate the resulting policy:

```sh
# Embed lerobot/libero with a frozen encoder and save the memory as a policy
lerobot-knn-fit --repo_id=lerobot/libero --output_dir=outputs/knn/policies/my_knn \
    --policy.image_encoder=resnet18_l3 --policy.image_pooling=grid4 --policy.k=8

# Evaluate it like any other LeRobot policy
MUJOCO_GL=egl lerobot-eval --policy.path=outputs/knn/policies/my_knn --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.observation_height=256 --env.observation_width=256 \
    --eval.n_episodes=50 --eval.batch_size=10
```

To use a VLA backbondocs/e for retrieval keys, pass `--policy.image_encoder=vla:<name>` (for example `vla:pi05_libero`) together with `--policy.image_pooling=grid2@L12 --policy.key_norm=zscore`.

Most of the code is in [`src/lerobot_policy_knn`](src/lerobot_policy_knn/):

* [`modeling_knn.py`](src/lerobot_policy_knn/modeling_knn.py) and
  [`configuration_knn.py`](src/lerobot_policy_knn/configuration_knn.py) contain the policy and its options
* [`memory.py`](src/lerobot_policy_knn/memory.py),
  [`features.py`](src/lerobot_policy_knn/features.py) and
  [`encoders.py`](src/lerobot_policy_knn/encoders.py) build the memory of demonstration embeddings
* [`vla.py`](src/lerobot_policy_knn/vla.py) turns the backbones of nine VLA families into retrieval keys
* [`experiments/`](src/lerobot_policy_knn/experiments/) contains the evaluation and analysis scripts of the paper

The options of the policy are described in [`docs/policy.md`](docs/policy.md).

### Installation

We use [uv](https://docs.astral.sh/uv) and Python 3.12.
The VLAs and the simulator require an NVIDIA GPU.

```sh
uv sync --extra all           # everything: LIBERO, Optuna and every VLA family
uv sync --extra experiments   # k-NN policy, LIBERO evaluation and hyperparameter search
```

LeRobot is an unmodified dependency, pinned in `pyproject.toml` to the commit used in the paper.

## Benchmarks

* **LIBERO**: the memory contains the 1,693 training demonstrations of [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), storing every second frame for approximately 137,000 entries.
We evaluate the Spatial, Object, Goal, and Long suites with initial states 0-49 for all 40 tasks, totaling 2,000 episodes per policy.
* **LIBERO-plus**: evaluation uses a fixed sample of 420 perturbed variants, with 60 per perturbation category across seven categories, shared across all policies and held out from training and hyperparameter tuning.
  [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) replaces the `libero` package and therefore requires a separate environment, set up with [`outputs/libero_plus/setup.sh`](outputs/libero_plus/setup.sh).

## Model hyperparameters and checkpoints

### k-NN

The multimodal k-NN variant uses the following hyperparameters, selected with Optuna on initial states 40–49.
The configuration file is available in [`outputs/knn/v2_39.json`](outputs/knn/v2_39.json).

|        Parameter         | Value                                   |
| :----------------------: | --------------------------------------- |
|      Image encoder       | ResNet-18, layer 3 (ImageNet)           |
|      Image pooling       | 4×4 grid                                |
| Camera weights (agent, wrist) | 0.53, 0.57                   |
|       State weight       | 0.034                                   |
|  Distance normalization  | pair median                             |
|    Task conditioning     | soft, text weight 0.48                  |
|       Time weight        | 0.016                                   |
|           `k`            | 13, one neighbour per episode           |
|       Temperature        | 0.093                                   |
|      History weight      | 1.0                                     |
|  Action mode bandwidth   | 0.35                                    |
|       Chunk size         | 8, re-planned every step                |
| Temporal ensemble coeff. | 0.42                                    |
|      Memory stride       | 2                                       |

### VLA checkpoints

We evaluate the authors' public LIBERO checkpoints as released in LeRobot, without training anything.
Retrieval keys before LIBERO fine-tuning use the corresponding base checkpoint.
Each checkpoint has a name in `VLA_SPECS` in [`vla.py`](src/lerobot_policy_knn/vla.py), used as `vla:<name>`.

|    VLA     | Before LIBERO fine-tuning                     | After LIBERO fine-tuning                          |
| :--------: | --------------------------------------------- | ------------------------------------------------- |
| $\pi _{0.5}$    | `lerobot/pi05_base`                           | `lerobot/pi05_libero_finetuned_v044`              |
| $\pi _{0}$     | `lerobot/pi0_base`                            | `lerobot/pi0_libero_finetuned_v044`               |
| $\pi _{0}$-FAST  | `lerobot/pi0fast-base`                        | `lerobot/pi0fast-libero`                          |
|  SmolVLA   | `lerobot/smolvla_base`                        | `lerobot/smolvla_libero`                          |
|   X-VLA    | `lerobot/xvla-base`                           | `lerobot/xvla-libero`                             |
|  VLA-JEPA  | `lerobot/VLA-JEPA-Pretrain`                   | `lerobot/VLA-JEPA-LIBERO`                         |
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B`                        | `nvidia/gr00t17-lerobot-<suite>-640`, one per suite |
| MolmoAct2  | `allenai/MolmoAct2`                           | `allenai/MolmoAct2-LIBERO-LeRobot`                |
|   Evo-1    | `OpenGVLab/InternVL3-1B-hf` (first 14 layers) | `zuoxingdong/evo1_libero`                         |

The `lerobot/pi05_base` and `lerobot/pi0_base` checkpoints are converted to bfloat16 with
[`experiments/convert_bf16.py`](src/lerobot_policy_knn/experiments/convert_bf16.py).

## Reproducing the experiments

Each experiment is a module in [`experiments/`](src/lerobot_policy_knn/experiments/).
To view its options, run:

```sh
uv run python -m lerobot_policy_knn.experiments.<module> --help
```

The steps, expected outputs and notes for each experiment are documented in [`docs/experiments/`](docs/experiments/README.md).

## Acknowledgments

This work builds on [LeRobot](https://github.com/huggingface/lerobot), whose policies, datasets and evaluation we use
unmodified, as well as the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) benchmarks.

## Citation

If you use the code in an academic context, please cite the following work:

```bibtex
@inproceedings{grachev2026knn,
  title         = {{How Much of Your VLA is kNN}},
  author        = {Grachev, Denis and LaRocque, Damien and Moured, Omar},
  booktitle     = {CoRL 2026 Workshop on G2A: From Generative Models to Robot Actions},
  year          = {2026},
  eprint        = {2611.13160},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.13160}
}
```

## License

This project is licensed under an [Apache 2.0](LICENSE) license.


<!-- ## Repository structure

```sh
.
├── src/lerobot_policy_knn
│   ├── configuration_knn.py, modeling_knn.py, processor_knn.py   # the policy
│   ├── memory.py, features.py, encoders.py, fit.py               # the memory and lerobot-knn-fit
│   ├── vla.py                                                    # VLA backbones as keys, VLA rollouts
│   └── experiments                                               # evaluation and analysis scripts
├── outputs
│   ├── knn             # k-NN configurations: Optuna trials, v2_39.json, ablations, sweeps
│   ├── vla             # keys on VLA backbones, held-out split
│   └── libero_plus     # LIBERO-plus environment and the 420 sampled variants
├── poster              # figure scripts, their data and the workshop poster (corl2026)
├── docs                # policy options, experiment notes, literature review
└── tests
``` -->
