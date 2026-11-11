"""Optuna search over KNN policy hyper-parameters, scored by LIBERO closed-loop success.

Every trial builds a policy from the feature cache (seconds), evaluates it with a persistent
:class:`LiberoPool`, and logs per-suite / per-task success rates as trial user attributes. Trials are
stored in an SQLite database so the study can be resumed and browsed with ``optuna-dashboard``::

    uv run python -m lerobot_policy_knn.experiments.search --study=knn_v1 --n_trials=100
    uv run optuna-dashboard sqlite:///outputs/knn/optuna.db

Episodes use LIBERO initial states ``--init_states`` (default 40..49, a *validation* slice); final
numbers should be reported on states the search never saw.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

from ..configuration_knn import KNNConfig
from ..encoders import TextEncoder
from ..features import DEFAULT_CACHE_DIR, load_frame_index
from ..memory import build_memory, cache_loader
from ..modeling_knn import KNNPolicy
from ..processor_knn import make_knn_pre_post_processors
from .common import KNN_OUT, REPO_ID, append_jsonl, parse_range, suite_scores
from .libero_pool import SUITES, EpisodeResult, Job, LiberoPool, make_jobs, summarize

logger = logging.getLogger(__name__)

AGENT_VIEW, WRIST = "observation.images.image", "observation.images.image2"


class MemoryFactory:
    """Builds policies from the feature cache, memoising raw features and the task text embeddings."""

    def __init__(self, repo_id: str = REPO_ID, cache_dir: Path = DEFAULT_CACHE_DIR, device: str = "cuda"):
        self.repo_id = repo_id
        self.frames = load_frame_index(repo_id, cache_dir=cache_dir)
        self._load = cache_loader(cache_dir, repo_id)
        self._raw: dict[tuple, np.ndarray] = {}
        self._text: torch.Tensor | None = None
        self._encoders: dict[tuple[str, int], torch.nn.Module] = {}
        self.device = device

    def load_raw(self, encoder: str, camera: str, pooling: str, resolution: int) -> np.ndarray:
        # memory-mapped: rows are paged in from the OS page cache, which stays reclaimable
        key = (encoder, camera, pooling, resolution)
        if key not in self._raw:
            self._raw[key] = self._load(*key)
        return self._raw[key]

    def demo_subset(self, fraction: float, seed: int = 0) -> np.ndarray:
        """Episode ids keeping ``fraction`` of every task's demonstrations (at least one each)."""
        rng = np.random.default_rng(seed)
        keep = []
        for task in np.unique(self.frames.task_index):
            eps = np.unique(self.frames.episode_index[self.frames.task_index == task])
            n = max(1, int(round(fraction * len(eps))))
            keep.append(rng.choice(eps, n, replace=False))
        return np.sort(np.concatenate(keep))

    def build(self, config: KNNConfig, episodes: np.ndarray | None = None) -> KNNPolicy:
        """A policy whose memory holds ``episodes`` (default: every demonstration)."""
        if self._text is None:
            self._text = TextEncoder(config.text_encoder).to(self.device)(self.frames.task_names).cpu()
        buffers, _ = build_memory(
            config,
            self.frames,
            self.load_raw,
            episodes=episodes,
            device=self.device,
            task_text_emb=self._text,
        )
        config.dataset_repo_id = self.repo_id
        config.input_features = {}
        policy = KNNPolicy(config)
        policy.load_memory(buffers)
        policy = policy.to(self.device).eval()
        # share frozen backbones across the policies of a search instead of reloading them per trial
        key = (config.image_encoder, config.image_resolution)
        if config.active_cameras:
            if key not in self._encoders:
                self._encoders[key] = policy.image_encoder()
            policy._frozen["image"] = (self._encoders[key], policy.device)  # noqa: SLF001
        return policy

    def release_encoders(self) -> None:
        """Drop the shared image backbones (a VLM used as a key takes most of the GPU) and free their memory."""
        self._encoders.clear()
        torch.cuda.empty_cache()


def sample_config(trial: optuna.Trial, space: str) -> KNNConfig:
    """Hyper-parameter search space (``space`` versions the space so studies stay reproducible)."""
    return SEARCH_SPACES.get(space, sample_config_v1)(trial)


def sample_config_v1(trial: optuna.Trial) -> KNNConfig:
    """v1: which encoder and pooling, camera / state / time weights, k, temperature, stickiness, chunking."""
    encoders = {
        "dinov2_small": ("cls", "mean", "grid2"),
        "dinov3_small": ("cls", "mean", "grid2"),
        "resnet18": ("cls", "grid2"),
        "pixels": ("p16",),
    }
    enc = trial.suggest_categorical("image_encoder", list(encoders))
    pooling = trial.suggest_categorical(f"pooling_{enc}", list(encoders[enc]))
    w_agent = trial.suggest_float("w_agentview", 0.0, 1.0)
    w_wrist = trial.suggest_float("w_wrist", 0.0, 1.0)
    if w_agent + w_wrist < 0.05:
        w_agent = 0.05
    state_w = trial.suggest_float("state_weight", 1e-3, 3.0, log=True)
    use_time = trial.suggest_categorical("use_time", [False, True])
    time_weight = trial.suggest_float("time_weight", 1e-3, 1.0, log=True) if use_time else 0.0
    k = trial.suggest_int("k", 1, 32, log=True)
    temperature = trial.suggest_float("temperature", 0.02, 5.0, log=True)
    one_per_episode = trial.suggest_categorical("one_per_episode", [True, False])
    sticky = trial.suggest_categorical("use_consistency", [False, True])
    consistency = trial.suggest_float("consistency_weight", 0.05, 0.95) if sticky else 0.0
    window = trial.suggest_int("consistency_window", 1, 16, log=True) if sticky else 4
    gripper_mode = trial.suggest_categorical("gripper_mode", ["sign", "mean"])
    chunk = trial.suggest_categorical("chunk_size", [4, 8, 16, 32])
    ens = trial.suggest_categorical("temporal_ensemble", [False, True])
    ens_coeff = trial.suggest_float("ensemble_coeff", 0.0, 0.5) if ens else None
    n_act = 1 if ens else trial.suggest_int("n_action_steps", 1, chunk, log=True)
    stride = trial.suggest_categorical("memory_stride", [1, 2, 4])
    return KNNConfig(
        device="cuda",
        image_encoder=enc,
        image_pooling=pooling,
        camera_weights={AGENT_VIEW: w_agent, WRIST: w_wrist},
        state_weight=state_w,
        time_weight=time_weight,
        k=k,
        temperature=temperature,
        one_per_episode=one_per_episode,
        consistency_weight=consistency,
        consistency_window=window,
        gripper_mode=gripper_mode,
        chunk_size=chunk,
        n_action_steps=n_act,
        temporal_ensemble_coeff=ens_coeff,
        memory_stride=stride,
    )


def sample_config_v2(trial: optuna.Trial) -> KNNConfig:
    """v2: CNN features only (v1 showed ImageNet ResNet18 >> DINOv2/v3 >> pixels), plus history matching,
    norm-preserving averaging, distance normalisation and soft (embedding-distance) language conditioning."""
    encoders = {"resnet18": ("cls", "grid2"), "resnet18_l3": ("grid2", "grid4")}
    enc = trial.suggest_categorical("image_encoder", list(encoders))
    pooling = trial.suggest_categorical(f"pooling_{enc}", list(encoders[enc]))
    w_agent = trial.suggest_float("w_agentview", 0.0, 1.0)
    w_wrist = trial.suggest_float("w_wrist", 0.0, 1.0)
    if w_agent + w_wrist < 0.05:
        w_agent = 0.05
    state_w = trial.suggest_float("state_weight", 1e-2, 3.0, log=True)
    normalization = trial.suggest_categorical("distance_normalization", ["pair_median", "nn_median"])
    conditioning = trial.suggest_categorical("task_conditioning", ["filter", "soft"])
    text_weight = trial.suggest_float("text_weight", 0.1, 30.0, log=True) if conditioning == "soft" else 1.0
    use_time = trial.suggest_categorical("use_time", [False, True])
    time_weight = trial.suggest_float("time_weight", 1e-3, 1.0, log=True) if use_time else 0.0
    use_history = trial.suggest_categorical("use_history", [False, True])
    history_weight = trial.suggest_float("history_weight", 0.05, 2.0, log=True) if use_history else 0.0
    k = trial.suggest_int("k", 1, 32, log=True)
    temperature = trial.suggest_float("temperature", 0.02, 5.0, log=True)
    one_per_episode = trial.suggest_categorical("one_per_episode", [True, False])
    norm_preserving = trial.suggest_categorical("norm_preserving", [False, True])
    use_mode = trial.suggest_categorical("use_action_mode", [False, True])
    mode_bw = trial.suggest_float("action_mode_bandwidth", 0.08, 1.0, log=True) if use_mode else 0.0
    use_ens_mode = trial.suggest_categorical("use_ensemble_mode", [False, True])
    ens_mode_bw = trial.suggest_float("ensemble_mode_bandwidth", 0.08, 1.0, log=True) if use_ens_mode else 0.0
    sticky = trial.suggest_categorical("use_consistency", [False, True])
    consistency = trial.suggest_float("consistency_weight", 0.05, 0.95) if sticky else 0.0
    window = trial.suggest_int("consistency_window", 1, 16, log=True) if sticky else 4
    gripper_mode = trial.suggest_categorical("gripper_mode", ["sign", "mean"])
    chunk = trial.suggest_categorical("chunk_size", [4, 8, 16, 32])
    ens = trial.suggest_categorical("temporal_ensemble", [False, True])
    ens_coeff = trial.suggest_float("ensemble_coeff", -0.3, 0.6) if ens else None
    n_act = 1 if ens else trial.suggest_int("n_action_steps", 1, chunk, log=True)
    stride = trial.suggest_categorical("memory_stride", [1, 2])
    return KNNConfig(
        device="cuda",
        image_encoder=enc,
        image_pooling=pooling,
        camera_weights={AGENT_VIEW: w_agent, WRIST: w_wrist},
        state_weight=state_w,
        distance_normalization=normalization,
        task_conditioning=conditioning,
        text_weight=text_weight,
        time_weight=time_weight,
        history_weight=history_weight,
        k=k,
        temperature=temperature,
        one_per_episode=one_per_episode,
        norm_preserving=norm_preserving,
        action_mode_bandwidth=mode_bw,
        ensemble_mode_bandwidth=ens_mode_bw,
        consistency_weight=consistency,
        consistency_window=window,
        gripper_mode=gripper_mode,
        chunk_size=chunk,
        n_action_steps=n_act,
        temporal_ensemble_coeff=ens_coeff,
        memory_stride=stride,
    )


def sample_config_v3(trial: optuna.Trial) -> KNNConfig:
    """v3 (per-suite studies): centred on what won v2 — ResNet features, temporal ensembling, mode-seeking,
    history and soft/hard prompt conditioning — plus test-time augmentation and the gripper-finger term."""

    def optional(name: str, lo: float, hi: float) -> float:
        return (
            trial.suggest_float(name, lo, hi, log=True)
            if trial.suggest_categorical(f"use_{name}", [False, True])
            else 0.0
        )

    encoders = {"resnet18": ("grid2",), "resnet18_l3": ("grid2", "grid4")}
    enc = trial.suggest_categorical("image_encoder", list(encoders))
    pooling = trial.suggest_categorical(f"pooling_{enc}", list(encoders[enc]))
    conditioning = trial.suggest_categorical("task_conditioning", ["filter", "soft"])
    chunk = trial.suggest_categorical("chunk_size", [8, 16])
    return KNNConfig(
        device="cuda",
        image_encoder=enc,
        image_pooling=pooling,
        camera_weights={
            AGENT_VIEW: trial.suggest_float("w_agentview", 0.1, 1.0),
            WRIST: trial.suggest_float("w_wrist", 0.1, 1.0),
        },
        state_weight=trial.suggest_float("state_weight", 5e-3, 1.0, log=True),
        gripper_weight=optional("gripper_weight", 0.02, 1.0),
        distance_normalization=trial.suggest_categorical(
            "distance_normalization", ["pair_median", "nn_median"]
        ),
        task_conditioning=conditioning,
        text_weight=trial.suggest_float("text_weight", 0.2, 5.0, log=True) if conditioning == "soft" else 1.0,
        time_weight=optional("time_weight", 1e-3, 0.1),
        history_weight=(hw := optional("history_weight", 0.1, 3.0)),
        history_gap=trial.suggest_categorical("history_gap", [0, 4, 8, 16]) if hw > 0 else 0,
        k=trial.suggest_int("k", 4, 32, log=True),
        temperature=trial.suggest_float("temperature", 0.03, 1.5, log=True),
        one_per_episode=trial.suggest_categorical("one_per_episode", [True, False]),
        action_mode_bandwidth=optional("action_mode_bandwidth", 0.15, 1.0),
        ensemble_mode_bandwidth=optional("ensemble_mode_bandwidth", 0.15, 1.0),
        gripper_mode=trial.suggest_categorical("gripper_mode", ["mean", "sign"]),
        chunk_size=chunk,
        n_action_steps=1,
        temporal_ensemble_coeff=trial.suggest_float("ensemble_coeff", -0.2, 0.8),
        memory_stride=trial.suggest_categorical("memory_stride", [1, 2]),
        query_tta=trial.suggest_categorical("query_tta", [0, 2, 4]),
    )


SEARCH_SPACES = {"v1": sample_config_v1, "v2": sample_config_v2, "v3": sample_config_v3}

# the KNNConfig fields a search may set: what a trial stores as its configuration
SEARCHED_FIELDS = (
    "image_encoder",
    "image_pooling",
    "camera_weights",
    "state_weight",
    "gripper_weight",
    "query_tta",
    "distance_normalization",
    "task_conditioning",
    "text_weight",
    "time_weight",
    "k",
    "temperature",
    "one_per_episode",
    "consistency_weight",
    "consistency_window",
    "history_weight",
    "history_gap",
    "norm_preserving",
    "action_mode_bandwidth",
    "ensemble_mode_bandwidth",
    "gripper_mode",
    "chunk_size",
    "n_action_steps",
    "temporal_ensemble_coeff",
    "memory_stride",
)


def config_dict(cfg: KNNConfig) -> dict[str, Any]:
    return {k: getattr(cfg, k) for k in SEARCHED_FIELDS}


def run_eval(
    factory: MemoryFactory,
    pool: LiberoPool,
    cfg: KNNConfig,
    jobs: list[Job],
    early_stop=None,
    progress: bool = False,
    episodes: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[EpisodeResult]]:
    """Build the policy of ``cfg`` (memory = ``episodes``) and run ``jobs``: (summary, per-episode results)."""
    policy = factory.build(cfg, episodes=episodes)
    pre, post = make_knn_pre_post_processors(cfg)
    t0 = time.time()
    results = pool.evaluate(policy, pre, post, jobs, progress=progress, early_stop=early_stop)
    summary = summarize(results)
    summary["eval_s"] = time.time() - t0
    del policy
    torch.cuda.empty_cache()
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="knn_v1")
    parser.add_argument("--space", default="v1")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--init_states", default="40:50")
    parser.add_argument("--stages", type=int, default=2, help="split init states into stages for pruning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enqueue", default=None, help="JSON file with a list of param dicts to run first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("lerobot_policy_knn.modeling_knn").setLevel(logging.WARNING)

    # the initial states are split into stages: a trial below the median after a stage is pruned
    states = list(parse_range(args.init_states))
    stage_jobs = [
        make_jobs(args.suites.split(","), init_states=states[i :: args.stages]) for i in range(args.stages)
    ]
    KNN_OUT.mkdir(parents=True, exist_ok=True)
    sampler_cls = optuna.samplers.TPESampler  # spellchecker:disable-line
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{KNN_OUT / 'optuna.db'}",
        direction="maximize",
        sampler=sampler_cls(seed=args.seed, multivariate=True, group=True, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0),
        load_if_exists=True,
    )
    study.set_user_attr("suites", args.suites)
    study.set_user_attr("init_states", args.init_states)
    study.set_user_attr("n_episodes", sum(len(j) for j in stage_jobs))
    study.set_user_attr("search_space", args.space)
    if args.enqueue:
        for params in json.loads(Path(args.enqueue).read_text()):
            study.enqueue_trial(params, skip_if_exists=True)

    factory = MemoryFactory()
    log_path = KNN_OUT / f"trials_{args.study}.jsonl"
    episodes_path = KNN_OUT / f"episodes_{args.study}.jsonl"
    with LiberoPool(n_workers=args.workers) as pool:

        def objective(trial: optuna.Trial) -> float:
            cfg = sample_config(trial, args.space)
            results: list[EpisodeResult] = []
            t0 = time.time()
            summary: dict[str, Any] = {}
            for stage, jobs in enumerate(stage_jobs):
                _, res = run_eval(factory, pool, cfg, jobs)
                results.extend(res)
                summary = summarize(results)
                trial.report(summary["success"], stage)
                if stage < len(stage_jobs) - 1 and trial.should_prune():
                    trial.set_user_attr("pruned_after_episodes", len(results))
                    logger.info(
                        "trial %d pruned at %.3f after %d episodes",
                        trial.number,
                        summary["success"],
                        len(results),
                    )
                    raise optuna.TrialPruned()
            summary["eval_s"] = time.time() - t0
            for key, value in summary.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("config", json.dumps(config_dict(cfg)))
            append_jsonl(log_path, [{"trial": trial.number, "params": trial.params, **summary}])
            append_jsonl(episodes_path, ({"trial": trial.number, **vars(r)} for r in results))
            logger.info(
                "trial %d: success %.3f (%s) in %.0fs",
                trial.number,
                summary["success"],
                suite_scores(summary, SUITES),
                summary["eval_s"],
            )
            return summary["success"]

        stop_file = KNN_OUT / f"STOP_{args.study}"

        def stop_if_requested(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
            # `touch outputs/knn/STOP_<study>` ends the search cleanly after the current trial
            if stop_file.exists():
                stop_file.unlink()
                study.stop()

        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True, callbacks=[stop_if_requested])


if __name__ == "__main__":
    main()
