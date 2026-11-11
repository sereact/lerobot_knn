"""Evaluate LIBERO-trained policies on LIBERO-plus, every policy on the same stratified sample of perturbed tasks.

LIBERO-plus (https://github.com/sylvestf/LIBERO-plus) rebuilds the four LIBERO suites as ~10k task variants in seven
perturbation categories (camera viewpoint, robot initial state, language, light, background, sensor noise, object
layout), scored with one trial each. ``--sample`` fixes ``--per_cell`` variants per (suite, category), drawn once with
seed 0, so all policies see the same episodes. A policy is ``knn:<KNNConfig overrides .json>`` (the lookup, memory =
the LIBERO demonstrations) or ``vla:<registry name | family@checkpoint>``. Runs in the LIBERO-plus venv:

    LIBERO_CONFIG_PATH=outputs/libero_plus/libero_config MUJOCO_GL=egl outputs/libero_plus/venv/bin/python \\
        -m lerobot_policy_knn.experiments.libero_plus_eval knn:outputs/knn/v2_39.json vla:pi05_libero

Rows go to outputs/libero_plus/results.jsonl (success per category, suite and difficulty level) and
results_episodes.jsonl.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..configuration_knn import KNNConfig
from ..features import load_frame_index
from ..vla import Identity, cache_tag, resolve_spec
from .common import PLUS_OUT, REPO_ID, append_jsonl
from .head_vs_knn import base_task
from .libero_pool import SUITES, EpisodeResult, Job, LiberoPool
from .search import MemoryFactory, run_eval
from .vla_rollouts import build_runner, save_records

Variant = dict  # one LIBERO-plus task variant: suite, task_id (in the suite), name, category, level


def make_sample(per_cell: int, seed: int = 0) -> list[Variant]:
    """``per_cell`` random variants of every (suite, perturbation category), with their task ids in the suite."""
    from libero.libero import benchmark

    from lerobot.envs.libero import _get_suite

    classes = json.loads((Path(benchmark.__file__).parent / "task_classification.json").read_text())
    rng = np.random.default_rng(seed)
    sample = []
    for suite in SUITES:
        task_id = {name: i for i, name in enumerate(_get_suite(suite).get_task_names())}
        by_category = defaultdict(list)
        for entry in classes[suite]:
            by_category[entry["category"]].append(entry)
        for category in sorted(by_category):
            entries = by_category[category]
            for entry in rng.choice(entries, min(per_cell, len(entries)), replace=False):
                sample.append(
                    {
                        "suite": suite,
                        "task_id": task_id[entry["name"]],
                        "name": entry["name"],
                        "category": category,
                        "level": entry["difficulty_level"],
                    }
                )
    return sample


def by_task(sample: list[Variant]) -> dict[tuple[str, int], Variant]:
    return {(v["suite"], v["task_id"]): v for v in sample}


def first_per_cell(sample: list[Variant], n: int) -> list[Variant]:
    """The first ``n`` variants of every (suite, category)."""
    seen: dict[tuple[str, str], int] = defaultdict(int)
    out = []
    for v in sample:
        cell = (v["suite"], v["category"])
        if seen[cell] < n:
            seen[cell] += 1
            out.append(v)
    return out


def original_instructions(sample: list[Variant]) -> dict[tuple[str, int], str]:
    """(suite, task_id) -> the instruction of the LIBERO task a LIBERO-plus variant was derived from."""
    tasks = load_frame_index(REPO_ID).task_names
    return {key: base_task(v["name"], tasks) for key, v in by_task(sample).items()}


def save_plus_records(records: list[dict], sample: list[Variant], path: Path) -> None:
    """Recorded re-planning steps as ``vla_rollouts --record`` saves them, plus each variant's perturbation category
    and the instruction of its LIBERO base task (``base_task_names``, so reworded prompts map to the right memory)."""
    variant = by_task(sample)
    base = original_instructions(sample)
    keys = [(r["suite"], r["task_id"]) for r in records]
    save_records(
        records,
        path,
        base_task_names=np.array([base[key] for key in keys]),
        category=np.array([variant[key]["category"] for key in keys]),
    )


def breakdown(sample: list[Variant], results: list[EpisodeResult]) -> dict[str, float]:
    """Success rate overall (``all``) and per perturbation category, suite and difficulty level."""
    variant = by_task(sample)
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        v = variant[(r.suite, r.task_id)]
        for key in ("all", f"category/{v['category']}", f"suite/{r.suite}", f"level/{v['level']}"):
            groups[key].append(r.success)
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="+", help="knn:<config.json> or vla:<name>")
    parser.add_argument("--per_cell", type=int, default=15)
    parser.add_argument("--sample", default=str(PLUS_OUT / "sample.json"))
    parser.add_argument("--n_action_steps", type=int, default=10, help="VLA re-planning interval")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--tag", default=None, help="default: libero_plus (libero_plus_original_instruction)")
    parser.add_argument(
        "--observation_size",
        type=int,
        default=None,
        help="render size (default: the VLA's own; LIBERO-plus fog needs 256)",
    )
    parser.add_argument(
        "--categories", default=None, help="comma-separated subset of perturbation categories"
    )
    parser.add_argument(
        "--suites", default=None, help="comma-separated subset of suites (per-suite checkpoints)"
    )
    parser.add_argument(
        "--original_instruction",
        action="store_true",
        help="give the policy each variant's original LIBERO instruction (a control: language variants keep the "
        "original scene, so with their original instruction they are clean LIBERO inside the LIBERO-plus env)",
    )
    parser.add_argument(
        "--record_per_cell",
        type=int,
        default=None,
        help="VLAs only: run the first N variants of every (suite, category) and save every re-planning observation "
        "with the chunk predicted there to outputs/libero_plus/rollouts/<name>.npz (input of head_vs_knn)",
    )
    args = parser.parse_args()
    path = Path(args.sample)
    if path.exists():
        sample = json.loads(path.read_text())
    else:
        sample = make_sample(args.per_cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sample, indent=1))
    # every variant keeps the seed it has in the full sample, whatever subset runs
    seeds = {(v["suite"], v["task_id"]): i for i, v in enumerate(sample)}
    if args.categories:
        keep = set(args.categories.split(","))
        sample = [v for v in sample if v["category"] in keep]
    if args.suites:
        keep = set(args.suites.split(","))
        sample = [v for v in sample if v["suite"] in keep]
    if args.record_per_cell:
        sample = first_per_cell(sample, args.record_per_cell)
    if args.tag is None:
        if args.original_instruction:
            args.tag = "libero_plus_original_instruction"
        else:
            args.tag = "libero_plus_record" if args.record_per_cell else "libero_plus"
    # one trial per variant, from its first initial state (the LIBERO-plus protocol)
    jobs = [Job(v["suite"], v["task_id"], 0, seed=seeds[(v["suite"], v["task_id"])]) for v in sample]
    variant = by_task(sample)

    # the render size the VLAs' evaluation uses, e.g. 448 px for Evo-1
    sizes = {resolve_spec(p.split(":", 1)[1]).observation_size for p in args.policies if p.startswith("vla:")}
    if args.observation_size:  # LIBERO-plus's fog noise assumes 256 px renders
        sizes = {args.observation_size}
    if len(sizes) > 1:
        raise ValueError(f"policies with different render sizes {sizes}: evaluate them separately")
    observation_size = sizes.pop() if sizes else 256

    factory = None
    with LiberoPool(n_workers=args.workers, libero_plus=True, observation_size=observation_size) as pool:
        if args.original_instruction:
            pool.task_override = original_instructions(sample)
        for policy_id in args.policies:
            kind, name = policy_id.split(":", 1)
            t0 = time.time()
            if kind == "knn":
                factory = factory or MemoryFactory()
                config = json.loads(Path(name).read_text())
                overrides = {k: v for k, v in config.items() if not k.startswith("_")}
                _, results = run_eval(
                    factory, pool, KNNConfig(device="cuda", **overrides), jobs, progress=True
                )
                failed = None
            else:
                record = {0} if args.record_per_cell else None  # every LIBERO-plus trial starts from state 0
                runner, env_pre, control_mode = build_runner(name, args.n_action_steps, record=record)
                pool.env_preprocessor_override = env_pre
                pool.set_control_mode(control_mode)
                results = pool.evaluate(runner, Identity(), Identity(), jobs, progress=True)
                pool.env_preprocessor_override = None
                pool.set_control_mode("relative")
                failed = runner.failed_predictions
                if runner.records:
                    save_plus_records(
                        runner.records, sample, PLUS_OUT / "rollouts" / f"{cache_tag(name)}.npz"
                    )
                del runner
            stats = breakdown(sample, results)
            row = {
                "tag": args.tag,
                "policy": policy_id,
                "n_episodes": len(results),
                "failed_predictions": failed,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "eval_s": time.time() - t0,
                **stats,
            }
            append_jsonl(PLUS_OUT / "results.jsonl", [row])
            append_jsonl(
                PLUS_OUT / "results_episodes.jsonl",
                (
                    {
                        "tag": args.tag,
                        "policy": policy_id,
                        "category": variant[(r.suite, r.task_id)]["category"],
                        "level": variant[(r.suite, r.task_id)]["level"],
                        **vars(r),
                    }
                    for r in results
                ),
            )
            categories = " ".join(
                f"{k.split('/')[1].split()[0]}={v:.2f}" for k, v in stats.items() if k.startswith("category/")
            )
            print(f"{policy_id}: {stats['all']:.4f} ({categories}) {row['eval_s']:.0f}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
