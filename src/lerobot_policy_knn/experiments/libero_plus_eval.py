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
from ..vla import Identity, cache_tag
from .libero_pool import SUITES, Job, LiberoPool
from .search import MemoryFactory, run_eval
from .vla_rollouts import build_runner

OUT = Path("outputs/libero_plus")


def make_sample(per_cell: int, seed: int = 0) -> list[dict]:
    """``per_cell`` random variants of every (suite, perturbation category), with their task ids in the suite."""
    from libero.libero import benchmark

    from lerobot.envs.libero import _get_suite

    classes = json.loads((Path(benchmark.__file__).parent / "task_classification.json").read_text())
    rng = np.random.default_rng(seed)
    sample = []
    for suite in SUITES:
        index = {name: i for i, name in enumerate(_get_suite(suite).get_task_names())}
        by_cat = defaultdict(list)
        for entry in classes[suite]:
            by_cat[entry["category"]].append(entry)
        for cat in sorted(by_cat):
            for entry in rng.choice(by_cat[cat], min(per_cell, len(by_cat[cat])), replace=False):
                sample.append({"suite": suite, "task_id": index[entry["name"]], "name": entry["name"],
                               "category": cat, "level": entry["difficulty_level"]})  # fmt: skip
    return sample


def original_instructions(sample: list[dict]) -> dict[tuple[str, int], str]:
    """(suite, task_id) -> the instruction of the LIBERO task a LIBERO-plus variant was derived from."""
    from ..features import load_frame_index
    from .head_vs_knn import base_task

    tasks = load_frame_index("lerobot/libero").task_names
    return {(s["suite"], s["task_id"]): base_task(s["name"], tasks) for s in sample}


def save_records(recs: list[dict], sample: list[dict], path: Path) -> None:
    """Recorded re-planning steps as ``vla_rollouts --record`` saves them, plus each variant's perturbation category
    and the instruction of its LIBERO base task (``base_task_names``, so reworded prompts map to the right memory)."""
    meta = {(s["suite"], s["task_id"]): s for s in sample}
    base = original_instructions(sample)
    tasks = sorted({r["task"] for r in recs})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        images=np.stack([r["images"] for r in recs]),
        state=np.stack([r["state"] for r in recs]),
        chunk=np.stack([r["chunk"] for r in recs]),
        t=np.array([r["t"] for r in recs]),
        task=np.array([tasks.index(r["task"]) for r in recs]),
        task_names=np.array(tasks),
        base_task_names=np.array([base[(r["suite"], r["task_id"])] for r in recs]),
        category=np.array([meta[(r["suite"], r["task_id"])]["category"] for r in recs]),
        suite=np.array([r["suite"] for r in recs]),
        task_id=np.array([r["task_id"] for r in recs]),
        init_state=np.array([r["init_state"] for r in recs]),
        episode=np.array([r["episode"] for r in recs]),
        success=np.array([r["success"] for r in recs]),
    )
    print(f"saved {len(recs)} chunks -> {path}", flush=True)


def breakdown(sample: list[dict], results) -> dict:
    meta = {(s["suite"], s["task_id"]): s for s in sample}
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        m = meta[(r.suite, r.task_id)]
        for key in ("all", f"category/{m['category']}", f"suite/{r.suite}", f"level/{m['level']}"):
            groups[key].append(r.success)
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="+", help="knn:<config.json> or vla:<name>")
    parser.add_argument("--per_cell", type=int, default=15)
    parser.add_argument("--sample", default=str(OUT / "sample.json"))
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
    # one trial per variant, from its first initial state (the LIBERO-plus protocol)
    seeds = {(s["suite"], s["task_id"]): i for i, s in enumerate(sample)}  # the main run's seeds
    if args.categories:
        keep = set(args.categories.split(","))
        sample = [s for s in sample if s["category"] in keep]
    if args.suites:
        keep = set(args.suites.split(","))
        sample = [s for s in sample if s["suite"] in keep]
    if args.record_per_cell:
        seen: dict[tuple[str, str], int] = defaultdict(int)
        sub = []
        for s in sample:
            if seen[(s["suite"], s["category"])] < args.record_per_cell:
                seen[(s["suite"], s["category"])] += 1
                sub.append(s)
        sample = sub
    default_tag = "libero_plus_record" if args.record_per_cell else "libero_plus"
    args.tag = args.tag or ("libero_plus_original_instruction" if args.original_instruction else default_tag)
    jobs = [Job(s["suite"], s["task_id"], 0, seed=seeds[(s["suite"], s["task_id"])]) for s in sample]
    factory = None
    # the render size the (first) VLA's evaluation uses, e.g. 448 px for Evo-1
    from ..vla import resolve_spec

    sizes = {resolve_spec(p.split(":", 1)[1]).observation_size for p in args.policies if p.startswith("vla:")}
    if args.observation_size:  # LIBERO-plus's fog noise assumes 256 px renders
        sizes = {args.observation_size}
    if len(sizes) > 1:
        raise ValueError(f"policies with different render sizes {sizes}: evaluate them separately")
    with LiberoPool(
        n_workers=args.workers, libero_plus=True, observation_size=sizes.pop() if sizes else 256
    ) as pool:
        if args.original_instruction:
            pool.task_override = original_instructions(sample)
        for spec in args.policies:
            kind, name = spec.split(":", 1)
            t0 = time.time()
            if kind == "knn":
                factory = factory or MemoryFactory()
                overrides = {
                    k: v for k, v in json.loads(Path(name).read_text()).items() if not k.startswith("_")
                }
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
                    save_records(runner.records, sample, OUT / "rollouts" / f"{cache_tag(name)}.npz")
                del runner
            stats = breakdown(sample, results)
            row = {"tag": args.tag, "policy": spec, "n_episodes": len(results), "failed_predictions": failed,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), "eval_s": time.time() - t0, **stats}  # fmt: skip
            OUT.mkdir(parents=True, exist_ok=True)
            with (OUT / "results.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            meta = {(s["suite"], s["task_id"]): s for s in sample}
            with (OUT / "results_episodes.jsonl").open("a") as f:
                for r in results:
                    m = meta[(r.suite, r.task_id)]
                    f.write(json.dumps({"tag": args.tag, "policy": spec, "category": m["category"],
                                        "level": m["level"], **r.__dict__}) + "\n")  # fmt: skip
            cats = " ".join(
                f"{k.split('/')[1].split()[0]}={v:.2f}" for k, v in stats.items() if k.startswith("category/")
            )
            print(f"{spec}: {stats['all']:.4f} ({cats}) {row['eval_s']:.0f}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
