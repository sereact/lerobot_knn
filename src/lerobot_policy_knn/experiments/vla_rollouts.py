"""Closed-loop LIBERO evaluation of a VLA in the parallel pool, recording its action chunks.

    uv run python -m lerobot_policy_knn.experiments.vla_rollouts smolvla_libero --init_states=20:30 --record=20:22

Appends the success rates to outputs/vla/results.jsonl and, for the ``--record`` initial states, saves every
re-planning observation (uint8 images, state, prompt) with the chunk the VLA predicted to
outputs/vla/rollouts/<name>.npz, the input of ``head_vs_knn``.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..vla import Identity, VLARollout, cache_tag, load_vla, resolve_spec
from .common import VLA_OUT, append_jsonl, parse_range, suite_scores
from .libero_pool import SUITES, LiberoPool, make_jobs, summarize


def build_runner(
    name: str, n_action_steps: int = 10, num_steps: int | None = None, record: set[int] | None = None
) -> tuple[VLARollout, Any, str]:
    """The pool-side runner of a VLA, the LIBERO observation processor it needs instead of the default (or None),
    and its LIBERO control mode."""
    policy, pre, post = load_vla(name)
    if num_steps is not None:
        for attr in ("num_steps", "num_inference_steps"):
            if hasattr(policy.config, attr):
                setattr(policy.config, attr, num_steps)
    env_pre, action_transform, control_mode = None, None, "relative"
    if policy.config.type == "xvla":
        # X-VLA ships its own LIBERO conventions: absolute end-effector actions as 6D rotations (its observation
        # steps live in the preprocessor, see vla.vla_processors)
        from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep
        from lerobot.processor import TransitionKey

        to_axis_angle = XVLARotation6DToAxisAngleProcessorStep()

        def action_transform(action):
            return to_axis_angle({TransitionKey.ACTION: action})[TransitionKey.ACTION]

        control_mode = "absolute"
    spec = resolve_spec(name)
    if spec.n_action_steps is None:
        steps = n_action_steps
    elif spec.n_action_steps == 0:
        steps = int(policy.config.n_action_steps)
    else:
        steps = spec.n_action_steps
    runner = VLARollout(
        policy,
        pre,
        post,
        n_action_steps=steps,
        record_init_states=record,
        action_transform=action_transform,
        camera_rename=spec.camera_rename,
    )
    return runner, env_pre, control_mode


def save_records(records: list[dict], path: Path, **extra: np.ndarray) -> None:
    """Save the re-planning steps a ``VLARollout`` recorded (one row each) as the ``.npz`` ``head_vs_knn`` reads.

    ``extra``: further per-row arrays to store alongside.
    """
    tasks = sorted({r["task"] for r in records})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        images=np.stack([r["images"] for r in records]),
        state=np.stack([r["state"] for r in records]),
        chunk=np.stack([r["chunk"] for r in records]),
        t=np.array([r["t"] for r in records]),
        task=np.array([tasks.index(r["task"]) for r in records]),
        task_names=np.array(tasks),
        **extra,
        suite=np.array([r["suite"] for r in records]),
        task_id=np.array([r["task_id"] for r in records]),
        init_state=np.array([r["init_state"] for r in records]),
        episode=np.array([r["episode"] for r in records]),
        success=np.array([r["success"] for r in records]),
    )
    print(f"saved {len(records)} chunks -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="VLA registry name or <family>@<checkpoint dir>")
    parser.add_argument("--init_states", default="20:30")
    parser.add_argument("--record", default=None, help="initial states whose chunks are saved, e.g. 20:22")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--n_action_steps", type=int, default=10)
    parser.add_argument(
        "--num_steps", type=int, default=None, help="flow-matching steps (default: checkpoint's)"
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--tag", default="vla")
    parser.add_argument(
        "--observation_size", type=int, default=None, help="render size (default: the VLA's own)"
    )
    args = parser.parse_args()

    record = set(parse_range(args.record)) if args.record else None
    runner, env_pre, control_mode = build_runner(args.name, args.n_action_steps, args.num_steps, record)
    jobs = make_jobs(args.suites.split(","), init_states=parse_range(args.init_states))
    observation_size = args.observation_size or resolve_spec(args.name).observation_size
    t0 = time.time()
    with LiberoPool(args.workers, observation_size=observation_size, control_mode=control_mode) as pool:
        pool.env_preprocessor_override = env_pre
        results = pool.evaluate(runner, Identity(), Identity(), jobs, progress=True)
    summary = summarize(results)
    if runner.failed_predictions:
        print(f"{runner.failed_predictions} predictions failed and held still", flush=True)
    row = {
        "kind": "vla",
        "tag": args.tag,
        "name": args.name,
        "init_states": args.init_states,
        "failed_predictions": runner.failed_predictions,
        "n_action_steps": args.n_action_steps,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_s": time.time() - t0,
        **{k: v for k, v in summary.items() if "/" not in k},
    }
    append_jsonl(VLA_OUT / "results.jsonl", [row])
    append_jsonl(
        VLA_OUT / "results_episodes.jsonl", ({"tag": args.tag, "name": args.name, **vars(r)} for r in results)
    )
    print(
        f"{args.name}: {summary['success']:.4f} ({suite_scores(summary, SUITES)}) {row['eval_s']:.0f}s",
        flush=True,
    )
    if runner.records:
        save_records(runner.records, VLA_OUT / "rollouts" / f"{cache_tag(args.name)}.npz")


if __name__ == "__main__":
    main()
