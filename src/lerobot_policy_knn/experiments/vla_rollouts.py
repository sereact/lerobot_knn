"""Closed-loop LIBERO evaluation of a VLA in the parallel pool, recording its action chunks.

    uv run python -m lerobot_policy_knn.experiments.vla_rollouts smolvla_libero --init_states=20:30 --record=20:22

Appends the success rates to outputs/vla/results.jsonl and, for the ``--record`` initial states, saves every
re-planning observation (uint8 images, state, prompt) with the chunk the VLA predicted to
outputs/vla/rollouts/<name>.npz, the input of ``head_vs_knn``.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..vla import Identity, VLARollout, cache_tag, load_vla, resolve_spec
from .libero_pool import SUITES, LiberoPool, make_jobs, summarize

OUT = Path("outputs/vla")


def _range(spec: str) -> range:
    lo, hi = (int(x) for x in spec.split(":"))
    return range(lo, hi)


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
    # X-VLA ships its own LIBERO conventions: observations, absolute EE actions
    if (
        policy.config.type == "xvla"
    ):  # absolute EE actions as 6D rotations (its image steps live in `pre`, vla.py)
        from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep
        from lerobot.processor import TransitionKey

        rot = XVLARotation6DToAxisAngleProcessorStep()
        action_transform = lambda a: rot({TransitionKey.ACTION: a})[TransitionKey.ACTION]  # noqa: E731
        control_mode = "absolute"
    spec = resolve_spec(name)
    steps = {None: n_action_steps, 0: int(policy.config.n_action_steps)}.get(
        spec.n_action_steps, spec.n_action_steps
    )
    runner = VLARollout(policy, pre, post, n_action_steps=steps,
                        record_init_states=record, action_transform=action_transform,
                        camera_rename=spec.camera_rename)  # fmt: skip
    return runner, env_pre, control_mode


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

    record = set(_range(args.record)) if args.record else None
    runner, env_pre, control_mode = build_runner(args.name, args.n_action_steps, args.num_steps, record)
    jobs = make_jobs(args.suites.split(","), init_states=_range(args.init_states))
    t0 = time.time()
    with LiberoPool(n_workers=args.workers, control_mode=control_mode,
                    observation_size=args.observation_size or resolve_spec(args.name).observation_size) as pool:  # fmt: skip
        pool.env_preprocessor_override = env_pre
        results = pool.evaluate(runner, Identity(), Identity(), jobs, progress=True)
    summary = summarize(results)
    if runner.failed_predictions:
        print(f"{runner.failed_predictions} predictions failed and held still", flush=True)
    row = {"kind": "vla", "tag": args.tag, "name": args.name, "init_states": args.init_states,
           "failed_predictions": runner.failed_predictions,
           "n_action_steps": args.n_action_steps, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "eval_s": time.time() - t0, **{k: v for k, v in summary.items() if "/" not in k}}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    with (OUT / "results_episodes.jsonl").open("a") as f:
        for r in results:
            f.write(json.dumps({"tag": args.tag, "name": args.name, **r.__dict__}) + "\n")
    suites = " ".join(f"{s.split('_')[1]}={summary[s]:.3f}" for s in SUITES if s in summary)
    print(f"{args.name}: {summary['success']:.4f} ({suites}) {row['eval_s']:.0f}s", flush=True)

    if runner.records:
        recs = runner.records
        tasks = sorted({r["task"] for r in recs})
        (OUT / "rollouts").mkdir(parents=True, exist_ok=True)
        np.savez(
            OUT / "rollouts" / f"{cache_tag(args.name)}.npz",
            images=np.stack([r["images"] for r in recs]),
            state=np.stack([r["state"] for r in recs]),
            chunk=np.stack([r["chunk"] for r in recs]),
            t=np.array([r["t"] for r in recs]),
            task=np.array([tasks.index(r["task"]) for r in recs]),
            task_names=np.array(tasks),
            suite=np.array([r["suite"] for r in recs]),
            task_id=np.array([r["task_id"] for r in recs]),
            init_state=np.array([r["init_state"] for r in recs]),
            episode=np.array([r["episode"] for r in recs]),
            success=np.array([r["success"] for r in recs]),
        )
        print(f"saved {len(recs)} chunks", flush=True)


if __name__ == "__main__":
    main()
