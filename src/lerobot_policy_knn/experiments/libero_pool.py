"""Fast, asynchronous LIBERO evaluation with a persistent pool of simulator processes.

The semantics match ``lerobot-eval --env.type=libero``: the same ``LiberoEnv`` (hard resets, the fixed
LIBERO initial states, 10 settling steps), the same env/policy processors, the same per-suite step limit,
and success = ``env.check_success()``; episode ``i`` of a task uses initial state ``i``.

What differs is scheduling: every (task, initial state) episode is a job for a fixed pool of worker
processes, and the policy is queried on whichever workers are waiting for an action, keeping per-env
policy state keyed by worker id (``select_action(..., env_ids=...)``). No worker idles while others
finish, and the pool survives across evaluations (e.g. hyper-parameter search trials).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class Job:
    """One episode to run: a task of a suite from one of its fixed initial states."""

    suite: str
    task_id: int
    init_state: int
    seed: int = 0


@dataclass
class EpisodeResult:
    suite: str
    task_id: int
    init_state: int
    task: str
    success: bool
    steps: int
    wall_s: float


def make_jobs(
    suites: tuple[str, ...] | list[str] = SUITES,
    init_states: range | list[int] = range(10),
    task_ids: list[int] | None = None,
    seed: int = 1000,
) -> list[Job]:
    jobs = []
    for suite in suites:
        for task_id in task_ids if task_ids is not None else range(10):
            for i in init_states:
                jobs.append(Job(suite, task_id, i, seed + i))
    # interleave suites so long-horizon libero_10 episodes are not all at the end
    jobs.sort(key=lambda j: (j.init_state, SUITES.index(j.suite) if j.suite in SUITES else 0, j.task_id))
    return jobs


_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MUJOCO_GL": "egl",
}

# Measured on an RTX 5090 box: one LIBERO worker holds ~0.6-1 GB of VRAM (EGL context) and ~2.2 GB of
# private RAM (unique set size) once warm. Budgets include a safety margin.
WORKER_VRAM_GB = 1.0
WORKER_RAM_GB = 2.6
# Abort an evaluation (rather than let the OOM killer pick someone else's process) below this.
EMERGENCY_FREE_RAM_GB = 8.0


class ResourceError(Exception):
    pass


def _available_ram_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 1e9


def check_resources(n_workers: int, min_free_vram_gb: float = 4.0, min_free_ram_gb: float = 16.0) -> int:
    """Largest pool size (<= n_workers) that leaves the requested VRAM/RAM headroom on this machine."""
    ram_free = _available_ram_gb()
    by_ram = int((ram_free - min_free_ram_gb) / WORKER_RAM_GB)
    by_vram = n_workers
    if torch.cuda.is_available():
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        by_vram = int((vram_free - min_free_vram_gb) / WORKER_VRAM_GB)
    allowed = max(0, min(n_workers, by_ram, by_vram))
    if allowed < n_workers:
        logger.warning(
            "Reducing LIBERO pool from %d to %d workers (free RAM %.1f GB, headroom %.0f/%.0f GB VRAM/RAM)",
            n_workers,
            allowed,
            ram_free,
            min_free_vram_gb,
            min_free_ram_gb,
        )
    if allowed == 0:
        raise RuntimeError("Not enough free GPU/CPU memory to start any LIBERO worker")
    return allowed


def _worker_main(conn: Connection, env_kwargs: dict[str, Any]) -> None:
    """A simulator process: builds the env of the task it is asked for and serves ``reset`` / ``step`` commands."""
    import warnings

    # If memory ever runs out, make the kernel OOM killer take a simulator worker, not a user app.
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    torch.set_num_threads(1)

    warnings.filterwarnings("ignore")
    logging.getLogger("robosuite_logs").setLevel(logging.ERROR)
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suites: dict[str, Any] = {}
    env: LiberoEnv | None = None
    current: tuple[str, int] | None = None
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "step":
                t0 = time.perf_counter()
                obs, _reward, terminated, _truncated, info = env.step(payload)  # type: ignore[union-attr]
                dt = time.perf_counter() - t0
                conn.send(("step", obs, bool(terminated), bool(info.get("is_success", False)), dt))
            elif cmd == "reset":
                job: Job = payload
                if current != (job.suite, job.task_id) or env is None:
                    if env is not None:
                        env.close()
                    if job.suite not in suites:
                        suites[job.suite] = _get_suite(job.suite)
                    env = LiberoEnv(
                        suites[job.suite],
                        job.task_id,
                        job.suite,
                        episode_index=job.init_state,
                        n_envs=1,
                        **env_kwargs,
                    )
                    current = (job.suite, job.task_id)
                env.init_state_id = job.init_state
                obs, _info = env.reset(seed=job.seed)
                conn.send(("reset", obs, env.task_description, env._max_episode_steps))
            elif cmd == "env_kwargs":  # e.g. another control mode: rebuild the env on the next reset
                env_kwargs = {**env_kwargs, **payload}
                if env is not None:
                    env.close()
                env, current = None, None
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if env is not None:
            env.close()


def _stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack per-env observation dicts like a gymnasium vector env would."""
    first = obs_list[0]
    out: dict[str, Any] = {}
    for key, value in first.items():
        if isinstance(value, dict):
            out[key] = _stack_obs([o[key] for o in obs_list])
        else:
            out[key] = np.stack([np.asarray(o[key]) for o in obs_list])
    return out


@dataclass
class _Running:
    """The episode a worker is running."""

    job: Job
    task: str = ""
    max_steps: int = 0
    t: int = 0
    start: float = 0.0
    obs: dict[str, Any] | None = None


class LiberoPool:
    """A pool of LIBERO simulator processes evaluating (policy, jobs) pairs."""

    def __init__(
        self,
        n_workers: int = 24,
        observation_size: int = 256,
        camera_name: str = "agentview_image,robot0_eye_in_hand_image",
        control_mode: str = "relative",
        libero_plus: bool = False,
    ):
        self.env_kwargs = {
            "obs_type": "pixels_agent_pos",
            "observation_height": observation_size,
            "observation_width": observation_size,
            "camera_name": camera_name,
            "init_states": True,
            "control_mode": control_mode,
        }
        if libero_plus:  # needs the LIBERO-plus fork in place of `libero` (see outputs/libero_plus/setup.sh)
            self.env_kwargs["is_libero_plus"] = True
        # a policy-specific LIBERO observation processor used instead of the env's default one
        self.env_preprocessor_override = None
        # (suite, task_id) -> the instruction given to the policy instead of the task's own
        self.task_override: dict[tuple[str, int], str] = {}
        # timing counters of the latest evaluate() call, for profiling
        self.last_stats: dict[str, float] = {}
        n_workers = check_resources(n_workers)
        ctx = mp.get_context("spawn")
        self.conns: list[Connection] = []
        self.procs = []
        # children inherit the environment at spawn: pin BLAS/OpenMP pools to one thread per simulator
        saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
        os.environ.update(_SINGLE_THREAD_ENV)
        try:
            for _ in range(n_workers):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(child, self.env_kwargs), daemon=True)
                proc.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(proc)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self._worker_task: list[tuple[str, int] | None] = [None] * len(self.conns)

        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

        self.env_cfg = LiberoEnvConfig(
            observation_height=observation_size, observation_width=observation_size
        )
        self.env_preprocessor, self.env_postprocessor = self.env_cfg.get_env_processors()

    def set_control_mode(self, mode: str) -> None:
        """Switch every simulator between relative (delta) and absolute end-effector actions."""
        if mode != self.env_kwargs["control_mode"]:
            self.env_kwargs["control_mode"] = mode
            for conn in self.conns:
                conn.send(("env_kwargs", {"control_mode": mode}))

    def close(self) -> None:
        for conn in self.conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(("close", None))
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()

    def __enter__(self) -> LiberoPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @torch.no_grad()
    def _policy_step(
        self, policy, preprocessor, postprocessor, obs: list[dict], tasks: list[str], wids: list[int]
    ) -> np.ndarray:
        """The exact ``lerobot-eval`` observation -> action path for a batch of workers."""
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        batch = preprocess_observation(_stack_obs(obs))
        batch["task"] = list(tasks)
        batch = (self.env_preprocessor_override or self.env_preprocessor)(batch)
        batch = preprocessor(batch)
        action = policy.select_action(batch, env_ids=wids)
        action = postprocessor(action)
        action = self.env_postprocessor({ACTION: action})[ACTION]
        return action.to("cpu").numpy()

    def evaluate(
        self,
        policy,
        preprocessor,
        postprocessor,
        jobs: list[Job],
        progress: bool = True,
        early_stop: Any = None,
    ) -> list[EpisodeResult]:
        """Run every job once. ``early_stop(results_so_far) -> bool`` can abort (e.g. optuna pruning).

        The main thread only moves messages: observations that require a policy forward pass are
        batched and handed to an inference thread (one batch in flight; requests arriving meanwhile form
        the next batch), while workers whose policy still has queued actions (``needs_observation``)
        get them immediately without touching the GPU.
        """
        from tqdm import tqdm

        from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
        from lerobot.utils.constants import OBS_STATE

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        # Jobs are grouped per task; a worker keeps drawing episodes of the task its simulator already
        # has loaded (rebuilding an env costs seconds), otherwise it joins the task with the most
        # remaining work per worker, long-horizon suites first.
        queues: dict[tuple[str, int], deque[Job]] = {}
        for job in jobs:
            queues.setdefault((job.suite, job.task_id), deque()).append(job)
        on_task: dict[tuple[str, int], int] = dict.fromkeys(queues, 0)
        running: dict[int, _Running] = {}
        results: list[EpisodeResult] = []
        outstanding: set[int] = set()  # workers currently simulating (reset or step)
        stats = {"wait_s": 0.0, "infer_s": 0.0, "step_s": 0.0}  # seconds
        stats.update(infer_calls=0, infer_rows=0, pops=0, steps=0)  # counts
        self.last_stats = stats
        bar = tqdm(total=len(jobs), disable=not progress, desc="episodes", leave=False)
        can_pop = hasattr(policy, "needs_observation")

        infer_queue: queue.Queue = queue.Queue()
        res_recv, res_send = mp.Pipe(duplex=False)

        def infer_loop() -> None:
            while True:
                item = infer_queue.get()
                if item is None:
                    return
                wids, obs, tasks = item
                t0 = time.perf_counter()
                try:
                    actions = self._policy_step(policy, preprocessor, postprocessor, obs, tasks, wids)
                    res_send.send(("ok", wids, actions, time.perf_counter() - t0))
                except Exception as e:  # surfaced in the main thread
                    res_send.send(("error", wids, repr(e), 0.0))

        thread = threading.Thread(target=infer_loop, daemon=True)
        thread.start()

        def dispatch(w: int) -> None:
            key = self._worker_task[w]
            if key is not None and key in on_task:
                on_task[key] -= 1
            if key is None or not queues.get(key):
                pending = [k for k, q in queues.items() if q]
                if not pending:
                    running.pop(w, None)
                    self._worker_task[w] = None
                    return
                key = max(
                    pending,
                    key=lambda k: len(queues[k]) * TASK_SUITE_MAX_STEPS.get(k[0], 500) / (1 + on_task[k]),
                )
            job = queues[key].popleft()
            on_task[key] += 1
            self._worker_task[w] = key
            running[w] = _Running(job=job, start=time.time())
            self.conns[w].send(("reset", job))
            outstanding.add(w)

        for w in range(len(self.conns)):
            dispatch(w)

        conn_to_worker = {id(c): w for w, c in enumerate(self.conns)}
        pending_infer: list[int] = []
        inflight: list[int] = []
        aborted = False
        last_mem_check = 0.0
        try:
            while outstanding or pending_infer or inflight:
                if time.time() - last_mem_check > 1.0:
                    last_mem_check = time.time()
                    free = _available_ram_gb()
                    if free < EMERGENCY_FREE_RAM_GB:
                        raise ResourceError(f"Only {free:.1f} GB RAM available: aborted LIBERO evaluation")
                t_wait = time.perf_counter()
                ready = wait([self.conns[w] for w in outstanding] + [res_recv])
                stats["wait_s"] += time.perf_counter() - t_wait
                pops: list[int] = []
                for conn in ready:
                    if conn is res_recv:
                        status, wids, payload, dt = res_recv.recv()
                        if status != "ok":
                            raise RuntimeError(f"Policy inference failed: {payload}")
                        stats["infer_s"] += dt
                        stats["infer_calls"] += 1
                        stats["infer_rows"] += len(wids)
                        for row, w in enumerate(wids):
                            self.conns[w].send(("step", payload[row]))
                            outstanding.add(w)
                        inflight = []
                        continue
                    w = conn_to_worker[id(conn)]
                    outstanding.discard(w)
                    msg = conn.recv()
                    run = running[w]
                    if msg[0] == "reset":
                        _, run.obs, run.task, run.max_steps = msg
                        run.task = self.task_override.get((run.job.suite, run.job.task_id), run.task)
                        run.t = 0
                        policy.reset_envs([w])
                        pending_infer.append(w)
                        continue
                    _, run.obs, terminated, success, step_dt = msg
                    stats["step_s"] += step_dt
                    stats["steps"] += 1
                    run.t += 1
                    if success or terminated or run.t >= run.max_steps:
                        results.append(
                            EpisodeResult(
                                suite=run.job.suite,
                                task_id=run.job.task_id,
                                init_state=run.job.init_state,
                                task=run.task,
                                success=bool(success),
                                steps=run.t,
                                wall_s=time.time() - run.start,
                            )
                        )
                        bar.update(1)
                        bar.set_postfix(success=f"{np.mean([r.success for r in results]):.3f}")
                        if hasattr(policy, "episode_done"):
                            policy.episode_done(w, results[-1])
                        policy.reset_envs([w])
                        if early_stop is not None and not aborted and early_stop(results):
                            aborted = True
                            for q in queues.values():
                                q.clear()
                        dispatch(w)
                    elif can_pop and not policy.needs_observation([w])[0]:
                        pops.append(w)
                    else:
                        pending_infer.append(w)
                if pops:
                    # Queue pops ignore the observation, so they skip the (device-moving) processors: the
                    # KNN policy keeps queued actions on the CPU and LIBERO's env postprocessor is the
                    # identity, so the action is exactly what the full path would return.
                    actions = policy.select_action({OBS_STATE: torch.zeros(len(pops), 1)}, env_ids=pops)
                    actions = actions.numpy()
                    for row, w in enumerate(pops):
                        self.conns[w].send(("step", actions[row]))
                        outstanding.add(w)
                    stats["pops"] += len(pops)
                if pending_infer and not inflight:
                    inflight, pending_infer = pending_infer, []
                    infer_queue.put(
                        (inflight, [running[w].obs for w in inflight], [running[w].task for w in inflight])
                    )
        except BaseException:
            bar.close()
            infer_queue.put(None)
            self.close()
            raise
        infer_queue.put(None)
        thread.join(timeout=10)
        bar.close()
        return results


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    """Success rate overall, per suite (``<suite>``) and per task (``<suite>/<task_id>``), and the mean length."""
    out: dict[str, Any] = {"n_episodes": len(results)}
    if not results:
        return out
    out["success"] = float(np.mean([r.success for r in results]))
    suites = sorted({r.suite for r in results}, key=lambda s: SUITES.index(s) if s in SUITES else 99)
    for suite in suites:
        rs = [r for r in results if r.suite == suite]
        out[suite] = float(np.mean([r.success for r in rs]))
        for task_id in sorted({r.task_id for r in rs}):
            rt = [r for r in rs if r.task_id == task_id]
            out[f"{suite}/{task_id}"] = float(np.mean([r.success for r in rt]))
    out["mean_steps"] = float(np.mean([r.steps for r in results]))
    return out
