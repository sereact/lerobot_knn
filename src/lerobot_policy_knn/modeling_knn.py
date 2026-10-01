"""k-nearest-neighbour action-chunk retrieval policy.

At every re-planning step the policy embeds the current observation (camera images with a frozen
vision backbone, plus the proprioceptive state and the episode step), finds the ``k`` closest frames
among the demonstrations of the current task, and returns the (softmax-weighted) average of the action
chunks that followed those frames in their demonstrations. Nothing is learned: the "weights" of the
policy are the embedded demonstration frames themselves (see ``lerobot-knn-fit``).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_knn import KNNConfig
from .encoders import ImageEncoder, TextEncoder, finalize_features
from .transforms import apply_query_transform

logger = logging.getLogger(__name__)

INT_BUFFERS = frozenset({"key_frame", "key_episode", "key_ptr", "task_offsets", "action_episode_end"})
TASK_KEY = "task"


def camera_key_name(camera: str) -> str:
    """Buffer name of a camera's memory keys (buffer names cannot contain dots)."""
    return "key_cam_" + camera.rsplit(".", 1)[-1]


def buffer_dtype(name: str) -> torch.dtype:
    if name in INT_BUFFERS:
        return torch.long
    if name.startswith("key_cam_"):
        return torch.float16
    return torch.float32


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """(..., 3) axis-angle -> (..., 9) flattened rotation matrix (Rodrigues)."""
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = (axis_angle / angle).unbind(-1)
    c, s = torch.cos(angle[..., 0]), torch.sin(angle[..., 0])
    one_c = 1 - c
    return torch.stack(
        [
            c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s,
            y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s,
            z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c,
        ],
        dim=-1,
    )  # fmt: skip


def featurize_state(state: Tensor, axis_angle_start: int | None) -> Tensor:
    state = state.float()
    if axis_angle_start is None:
        return state
    a = axis_angle_start
    return torch.cat([state[..., :a], axis_angle_to_matrix(state[..., a : a + 3]), state[..., a + 3 :]], -1)


def mode_seeking_weights(
    samples: Tensor, weights: Tensor, bandwidth: float, relative: bool = False
) -> Tensor:
    """Re-weight samples toward their dominant mode so that dissimilar actions are not averaged.

    samples: (B, n, D) candidate actions (e.g. flattened neighbour chunks); weights: (B, n), rows sum to
    one (zero = absent). Pairwise distances are per-element mean squared differences, i.e. in squared
    action units. With a Gaussian kernel exp(-d / sigma^2), where sigma = ``bandwidth`` (absolute, action
    units) or sigma^2 = bandwidth * median pairwise distance (``relative``), the mode is the sample with
    the largest weighted kernel support; each weight is multiplied by its kernel similarity to the mode
    and renormalised. Neighbours that agree within ~sigma keep their weights (plain weighted mean);
    separated minority modes are suppressed.
    """
    b, n, dim = samples.shape
    if n == 1:
        return weights
    d = torch.cdist(samples.float(), samples.float()).pow(2) / dim  # (B, n, n)
    valid = weights > 0
    if relative:
        pair_ok = valid[:, :, None] & valid[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=d.device)
        pair = d.masked_fill(~pair_ok, float("nan")).flatten(1)
        sigma2 = (bandwidth * torch.nanmedian(pair, dim=1).values).nan_to_num(1.0).clamp_min(1e-8)
    else:
        sigma2 = torch.full((b,), bandwidth**2, device=d.device)
    kernel = torch.exp(-d / sigma2[:, None, None])
    support = (kernel * weights[:, None, :]).sum(-1).masked_fill(~valid, float("-inf"))
    mode = support.argmax(-1)
    new = weights * kernel[torch.arange(b, device=d.device), mode]
    return new / new.sum(-1, keepdim=True).clamp_min(1e-12)


def _tta_offsets(n: int, px: int) -> list[tuple[int, int]]:
    """The unshifted view plus up to 8 compass shifts of ``px`` pixels."""
    compass = [(0, px), (0, -px), (px, 0), (-px, 0), (px, px), (-px, -px), (px, -px), (-px, px)]
    return [(0, 0), *compass[: max(0, min(n, len(compass)))]]


def _shift(img: Tensor, dy: int, dx: int) -> Tensor:
    """Translate (B, C, H, W) images by (dy, dx) pixels, replicating the border."""
    if dy == 0 and dx == 0:
        return img
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    padded = torch.nn.functional.pad(img, pad, mode="replicate")
    h, w = img.shape[-2:]
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return padded[..., y0 : y0 + h, x0 : x0 + w]


@dataclass
class _EnvSlot:
    """Per-environment inference state (policies are called on batches of independent envs)."""

    t: int = 0
    queue: deque[Tensor] = field(default_factory=deque)
    chunks: list[tuple[int, Tensor]] = field(default_factory=list)  # (start step, chunk) for ensembling
    # top-1 match of the previous retrieval: (episode, frame, step t at retrieval), for stickiness
    last_match: tuple[int, int, int] | None = None
    # embedded observation of the previous retrieval and its step, for history matching
    prev_query: dict[str, Tensor] | None = None
    prev_t: int = 0
    # (step, embedded query) of recent retrievals, for history matching with a fixed gap
    past: deque[tuple[int, dict[str, Tensor]]] = field(default_factory=deque)
    # retry / backtracking state
    executed: deque[Tensor] = field(default_factory=deque)  # recent actions, newest last
    positions: deque[Tensor] = field(default_factory=deque)  # recent end-effector positions
    followed: deque[int] = field(default_factory=deque)  # top-1 demos of recent retrievals
    backtrack: deque[Tensor] = field(default_factory=deque)  # pending rewind actions
    tabu: set[int] = field(default_factory=set)
    unseen_streak: int = 0
    retries: int = 0
    cooldown_until: int = 0


class KNNPolicy(PreTrainedPolicy):
    config_class = KNNConfig
    name = "knn"

    def __init__(self, config: KNNConfig, dataset_stats: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config: KNNConfig = config
        for buf_name, shape in config.memory_shapes.items():
            self.register_buffer(buf_name, torch.zeros(shape, dtype=buffer_dtype(buf_name)))
        # Frozen encoders are created lazily and kept out of the state dict: the checkpoint only
        # stores the memory, the backbones are re-downloaded from the Hub.
        self._frozen: dict[str, Any] = {}
        self._task_lookup: dict[str, int] = {name: i for i, name in enumerate(config.task_names)}
        self._retrieval_index: dict[str, Tensor] = {}  # built lazily by _index()
        self._slots: dict[int, _EnvSlot] = {}
        # Debugging aid: when set to a list, every retrieval appends (query, task ids, t, rows, distances).
        self.query_log: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------ plumbing
    @property
    def device(self) -> torch.device:
        if hasattr(self, "actions"):
            return self.actions.device
        return torch.device(self.config.device or "cpu")

    @property
    def memory_size(self) -> int:
        return int(self.key_ptr.shape[0]) if hasattr(self, "key_ptr") else 0

    def load_memory(self, buffers: dict[str, Tensor]) -> None:
        """Install a memory built by :func:`lerobot_policy_knn.memory.build_memory`."""
        device = self.device
        for buf_name, value in buffers.items():
            value = value.to(device=device, dtype=buffer_dtype(buf_name))
            if buf_name in self._buffers:
                self._buffers[buf_name] = value
            else:
                self.register_buffer(buf_name, value)
            self.config.memory_shapes[buf_name] = list(value.shape)
        self._retrieval_index.clear()

    def _frozen_module(self, key: str, factory: Any) -> Any:
        module, device = self._frozen.get(key, (None, None))
        if module is None:
            module = factory()
        if device != self.device:
            module = module.to(self.device)
        self._frozen[key] = (module, self.device)
        return module

    def key_features(self, key_name: str, raw: Tensor) -> Tensor:
        """Raw pooled features -> unit-norm retrieval key, exactly as the memory keys were built."""
        raw = self.standardize_raw(key_name, raw)
        proj = getattr(self, f"norm_proj_{key_name.removeprefix('key_cam_')}", None)
        if proj is not None:
            return torch.nn.functional.normalize(raw.float() @ proj, dim=-1)
        return finalize_features(raw, self.config.image_pooling)

    def standardize_raw(self, key_name: str, raw: Tensor) -> Tensor:
        """Apply the memory's per-dimension standardisation (``key_norm="zscore"``) to raw query features."""
        name = key_name.removeprefix("key_cam_")
        mean = getattr(self, f"norm_mean_{name}", None)
        if mean is None:
            return raw
        return (raw.float() - mean) / getattr(self, f"norm_std_{name}")

    def image_encoder(self) -> Any:
        cfg = self.config
        if cfg.image_encoder.startswith("vla:"):
            from .vla import VLAEmbedder

            taps = (cfg.image_pooling.split("@", 1)[1],) if "@" in cfg.image_pooling else ()
            return self._frozen_module(
                "image", lambda: VLAEmbedder(cfg.image_encoder[4:], str(self.device), taps=taps)
            )
        return self._frozen_module("image", lambda: ImageEncoder(cfg.image_encoder, cfg.image_resolution))

    def text_encoder(self) -> TextEncoder:
        return self._frozen_module("text", lambda: TextEncoder(self.config.text_encoder))

    def _as_device_tensor(self, values: list[int]) -> Tensor:
        """Small host list -> device tensor without a blocking pageable copy."""
        out = torch.tensor(values, dtype=torch.long)
        if self.device.type == "cuda":
            return out.pin_memory().to(self.device, non_blocking=True)
        return out.to(self.device)

    def get_optim_params(self) -> dict:
        return {"params": list(self.parameters())}

    def reset(self) -> None:
        self._slots.clear()

    def needs_observation(self, env_ids: list[int]) -> list[bool]:
        """Whether the next ``select_action`` of each env runs a retrieval (and so needs its images).

        Envs that still have queued actions only pop them; callers may skip building their images.
        """
        out = []
        for env_id in env_ids:
            slot = self._slots.get(env_id)
            if slot is None:
                out.append(True)
            elif slot.backtrack:
                out.append(False)
            else:
                out.append(self.config.temporal_ensemble_coeff is not None or not slot.queue)
        return out

    def reset_envs(self, env_ids: list[int]) -> None:
        """Forget the inference state of some environments (e.g. after their episode ended)."""
        for env_id in env_ids:
            self._slots.pop(env_id, None)

    # ------------------------------------------------------------------ embedding
    @torch.no_grad()
    def encode_observation(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Embed the retrieval modalities of a batch of observations."""
        cfg = self.config
        feats: dict[str, Tensor] = {}
        cameras = cfg.active_cameras
        if cameras and cfg.image_encoder.startswith("vla:"):
            # a VLA embeds all cameras, the prompt and the state jointly (see vla.py)
            if cfg.query_transform and cfg.query_transform != "none":
                batch = dict(batch)
                for cam in cameras:
                    if cam in batch:
                        img = batch[cam]
                        img = img.float() / 255.0 if img.dtype == torch.uint8 else img
                        batch[cam] = apply_query_transform(img.to(self.device), cfg.query_transform)
            pooled = self.image_encoder()(batch)[cfg.image_pooling]
            for cam in cameras:
                name = camera_key_name(cam)
                feats[name] = self.key_features(name, pooled[cam])
        elif cameras:
            images = []
            for cam in cameras:
                img = batch[cam]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                img = apply_query_transform(img.to(self.device, non_blocking=True), cfg.query_transform)
                images.append(img)
            # test-time augmentation: embed shifted copies and average their unit-norm keys; every camera and
            # view goes through the backbone in one batch when the resolutions agree
            views = _tta_offsets(cfg.query_tta, cfg.query_tta_px)
            stacks = [torch.cat([_shift(img, dy, dx) for dy, dx in views]) for img in images]
            if len({tuple(x.shape[-2:]) for x in stacks}) == 1:
                raws = self.image_encoder()(torch.cat(stacks))[cfg.image_pooling].chunk(len(cameras))
            else:
                raws = tuple(self.image_encoder()(x)[cfg.image_pooling] for x in stacks)
            for cam, img, raw in zip(cameras, images, raws, strict=True):
                keys = self.key_features(camera_key_name(cam), raw).view(len(views), img.shape[0], -1).mean(0)
                feats[camera_key_name(cam)] = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if cfg.state_weight > 0 or cfg.gripper_weight > 0:
            state = featurize_state(batch[OBS_STATE].to(self.device), cfg.state_axis_angle_start)
            feats["key_state"] = (state - self.state_mean) / self.state_std
        return feats

    @torch.no_grad()
    def task_ids(self, tasks: list[str] | None) -> Tensor | None:
        """Map prompts to memory task indices (nearest task-prompt embedding)."""
        if self.config.task_conditioning == "none" or not tasks or not self.config.task_names:
            return None
        unknown = sorted({t for t in tasks if t not in self._task_lookup})
        if unknown:
            emb = self.text_encoder()(unknown)
            sim = emb @ self.task_text_emb.T
            best = sim.argmax(-1).tolist()
            for prompt, idx, score in zip(unknown, best, sim.max(-1).values.tolist(), strict=True):
                logger.info(
                    "Prompt %r mapped to task %r (cosine %.3f)", prompt, self.config.task_names[idx], score
                )
                self._task_lookup[prompt] = idx
        return self._as_device_tensor([self._task_lookup[t] for t in tasks])

    # ------------------------------------------------------------------ retrieval
    def _index(self) -> dict[str, Tensor]:
        """Padded global key indices of every task window and of every episode segment.

        ``windows[t]`` lists the keys of task ``t`` and ``segments[t, e]`` the keys of its ``e``-th
        demonstration; padding points to the sentinel column ``memory_size`` whose distance is +inf.
        Without task conditioning there is a single "task" spanning the whole memory.
        """
        cached = self._retrieval_index
        if cached and cached["windows"].device == self.device:
            return cached
        n = self.memory_size
        dev = self.device
        episode = self.key_episode.cpu()
        offsets = self.task_offsets.cpu().tolist()
        if self.config.task_conditioning == "filter":
            bounds = list(zip(offsets[:-1], offsets[1:], strict=True))
        else:
            bounds = [(0, n)]
        lmax = max(1, max(hi - lo for lo, hi in bounds))
        windows = torch.full((len(bounds), lmax), n, dtype=torch.long)
        seg_lists = []
        for t, (lo, hi) in enumerate(bounds):
            windows[t, : hi - lo] = torch.arange(lo, hi)
            if hi > lo:
                _, counts = torch.unique_consecutive(episode[lo:hi], return_counts=True)
                starts = (torch.cumsum(counts, 0) - counts + lo).tolist()
                seg_lists.append(list(zip(starts, counts.tolist(), strict=True)))
            else:
                seg_lists.append([])
        emax = max(1, max(len(x) for x in seg_lists))
        lep = max(1, max((c for x in seg_lists for _, c in x), default=1))
        segments = torch.full((len(bounds), emax, lep), n, dtype=torch.long)
        for t, segs in enumerate(seg_lists):
            for e, (st, c) in enumerate(segs):
                segments[t, e, :c] = torch.arange(st, st + c)
        counts_per_task = torch.diff(torch.tensor(offsets))
        key_task = torch.repeat_interleave(torch.arange(len(counts_per_task)), counts_per_task)
        index = {"windows": windows.to(dev), "segments": segments.to(dev), "key_task": key_task.to(dev)}
        if self.config.state_weight > 0:
            index["state_sq"] = self.key_state.pow(2).sum(-1)
        self._retrieval_index = index
        return index

    def _scales(self) -> dict[str, float]:
        cfg = self.config
        if cfg.distance_normalization == "none":
            return {}
        if cfg.distance_normalization == "nn_median":
            return {**cfg.modality_scales, **cfg.modality_nn_scales}
        return cfg.modality_scales

    def _distances(
        self, query: dict[str, Tensor], t: Tensor | None, task_ids: Tensor | None = None
    ) -> Tensor:
        """(B, memory_size + 1) weighted distances to every key, plus a +inf sentinel column."""
        cfg = self.config
        scales = self._scales()
        index = self._index()
        b = next(iter(query.values())).shape[0]
        dist = torch.zeros(b, self.memory_size + 1, device=self.device)
        body = dist[:, :-1]
        for cam in cfg.active_cameras:
            name = camera_key_name(cam)
            w = cfg.camera_weights[cam] / scales.get(name, 1.0)
            # keys and queries are unit norm: ||q - m||^2 = 2 - 2 q.m (fp16 inputs, fp32 accumulation)
            dots = torch.mm(query[name].half(), getattr(self, name).T, out_dtype=torch.float32)
            body.add_((2.0 - 2.0 * dots).clamp_(min=0.0), alpha=w)
        if cfg.state_weight > 0:
            q = query["key_state"].float()
            d = q.pow(2).sum(-1, keepdim=True) + index["state_sq"][None] - 2.0 * q @ self.key_state.T
            body.add_(d.clamp_(min=0.0), alpha=cfg.state_weight / scales.get("key_state", 1.0) / q.shape[-1])
        if cfg.gripper_weight > 0:
            g = cfg.gripper_state_dims
            q = query["key_state"][:, -g:].float()
            keys = self.key_state[:, -g:]
            d = q.pow(2).sum(-1, keepdim=True) + keys.pow(2).sum(-1)[None] - 2.0 * q @ keys.T
            body.add_(d.clamp_(min=0.0), alpha=cfg.gripper_weight / g)
        if cfg.time_weight > 0 and t is not None:
            d = (t[:, None].float() - self.key_frame[None].float()).pow(2)
            body.add_(d, alpha=cfg.time_weight / scales.get("time", 1.0))
        if cfg.task_conditioning == "soft" and task_ids is not None and cfg.text_weight > 0:
            emb = self.task_text_emb
            text_d = (2.0 - 2.0 * emb[task_ids] @ emb.T).clamp_(min=0.0)  # (B, T)
            body.add_(text_d[:, index["key_task"]], alpha=cfg.text_weight / scales.get("text", 1.0))
        dist[:, -1] = float("inf")
        return dist

    @torch.no_grad()
    def retrieve(
        self,
        query: dict[str, Tensor],
        t: Tensor | None = None,
        task_ids: Tensor | None = None,
        exclude_episode: Tensor | None = None,
        continuation: tuple[Tensor, Tensor] | None = None,
        extra_distance: Tensor | None = None,
        excluded: list[set[int]] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (rows, weights, distances), each (B, k): the k retrieved memory keys per query.

        ``extra_distance`` (B, memory_size) is added to the observation distances (history matching).
        """
        if self.memory_size == 0:
            raise RuntimeError("The KNN policy has an empty memory; build it with `lerobot-knn-fit`.")
        cfg = self.config
        n = self.memory_size
        dist = self._distances(query, t, task_ids)
        b = dist.shape[0]
        if extra_distance is not None:
            dist[:, :-1].add_(extra_distance)
        if exclude_episode is not None:
            dist[:, :-1].masked_fill_(self.key_episode[None] == exclude_episode[:, None], float("inf"))
        if excluded is not None and any(excluded):
            for r, eps in enumerate(excluded):
                if eps:
                    banned = torch.isin(self.key_episode, torch.tensor(sorted(eps), device=self.device))
                    dist[r, :-1].masked_fill_(banned, float("inf"))
        if continuation is not None and cfg.consistency_weight > 0:
            # Trajectory stickiness: frames that continue the previously matched demonstration (same
            # episode, frame within +-consistency_window of where it would be now) get their distance
            # scaled by (1 - consistency_weight). Rows without a previous match pass episode -1.
            episode, frame = continuation
            near = (self.key_episode[None] == episode[:, None]) & (
                (self.key_frame[None] - frame[:, None]).abs() <= cfg.consistency_window
            )
            dist[:, :-1].mul_(torch.where(near, 1.0 - cfg.consistency_weight, 1.0))
        index = self._index()
        if cfg.task_conditioning == "filter" and task_ids is not None:
            group = task_ids
        else:
            group = torch.zeros(b, dtype=torch.long, device=self.device)
        if cfg.one_per_episode:
            seg = index["segments"][group]  # (B, E, L)
            per_frame = dist.gather(1, seg.flatten(1)).view(seg.shape)
            ep_min, ep_arg = per_frame.min(-1)  # (B, E)
            kk = min(cfg.k, ep_min.shape[1])
            vals, ep_sel = ep_min.topk(kk, largest=False)
            frame_sel = ep_arg.gather(1, ep_sel)
            rows = seg[torch.arange(b, device=seg.device)[:, None], ep_sel, frame_sel]
        else:
            win = index["windows"][group]  # (B, L)
            kk = min(cfg.k, win.shape[1])
            vals, pos = dist.gather(1, win).topk(kk, largest=False)
            rows = win.gather(1, pos)
        if kk < cfg.k:
            pad = cfg.k - kk
            vals = torch.cat([vals, vals.new_full((b, pad), float("inf"))], 1)
            rows = torch.cat([rows, rows[:, :1].expand(b, pad)], 1)
        # padding (sentinel) rows carry +inf distance and zero weight; point them at a valid key
        rows = torch.where(rows >= n, rows[:, :1], rows)
        return rows, self._neighbour_weights(vals), vals

    def _neighbour_weights(self, dists: Tensor) -> Tensor:
        finite = torch.isfinite(dists)
        if self.config.k == 1 or self.config.temperature <= 0:
            w = torch.zeros_like(dists)
            w[:, 0] = 1.0
            return w
        d = dists.masked_fill(~finite, 0.0)
        denom = (d.sum(-1, keepdim=True) / finite.sum(-1, keepdim=True).clamp_min(1)).clamp_min(1e-8)
        logits = (-d / (self.config.temperature * denom)).masked_fill(~finite, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def gather_chunks(self, rows: Tensor, horizon: int | None = None) -> Tensor:
        """(B, k) memory rows -> (B, k, H, A) action chunks that followed them (held at episode end)."""
        horizon = horizon or self.config.chunk_size
        ptr = self.key_ptr[rows]
        idx = ptr[..., None] + torch.arange(horizon, device=ptr.device)
        idx = torch.minimum(idx, self.action_episode_end[ptr][..., None])
        return self.actions[idx]

    def _shifted_rows(self, steps: int) -> Tensor:
        """Key row ``steps`` env steps earlier in the same demonstration (clamped at its start)."""
        index = self._index()
        cache_key = f"shift_{steps}"
        if cache_key not in index:
            if "episode_start" not in index:
                ep = self.key_episode
                change = torch.ones_like(ep, dtype=torch.bool)
                change[1:] = ep[1:] != ep[:-1]
                starts = torch.nonzero(change)[:, 0]
                counts = torch.diff(torch.cat([starts, starts.new_tensor([len(ep)])]))
                index["episode_start"] = torch.repeat_interleave(starts, counts)
            back = max(0, round(steps / self.config.memory_stride))
            rows = torch.arange(self.memory_size, device=self.device) - back
            index[cache_key] = torch.maximum(rows, index["episode_start"])
        return index[cache_key]

    def history_distance(self, slots: list[_EnvSlot]) -> Tensor | None:
        """(B, memory_size) distance between each env's previous query and, for every key, the key that
        preceded it by the same number of steps in its demonstration. Zero for envs without history."""
        cfg = self.config
        if cfg.history_weight <= 0:
            return None
        # past observation per env: the one `history_gap` steps back (most recent at least that old), or,
        # with history_gap = 0, simply the previous retrieval
        chosen: dict[int, tuple[int, dict[str, Tensor]]] = {}
        for r, slot in enumerate(slots):
            if cfg.history_gap > 0:
                old = [(t, q) for t, q in slot.past if t <= slot.t - cfg.history_gap]
                if old:
                    chosen[r] = old[-1]
            elif slot.prev_query is not None:
                chosen[r] = (slot.prev_t, slot.prev_query)
        if not chosen:
            return None
        out = torch.zeros(len(slots), self.memory_size, device=self.device)
        by_gap: dict[int, list[int]] = {}
        for r, (t_prev, _q) in chosen.items():
            by_gap.setdefault(slots[r].t - t_prev, []).append(r)
        for gap, rows in by_gap.items():
            prev = {name: torch.stack([chosen[r][1][name] for r in rows]) for name in chosen[rows[0]][1]}
            t_prev = self._as_device_tensor([chosen[r][0] for r in rows])
            d = self._distances(prev, t_prev)[:, :-1]
            out[self._as_device_tensor(rows)] = d[:, self._shifted_rows(gap)]
        return out.mul_(self.config.history_weight)

    @torch.no_grad()
    def task_distance_scale(self) -> Tensor:
        """(T,) median top-1 distance of each task's demo frames queried against its *other* demos
        (without history/stickiness terms): the "normal" distance used to detect unseen states."""
        index = self._index()
        if "calib" in index:
            return index["calib"]
        g = torch.Generator().manual_seed(0)
        offsets = self.task_offsets.cpu().tolist()
        n_tasks = len(offsets) - 1
        cfg = self.config
        names = [camera_key_name(c) for c in cfg.active_cameras] + (
            ["key_state"] if cfg.state_weight > 0 or cfg.gripper_weight > 0 else []
        )
        rows, tasks = [], []
        for t in range(n_tasks):
            lo, hi = offsets[t], offsets[t + 1]
            if hi > lo:
                pick = lo + torch.randperm(hi - lo, generator=g)[:64]
                rows.append(pick)
                tasks.append(torch.full((len(pick),), t))
        rows_t, tasks_t = torch.cat(rows).to(self.device), torch.cat(tasks).to(self.device)
        top1 = []
        for s in range(0, len(rows_t), 256):
            r = rows_t[s : s + 256]
            query = {n: getattr(self, n)[r] for n in names}
            task = tasks_t[s : s + 256] if self.config.task_conditioning != "none" else None
            _, _, d = self.retrieve(query, self.key_frame[r], task, exclude_episode=self.key_episode[r])
            top1.append(d[:, 0])
        top1_t = torch.cat(top1)
        calib = torch.ones(max(n_tasks, 1), device=self.device)
        for t in range(n_tasks):
            m = top1_t[tasks_t == t]
            m = m[torch.isfinite(m)]
            if len(m):
                calib[t] = m.median().clamp_min(1e-6)
        index["calib"] = calib
        return calib

    def aggregate(self, rows: Tensor, weights: Tensor) -> Tensor:
        chunks = self.gather_chunks(rows)
        if self.config.action_mode_bandwidth > 0:
            # only average neighbours whose (motion) chunks agree with the dominant one
            motion = chunks[..., : self.config.gripper_index % chunks.shape[-1]].flatten(2)
            weights = mode_seeking_weights(motion, weights, self.config.action_mode_bandwidth)
        out = (weights[..., None, None] * chunks).sum(1)
        if self.config.norm_preserving:
            # averaging motions that point in different directions shrinks them; restore the
            # neighbours' mean translation / rotation magnitude
            for lo, hi in ((0, 3), (3, 6)):
                mean_norm = (weights[..., None] * chunks[..., lo:hi].norm(dim=-1)).sum(1)
                norm = out[..., lo:hi].norm(dim=-1, keepdim=True).clamp_min(1e-6)
                out[..., lo:hi] *= mean_norm[..., None] / norm
        if self.config.gripper_mode == "sign":
            g = self.config.gripper_index
            out[..., g] = torch.where(out[..., g] >= 0, 1.0, -1.0)
        return out

    # ------------------------------------------------------------------ policy API
    def _predict(self, batch: dict[str, Any], t: Tensor, slots: list[_EnvSlot] | None = None) -> Tensor:
        query = self.encode_observation(batch)
        tasks = batch.get(TASK_KEY)
        if isinstance(tasks, str):
            tasks = [tasks]
        continuation = None
        if slots is not None and self.config.consistency_weight > 0:
            episodes, frames = [], []
            for slot in slots:
                if slot.last_match is None:
                    episodes.append(-1)
                    frames.append(0)
                else:
                    ep, fr, t0 = slot.last_match
                    episodes.append(ep)
                    frames.append(fr + slot.t - t0)
            continuation = (self._as_device_tensor(episodes), self._as_device_tensor(frames))
        history = self.history_distance(slots) if slots is not None else None
        task_ids = self.task_ids(tasks)
        excluded = [slot.tabu for slot in slots] if slots is not None and self.config.retry_tabu else None
        rows, weights, dists = self.retrieve(
            query, t, task_ids, continuation=continuation, extra_distance=history, excluded=excluded
        )
        cfg = self.config
        if slots is not None and (cfg.retry_distance_threshold > 0 or cfg.retry_stuck_steps > 0):
            top_eps = self.key_episode[rows[:, 0]].tolist()
            for slot, ep in zip(slots, top_eps, strict=True):
                slot.followed.append(ep)
                while len(slot.followed) > cfg.retry_backtrack_steps:
                    slot.followed.popleft()
        if slots is not None and cfg.retry_distance_threshold > 0:
            calib = self.task_distance_scale()
            scale = calib[task_ids] if task_ids is not None else calib.median().expand(len(slots))
            # history matching adds to the distance; compare observation-only distances with the calibration
            obs_top1 = dists[:, 0] - (history.gather(1, rows[:, :1])[:, 0] if history is not None else 0.0)
            for slot, nd in zip(slots, (obs_top1 / scale).tolist(), strict=True):
                slot.unseen_streak = slot.unseen_streak + 1 if nd > cfg.retry_distance_threshold else 0
        if self.query_log is not None:
            self.query_log.append(
                {
                    "query": {k: v.detach().cpu() for k, v in query.items()},
                    "task": None if task_ids is None else task_ids.cpu(),
                    "t": t.cpu(),
                    "rows": rows.cpu(),
                    "dists": dists.cpu(),
                }
            )
        if slots is not None and self.config.history_weight > 0:
            for r, slot in enumerate(slots):
                slot.prev_query = {name: v[r] for name, v in query.items()}
                slot.prev_t = slot.t
                if self.config.history_gap > 0:
                    slot.past.append((slot.t, slot.prev_query))
                    while len(slot.past) > 1 and slot.past[1][0] <= slot.t - self.config.history_gap:
                        slot.past.popleft()
        if slots is not None and self.config.consistency_weight > 0:
            top = rows[:, 0]
            matches = torch.stack([self.key_episode[top], self.key_frame[top]], 1).tolist()
            for slot, (ep, fr) in zip(slots, matches, strict=True):
                slot.last_match = (ep, fr, slot.t)
        return self.aggregate(rows, weights)

    @staticmethod
    def _batch_size(batch: dict[str, Any]) -> int:
        for key, value in batch.items():
            if isinstance(value, Tensor) and key.startswith("observation."):
                return value.shape[0]
        raise ValueError("Cannot infer the batch size of an observation without tensors")

    @staticmethod
    def _select_rows(batch: dict[str, Any], rows: list[int], b: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == b:
                out[key] = value[rows]
            elif isinstance(value, (list, tuple)) and len(value) == b:
                out[key] = [value[r] for r in rows]
            else:
                out[key] = value
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        env_ids = kwargs.get("env_ids")
        b = self._batch_size(batch)
        ids = list(env_ids) if env_ids is not None else list(range(b))
        t = self._as_device_tensor([self._slots[i].t if i in self._slots else 0 for i in ids])
        return self._predict(batch, t)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Next action for every env of the batch. ``env_ids`` identifies envs across calls (default:
        the batch position), so callers may run envs asynchronously and reset them independently."""
        self.eval()
        cfg = self.config
        env_ids = kwargs.get("env_ids")
        b = self._batch_size(batch)
        ids = list(env_ids) if env_ids is not None else list(range(b))
        slots = [self._slots.setdefault(i, _EnvSlot()) for i in ids]
        ensemble = cfg.temporal_ensemble_coeff is not None
        need = [r for r, s in enumerate(slots) if not s.backtrack and (ensemble or not s.queue)]
        retry = cfg.retry_distance_threshold > 0 or cfg.retry_stuck_steps > 0
        if retry and cfg.retry_stuck_steps > 0 and need and OBS_STATE in batch:
            pos = batch[OBS_STATE][:, :3].detach().float().cpu()
            if pos.shape[0] == b:
                for r in need:
                    slots[r].positions.append(pos[r])
                    while len(slots[r].positions) > cfg.retry_stuck_steps:
                        slots[r].positions.popleft()
        if need:
            sub = batch if len(need) == b else self._select_rows(batch, need, b)
            t = self._as_device_tensor([slots[r].t for r in need])
            # queued actions live on the CPU: popping them later never touches (or waits for) the GPU
            chunks = self._predict(sub, t, [slots[r] for r in need]).cpu()
            for j, r in enumerate(need):
                if ensemble:
                    slots[r].chunks.append((slots[r].t, chunks[j]))
                else:
                    slots[r].queue.extend(chunks[j, : cfg.n_action_steps])
        actions = []
        for slot in slots:
            if slot.backtrack:
                action = slot.backtrack.popleft()
            else:
                action = self._ensembled_action(slot) if ensemble else slot.queue.popleft()
                if retry and self._should_retry(slot):
                    self._start_backtrack(slot)
                    action = slot.backtrack.popleft()
                elif retry:
                    slot.executed.append(action)
                    while len(slot.executed) > cfg.retry_backtrack_steps:
                        slot.executed.popleft()
            actions.append(action)
            slot.t += 1
        return torch.stack(actions)

    def _should_retry(self, slot: _EnvSlot) -> bool:
        cfg = self.config
        if slot.retries >= cfg.retry_max or slot.t < slot.cooldown_until or len(slot.executed) < 4:
            return False
        if cfg.retry_distance_threshold > 0 and slot.unseen_streak >= cfg.retry_patience:
            return True
        if cfg.retry_stuck_steps > 0 and len(slot.positions) >= cfg.retry_stuck_steps:
            window = torch.stack(list(slot.positions))
            if float((window - window[-1]).norm(dim=-1).max()) < cfg.retry_stuck_eps:
                return True
        return False

    def _start_backtrack(self, slot: _EnvSlot) -> None:
        """Rewind: the recent motion commands negated, newest first, with the gripper as it was then."""
        g = self.config.gripper_index % slot.executed[0].shape[-1]
        rewind = []
        for a in reversed(slot.executed):
            r = -a
            r[g] = a[g]
            rewind.append(r)
        slot.backtrack = deque(rewind)
        if self.config.retry_tabu:
            slot.tabu |= set(slot.followed)
        slot.retries += 1
        slot.cooldown_until = slot.t + len(rewind) + self.config.retry_cooldown
        slot.executed.clear()
        slot.followed.clear()
        slot.positions.clear()
        slot.queue.clear()
        slot.chunks.clear()
        slot.unseen_streak = 0
        slot.prev_query = None
        slot.last_match = None

    def _ensembled_action(self, slot: _EnvSlot) -> Tensor:
        cfg = self.config
        slot.chunks = [(s, c) for s, c in slot.chunks if slot.t - s < c.shape[0]]
        preds = torch.stack([c[slot.t - s] for s, c in slot.chunks])  # oldest first
        w = torch.exp(-cfg.temporal_ensemble_coeff * torch.arange(len(preds), device=preds.device))
        w = w / w.sum()
        if cfg.ensemble_mode_bandwidth > 0 and len(preds) > 2:
            g = cfg.gripper_index % preds.shape[-1]
            w = mode_seeking_weights(preds[None, :, :g], w[None], cfg.ensemble_mode_bandwidth)[0]
        action = (w[:, None] * preds).sum(0)
        if cfg.gripper_mode == "sign":
            g = cfg.gripper_index
            action[g] = 1.0 if action[g] >= 0 else -1.0
        return action

    @torch.no_grad()
    def forward(self, batch: dict[str, Any], reduction: str = "mean") -> tuple[Tensor, dict | None]:
        """Retrieval error on a dataset batch (the batch's own episode is excluded from the memory).

        There is nothing to train; this exists so the policy can be scored with the standard loop.
        """
        query = self.encode_observation(batch)
        t = batch.get("frame_index")
        exclude = batch.get("episode_index")
        tasks = batch.get(TASK_KEY)
        rows, weights, _ = self.retrieve(
            query,
            t.to(self.device).view(-1) if isinstance(t, Tensor) else None,
            self.task_ids(list(tasks) if tasks is not None else None),
            exclude.to(self.device).view(-1) if isinstance(exclude, Tensor) else None,
        )
        pred = self.aggregate(rows, weights)
        target = batch[ACTION].to(self.device)
        horizon = min(pred.shape[1], target.shape[1]) if target.ndim == 3 else 1
        err = (pred[:, :horizon] - (target[:, :horizon] if target.ndim == 3 else target[:, None])).pow(2)
        loss = err.mean(dim=(1, 2)) if reduction == "none" else err.mean()
        return loss, {"retrieval_mse": float(err.mean())}
