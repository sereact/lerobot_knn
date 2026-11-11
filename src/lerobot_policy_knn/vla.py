"""VLA backbones as retrieval embedders, and per-environment action queues for VLA rollouts.

``VLAEmbedder`` runs only the *prefix* of a vision-language-action model (images + prompt [+ state]) through
its VLM and pools the final hidden states. The kNN policy uses these vectors as retrieval keys
(``image_encoder="vla:<name>"``), which lets us ask whether a VLA's action head adds anything beyond a lookup
in the space its (fine-tuned) VLM has learned.

Poolings (``image_pooling``), each a dict ``{camera: (B, D)}``:

* ``img``: mean of each camera's image tokens (these attend to the prompt, so they are task-conditioned)
* ``lang``: mean of the valid prompt tokens (pi0.5 writes the discretised state into the prompt)
* ``all``: mean of every valid prefix token
* ``state``: SmolVLA's state token, which attends to the whole prefix (a summary of image, task and state)

Non-camera poolings are stored under the pseudo camera ``observation.vla``.
"""

import logging
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .encoders import grid_pool

logger = logging.getLogger(__name__)

VLA_CAMERA = "observation.vla"
DATASET_CAMERAS = ("observation.images.image", "observation.images.image2")
VLA_ROOT = Path("outputs/vla")


@dataclass(frozen=True)
class VLASpec:
    """Where a VLA's weights and LIBERO processors come from, and its documented LIBERO evaluation settings."""

    family: str  # LeRobot policy type: "smolvla", "pi05", "pi0", "pi0_fast", "xvla", "vla_jepa", "groot", ...
    # Hub repo or local directory of the weights. None: the VLM as originally published (no robot data), built from
    # the processor checkpoint's config with a randomly initialised action head.
    weights: str | None
    # Checkpoint the LIBERO pre/post processors (normalisation statistics, tokenizer) and camera layout come from.
    processor: str
    note: str = ""
    # Policy config fields set before loading (each family's settings in LeRobot's docs/source/<policy>.mdx).
    config_overrides: dict[str, Any] = field(default_factory=dict)
    # Pinned Hub revision of the weights and processors.
    revision: str | None = None
    # Dataset camera key -> the key the policy expects.
    camera_rename: dict[str, str] = field(default_factory=dict)
    # LIBERO render size its evaluation uses.
    observation_size: int = 256
    # Its documented re-planning interval. None: the caller's choice; 0: the checkpoint config's n_action_steps.
    n_action_steps: int | None = None
    # Config fields of an older checkpoint that current LeRobot renamed (old -> new) or removed (old -> None).
    legacy_config: dict[str, str | None] = field(default_factory=dict)


SMOLVLA_LIBERO = "lerobot/smolvla_libero"
PI05_LIBERO = "lerobot/pi05_libero_finetuned_v044"
PI0_LIBERO = "lerobot/pi0_libero_finetuned_v044"
GROOT_BASE = "nvidia/GR00T-N1.7-3B"
GROOT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MOLMOACT2_LIBERO = "allenai/MolmoAct2-LIBERO-LeRobot"
EVO1_LIBERO = "zuoxingdong/evo1_libero"


def _groot_libero(suite: str) -> VLASpec:
    """GR00T N1.7's LIBERO checkpoint of one suite.

    These are LeRobot checkpoints (weights + serialized processors holding the normalisation statistics). Their
    config's ``base_model_path`` is a path on NVIDIA's machine, so the architecture is built from the checkpoint repo
    itself before its weights load. 8 of the 16 predicted actions run before re-planning, as in NVIDIA's LIBERO
    rollout wrapper.
    """
    repo = f"nvidia/gr00t17-lerobot-{suite}-640"
    return VLASpec(
        "groot",
        repo,
        repo,
        f"GR00T N1.7 fine-tuned on {suite}",
        config_overrides={"base_model_path": repo},
        n_action_steps=8,
    )


# MolmoAct2 (LeRobot port): wrist camera as "wrist_image", continuous actions in bfloat16.
_MOLMOACT2 = {
    "processor": MOLMOACT2_LIBERO,
    "config_overrides": {
        "inference_action_mode": "continuous",
        # the checkpoint is a full fine-tune (enable_lora_vlm=False): no LoRA wrapping, or its keys do not load
        "train_mode_vlm": "fft",
        "dtype": "bfloat16",
        "enable_inference_cuda_graph": False,
    },
    "camera_rename": {"observation.images.image2": "observation.images.wrist_image"},
    "n_action_steps": 0,
    # the Hub config predates LeRobot #4249: training-only switches (all off) and the dtype field's old name
    "legacy_config": {
        "enable_lora_vlm": None,
        "enable_lora_action_expert": None,
        "train_action_expert_only": None,
        "model_dtype": "dtype",
    },
}

# Evo-1 (InternVL3-1B backbone): the verified Stage-2 checkpoint at 448 px renders, 14 actions per re-plan.
_EVO1 = {
    "processor": EVO1_LIBERO,
    "revision": "515921f4a2c1d3f3ad523721eafa26fdf2af315b",
    "config_overrides": {
        "vlm_model_name": "OpenGVLab/InternVL3-1B-hf",
        "use_amp": True,
        "vlm_dtype": "bfloat16",
        "use_flash_attn": False,
        "enable_gradient_checkpointing": False,
        "vlm_num_layers": 14,
        "image_resolution": [448, 448],
        "max_text_length": 1024,
        "chunk_size": 50,
        "n_action_steps": 14,
        "max_state_dim": 24,
        "max_action_dim": 24,
        "num_inference_timesteps": 32,
        "postprocess_action_dim": 7,
        "binarize_gripper": True,
        "gripper_threshold": 0.0,
        "gripper_below_threshold_value": -1.0,
        "gripper_above_threshold_value": 1.0,
    },
    "observation_size": 448,
    "n_action_steps": 14,
}

VLA_SPECS: dict[str, VLASpec] = {
    "smolvla_base": VLASpec(
        "smolvla", "lerobot/smolvla_base", SMOLVLA_LIBERO, "SmolVLA pretrained, no LIBERO"
    ),
    "smolvla_libero": VLASpec("smolvla", SMOLVLA_LIBERO, SMOLVLA_LIBERO, "SmolVLA fine-tuned on LIBERO"),
    # the base checkpoints of pi0 and pi0.5 are local bfloat16 copies (experiments/convert_bf16.py)
    "pi05_base": VLASpec("pi05", str(VLA_ROOT / "pi05_base_bf16"), PI05_LIBERO, "pi0.5 base, no LIBERO"),
    "pi05_libero": VLASpec("pi05", PI05_LIBERO, PI05_LIBERO, "pi0.5 fine-tuned on LIBERO"),
    "pi0_base": VLASpec("pi0", str(VLA_ROOT / "pi0_base_bf16"), PI0_LIBERO, "pi0 base"),
    "pi0_libero": VLASpec("pi0", PI0_LIBERO, PI0_LIBERO, "pi0 LIBERO"),
    # X-VLA (Florence-2 backbone): pretrained base -> LeRobot's LIBERO fine-tune
    "xvla_base": VLASpec("xvla", "lerobot/xvla-base", "lerobot/xvla-libero", "X-VLA base, no LIBERO"),
    "xvla_libero": VLASpec(
        "xvla", "lerobot/xvla-libero", "lerobot/xvla-libero", "X-VLA fine-tuned on LIBERO"
    ),
    # pi0-FAST (PaliGemma, autoregressive FAST action tokens): base -> LIBERO fine-tune
    "pi0fast_base": VLASpec("pi0_fast", "lerobot/pi0fast-base", "lerobot/pi0fast-libero", "pi0-FAST base"),
    "pi0fast_libero": VLASpec(
        "pi0_fast", "lerobot/pi0fast-libero", "lerobot/pi0fast-libero", "pi0-FAST LIBERO"
    ),
    # VLA-JEPA (Qwen3-VL-2B): DROID pretraining -> LIBERO fine-tune
    "vla_jepa_pretrain": VLASpec(
        "vla_jepa", "lerobot/VLA-JEPA-Pretrain", "lerobot/VLA-JEPA-LIBERO", "VLA-JEPA DROID"
    ),
    "vla_jepa_libero": VLASpec(
        "vla_jepa", "lerobot/VLA-JEPA-LIBERO", "lerobot/VLA-JEPA-LIBERO", "VLA-JEPA LIBERO"
    ),
    # GR00T N1.7 (Cosmos-Reason2 / Qwen3-VL backbone): one LIBERO checkpoint per suite
    **{f"groot_{suite}": _groot_libero(suite) for suite in GROOT_SUITES},
    # The raw base checkpoint with a LIBERO checkpoint's saved processors. Its VLM is bit-identical to the one every
    # LIBERO checkpoint keeps (they froze it), so this is GR00T's key, on the images its LIBERO policies see.
    "groot_base": VLASpec(
        "groot",
        GROOT_BASE,
        "nvidia/gr00t17-lerobot-libero_spatial-640",
        "GR00T N1.7 base",
        config_overrides={"embodiment_tag": "libero_sim"},
    ),
    "molmoact2_libero": VLASpec("molmoact2", MOLMOACT2_LIBERO, note="MolmoAct2 LIBERO", **_MOLMOACT2),
    # the released base, built from its HF checkpoint with the LIBERO fine-tune's processors and camera layout
    "molmoact2_base": VLASpec(
        "molmoact2",
        None,
        note="MolmoAct2 before LIBERO (allenai/MolmoAct2)",
        **{
            **_MOLMOACT2,
            "config_overrides": {**_MOLMOACT2["config_overrides"], "checkpoint_path": "allenai/MolmoAct2"},
        },
    ),
    "evo1_libero": VLASpec("evo1", EVO1_LIBERO, note="Evo-1 LIBERO", **_EVO1),
    # no Evo-1 checkpoint before LIBERO: its VLM as published, built from evo1_libero's config (action head unused)
    "evo1_base": VLASpec(
        "evo1", None, note="Evo-1's VLM before LIBERO: the raw InternVL3-1B (its first 14 layers)", **_EVO1
    ),
}


def resolve_spec(name: str) -> VLASpec:
    """Registry name, or ``<family>@<local checkpoint dir or Hub repo>`` for any other checkpoint."""
    if name in VLA_SPECS:
        return VLA_SPECS[name]
    if "@" not in name:
        raise KeyError(f"Unknown VLA embedder {name!r}; known: {sorted(VLA_SPECS)}")
    family, path = name.split("@", 1)
    # the checkpoint's own processors, unless it is a local directory saved without them
    has_processors = not Path(path).exists() or (Path(path) / "policy_preprocessor.json").exists()
    return VLASpec(family, path, path if has_processors else SMOLVLA_LIBERO)


def cache_tag(name: str) -> str:
    """Feature-cache directory name of a VLA embedder."""
    if "@" in name:
        family, path = name.split("@", 1)
        # .../checkpoints/<step>/pretrained_model -> <run>_<step>
        parts = [x for x in Path(path).parts if x not in ("pretrained_model", "checkpoints")]
        name = f"{family}_{'_'.join(parts[-2:])}"
    return "vla_" + name.replace("/", "_")


def load_vla(name: str, device: str = "cuda"):
    """Return ``(policy, preprocessor, postprocessor)`` of a registered VLA."""
    from lerobot.policies.factory import get_policy_class

    spec = resolve_spec(name)
    if spec.family == "groot" and spec.weights == GROOT_BASE:  # a raw NVIDIA checkpoint
        return _load_groot(spec, device)
    policy_cls = get_policy_class(spec.family)
    cfg = _pretrained_config(spec.processor, spec)
    if spec.weights is None:
        cfg.load_vlm_weights = True  # the VLM's own pretrained weights; the action expert stays random
    else:
        # the LIBERO camera / state layout comes from the processor checkpoint
        features = cfg.input_features, cfg.output_features
        cfg = _pretrained_config(spec.weights, spec)
        cfg.input_features, cfg.output_features = features
    cfg.device = device
    for key, value in spec.config_overrides.items():
        setattr(cfg, key, value)
    if spec.weights is None:
        policy = policy_cls(cfg)
    else:
        if getattr(cfg, "compile_model", False):
            # some Hub checkpoints ship compile_model=True (max-autotune): it recompiles for every batch size the
            # evaluation pool sends and stalls for many minutes, so run eagerly
            cfg.compile_model = False
        revision = {"revision": spec.revision} if spec.revision is not None else {}
        policy = policy_cls.from_pretrained(spec.weights, config=cfg, **revision)
    policy = policy.to(device).eval().requires_grad_(False)
    pre, post = vla_processors(spec, cfg, device)
    return policy, pre, post


def _pretrained_config(repo: str, spec: VLASpec):
    """A checkpoint's policy config, with ``spec.legacy_config`` fields renamed / dropped first."""
    from lerobot.configs import PreTrainedConfig

    if not spec.legacy_config:
        return PreTrainedConfig.from_pretrained(repo, revision=spec.revision)
    import json
    import tempfile

    from huggingface_hub import hf_hub_download

    path = Path(repo) / "config.json"
    if not Path(repo).exists():
        path = Path(hf_hub_download(repo, "config.json", revision=spec.revision))
    raw = json.loads(path.read_text())
    for old, new in spec.legacy_config.items():
        if old in raw:
            value = raw.pop(old)
            if new is not None:
                raw[new] = value
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(raw))
        return PreTrainedConfig.from_pretrained(tmp)


def _load_groot(spec: VLASpec, device: str):
    """A raw GR00T N1.7 checkpoint as ``lerobot-eval --policy.type=groot --policy.base_model_path=<ckpt>``: config
    from the checkpoint, features from the LIBERO env, processors built from the checkpoint's embodiment metadata."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig

    cfg = GrootConfig(base_model_path=spec.weights, device=device, **spec.config_overrides)
    policy = make_policy(cfg, env_cfg=LiberoEnvConfig()).to(device).eval().requires_grad_(False)
    # another checkpoint's saved processors, if the spec names one
    pretrained = spec.processor if spec.processor != spec.weights else None
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=pretrained, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post


_RZ_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _axis_angle_to_matrix(aa: Tensor) -> Tensor:
    """Rodrigues' formula, batched: (..., 3) axis-angle -> (..., 3, 3) rotation matrices."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    kmat = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(kmat)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * kmat + (1 - c) * kmat @ kmat


class XVLAFromDatasetFormat:
    """Processor step: dataset-format LIBERO observations -> X-VLA's own LIBERO convention.

    Dataset format (``lerobot/libero``, the env's default ``LiberoProcessorStep``): both cameras rotated by 180°, an
    8-D state (end-effector position, axis-angle, gripper joints). X-VLA's ``LiberoProcessorStep`` rotates only the
    agent view and builds a 20-D state: [position, 6-D rotation, 0] followed by ten zeros.
    """

    def __init__(self) -> None:
        from lerobot.policies.xvla.processor_xvla import LiberoProcessorStep as XVLALiberoProcessorStep

        self._rot6d = XVLALiberoProcessorStep()._mat_to_rotate6d  # noqa: SLF001

    def __call__(self, transition):
        from lerobot.processor import TransitionKey

        obs = dict(transition[TransitionKey.OBSERVATION])
        wrist = f"{OBS_IMAGES}.image2"
        if wrist in obs:
            obs[wrist] = torch.flip(obs[wrist], dims=[-2, -1])
        state = obs.get(OBS_STATE)
        if state is not None and state.shape[-1] == 8:
            state = state.float()
            pos, aa = state[..., :3], state[..., 3:6]
            # the simulator's end-effector matrix (X-VLA's source) is its quaternion's (the dataset's) rotated by
            # -90° about z
            mat = _axis_angle_to_matrix(aa) @ _RZ_MINUS_90.to(aa)
            proprio = torch.cat([pos, self._rot6d(mat), torch.zeros_like(pos[..., :1])], dim=-1)
            obs[OBS_STATE] = torch.cat([proprio, torch.zeros_like(proprio)], dim=-1)
        return {**transition, TransitionKey.OBSERVATION: obs}

    # the rest of the processor-step interface: nothing to transform, save or reset
    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {}

    def reset(self) -> None:
        pass


def vla_processors(spec: VLASpec, cfg, device: str):
    """The LIBERO pre/post processors of a VLA spec."""
    from lerobot.policies.factory import make_pre_post_processors

    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=spec.processor,
        pretrained_revision=spec.revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    if spec.family == "xvla":
        from lerobot.policies.xvla.processor_xvla import (
            XVLAAddDomainIdProcessorStep,
            XVLAImageNetNormalizeProcessorStep,
        )

        # X-VLA's LIBERO processing is its own LiberoProcessorStep (not the env's default one) + these two steps.
        # Running them here, after a conversion from dataset format, keeps every consumer (rollouts, recordings,
        # embedder) on standard dataset-format observations.
        pre.steps = [
            XVLAFromDatasetFormat(),
            XVLAImageNetNormalizeProcessorStep(),
            XVLAAddDomainIdProcessorStep(),
            *pre.steps,
        ]
    return pre, post


# ---------------------------------------------------------------------------------------------- embedder
Pooled = dict[str, dict[str, Tensor]]  # pooling -> camera -> (B, D)

# family -> the text config of its VLM (the hidden size of the tokens the embedder pools)
_TEXT_CONFIG = {
    "smolvla": lambda policy: policy.model.vlm_with_expert.config.text_config,
    "xvla": lambda policy: policy.model.vlm.config.text_config,
    "vla_jepa": lambda policy: policy.model.qwen.model.config.text_config,
    "groot": lambda policy: policy._groot_model.backbone.model.config.text_config,  # noqa: SLF001
    "evo1": lambda policy: policy.model.embedder.model.config.text_config,
    "molmoact2": lambda policy: policy._backbone().config.text_config,  # noqa: SLF001
    "pi0_fast": lambda policy: policy.model.paligemma_with_expert.paligemma.config.text_config,
}
_PALIGEMMA_TEXT_CONFIG = _TEXT_CONFIG["pi0_fast"]  # pi0 and pi0.5 as well


# family -> the VLAEmbedder method that runs its prefix; pi0 and pi0.5 (the default) share one
_EMBED_METHODS = {
    "smolvla": "_smolvla",
    "xvla": "_xvla",
    "pi0_fast": "_pi0fast",
    "vla_jepa": "_vla_jepa",
    "groot": "_groot",
    "evo1": "_evo1",
    "molmoact2": "_molmoact2",
}


def _masked_mean(h: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


def _square_tokens(tokens: Tensor) -> Tensor:
    """Drop a leading summary token if that leaves a square patch grid."""
    p = tokens.shape[1]
    if int(round(p**0.5)) ** 2 != p and int(round((p - 1) ** 0.5)) ** 2 == p - 1:
        return tokens[:, 1:]
    return tokens


def _grid2(tokens: Tensor, shape: tuple[int, int] | None = None) -> Tensor:
    """(B, P, D) image tokens -> (B, 4 * D): the 2x2 average-pooled cells of their grid, flattened cell-major."""
    return grid_pool(tokens.float(), 2, shape).flatten(1)


class VLAEmbedder(nn.Module):
    """Pooled final-layer hidden states of a VLA's VLM prefix (see module docstring)."""

    def __init__(self, name: str, device: str = "cuda", taps: tuple[str, ...] = ()):
        """``taps``: extra hidden states pooled as ``grid2@<tap>``. ``"L<n>"`` is the residual stream after ``n``
        layers, ``"vision"`` the vision tower's image tokens as they enter the language model, and
        ``"L12+L15+last"`` a multi-layer key: every named layer is tapped and their ``grid2`` concatenated."""
        super().__init__()
        self.name = name
        layers = [layer for tap in taps for layer in tap.split("+")]
        self.taps = tuple(dict.fromkeys(layer for layer in layers if layer != "last"))
        self.combos = tuple(f"grid2@{tap}" for tap in taps if "+" in tap)
        self.spec = resolve_spec(name)
        self.policy, self.pre, _ = load_vla(name, device=device)
        self._pre_device = torch.device(device)
        # model camera order -> dataset camera names (inverse of the checkpoint's rename map)
        rename: dict[str, str] = {}
        for step in getattr(self.pre, "steps", []):
            rename.update(getattr(step, "rename_map", {}) or {})
        rename.update(self.spec.camera_rename)  # applied before the processors (_preprocess, VLARollout)
        inverse = {v: k for k, v in rename.items()}
        self.model_cameras = list(self.policy.config.image_features)
        self.cameras = [inverse.get(k, k) for k in self.model_cameras]
        self.present = [c for c in self.cameras if c in DATASET_CAMERAS]
        if self.spec.family == "groot":  # its processor takes the dataset's cameras under their own names
            self.cameras = self.present = list(DATASET_CAMERAS)
        text_config = _TEXT_CONFIG.get(self.spec.family, _PALIGEMMA_TEXT_CONFIG)(self.policy)
        # Florence-2 (X-VLA) names its width d_model
        self.hidden_size = int(text_config.d_model if self.spec.family == "xvla" else text_config.hidden_size)

    @property
    def poolings(self) -> tuple[str, ...]:
        base = ("img", "grid2", "lang", "all", *(f"grid2@{t}" for t in self.taps))
        return (*base, "state") if self.spec.family == "smolvla" else base

    def _device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        dev = self._device()
        if dev != self._pre_device:
            self.pre, _ = vla_processors(self.spec, self.policy.config, str(dev))
            self._pre_device = dev
        keep = {k: v for k, v in batch.items() if k in DATASET_CAMERAS or k in (OBS_STATE, "task")}
        keep = {self.spec.camera_rename.get(k, k): v for k, v in keep.items()}
        return self.pre(keep)

    @torch.no_grad()
    def forward(self, batch: dict[str, Any]) -> Pooled:
        """``batch``: dataset-format observations (images in [0, 1], raw 8-D state, ``task`` prompts)."""
        # pi0 shares the PaliGemma prefix path of pi0.5 (its state enters the suffix instead)
        embed = getattr(self, _EMBED_METHODS.get(self.spec.family, "_pi05"))
        out = embed(self._preprocess(batch))
        for combo in self.combos:  # multi-layer keys, e.g. "grid2@L12+L15+last"
            from .features import layer_parts

            parts = layer_parts(combo)
            out[combo] = {cam: torch.cat([out[p][cam] for p in parts], dim=1) for cam in out[parts[0]]}
        return out

    # ------------------------------------------------------------------ shared pieces
    @contextmanager
    def _tapped(self, layers, last: bool = False) -> Iterator[dict[str, Tensor]]:
        """Record, during a forward pass, the residual stream after ``n`` layers for every ``L<n>`` tap, and under
        ``"last"`` the output of the final layer (the first time it runs) if asked."""
        captured: dict[str, Tensor] = {}

        def record(key: str):
            def hook(_module, _inputs, out):
                hidden = out[0] if isinstance(out, tuple) else out
                if key == "last":
                    captured.setdefault(key, hidden)
                else:
                    captured[key] = hidden

            return hook

        # the output of layer n-1 is the residual stream after n layers
        hooks = [
            layers[int(t[1:]) - 1].register_forward_hook(record(t)) for t in self.taps if t.startswith("L")
        ]
        if last:
            hooks.append(layers[-1].register_forward_hook(record("last")))
        try:
            yield captured
        finally:
            for hook in hooks:
                hook.remove()

    def _pool_prefix(
        self,
        h: Tensor,
        pad: Tensor,
        tokens_per_image: int,
        n_lang: int,
        taps: dict[str, Tensor],
        state_token: bool,
    ) -> Pooled:
        """Poolings of a prefix laid out as [image tokens of every present camera, prompt (, state token)]."""
        out: Pooled = {"img": {}, "grid2": {}, "lang": {}, "all": {}}
        spans = {
            cam: slice(i * tokens_per_image, (i + 1) * tokens_per_image) for i, cam in enumerate(self.present)
        }
        for tap, hidden in taps.items():
            out[f"grid2@{tap}"] = {cam: _grid2(hidden[:, span]) for cam, span in spans.items()}
        for cam, span in spans.items():
            tokens = h[:, span].float()
            out["img"][cam] = tokens.mean(1)
            # the action head attends to every token, so also keep the image layout: 2x2 cells of the token grid
            out["grid2"][cam] = _grid2(tokens)
        start = len(self.present) * tokens_per_image
        lang = slice(start, start + n_lang)
        out["lang"][VLA_CAMERA] = _masked_mean(h[:, lang].float(), pad[:, lang])
        out["all"][VLA_CAMERA] = _masked_mean(h.float(), pad)
        if state_token:
            out["state"] = {VLA_CAMERA: h[:, start + n_lang].float()}
        return out

    @staticmethod
    def _pool_views(
        last: Tensor,
        taps: dict[str, Tensor],
        image_positions: Tensor,
        views: list[tuple[str, int, tuple[int, int]]],
        text_mask: Tensor,
        valid_mask: Tensor,
    ) -> Pooled:
        """Poolings of a sequence whose image tokens sit at ``image_positions`` (B, n), anywhere in the prompt.

        ``views``: (camera, index of its first image token, (height, width) of its token grid) per camera to keep.
        """

        def grid2(hidden: Tensor) -> dict[str, Tensor]:
            index = image_positions[..., None].expand(-1, -1, hidden.shape[-1])
            tokens = torch.gather(hidden.float(), 1, index)
            return {cam: _grid2(tokens[:, start : start + h * w], (h, w)) for cam, start, (h, w) in views}

        out: Pooled = {"grid2": grid2(last), "img": {}}
        for tap, hidden in taps.items():
            out[f"grid2@{tap}"] = grid2(hidden)
        for cam, cells in out["grid2"].items():
            out["img"][cam] = cells.view(len(cells), 4, -1).mean(1)
        out["lang"] = {VLA_CAMERA: _masked_mean(last.float(), text_mask)}
        out["all"] = {VLA_CAMERA: _masked_mean(last.float(), valid_mask)}
        return out

    def _dataset_camera(self, model_camera: str) -> str:
        """The dataset camera a model camera key reads (itself if the checkpoint does not rename it)."""
        if model_camera in self.model_cameras:
            return self.cameras[self.model_cameras.index(model_camera)]
        return model_camera

    # ------------------------------------------------------------------ one method per family
    def _smolvla(self, b: dict[str, Any]) -> Pooled:
        """SmolVLA: SmolVLM2 over [images, prompt, state token]."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        pol = self.policy
        images, img_masks = pol.prepare_images(b)
        state = pol.prepare_state(b)
        tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
        embs, pad, att = pol.model.embed_prefix(images, img_masks, tokens, masks, state=state)
        # cache-filling mode runs plain self-attention over the prefix (as in VLAFlowMatching.sample_actions);
        # its custom layer loop is tapped by recording each layer's input residual stream
        vlm = pol.model.vlm_with_expert
        wanted = {int(t[1:]): t for t in self.taps if t.startswith("L")}
        captured: dict[str, Tensor] = {}
        forward_layer = vlm.forward_attn_layer

        def tapped(model_layers, inputs_embeds, layer_idx, *args, **kwargs):
            if layer_idx in wanted:
                captured[wanted[layer_idx]] = inputs_embeds[0]
            return forward_layer(model_layers, inputs_embeds, layer_idx, *args, **kwargs)

        vlm.forward_attn_layer = tapped
        try:
            (h, _), _ = vlm.forward(
                attention_mask=make_att_2d_masks(pad, att),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=True,
            )
        finally:
            del vlm.forward_attn_layer
        if "vision" in self.taps:
            captured["vision"] = embs
        n_lang = tokens.shape[1]
        per_image = (embs.shape[1] - n_lang - 1) // len(images)
        if per_image * len(images) + n_lang + 1 != embs.shape[1]:
            raise RuntimeError("Unexpected SmolVLA prefix layout (prefix padding is not supported)")
        return self._pool_prefix(h, pad, per_image, n_lang, captured, state_token=True)

    def _paligemma_inputs(self, b: dict[str, Any]) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        """Images, image masks, prompt tokens and prompt mask of a PaliGemma prefix. Empty cameras are fully masked:
        dropping them (and trailing prompt padding) leaves every valid token's hidden state unchanged while saving
        most of the compute."""
        images, img_masks = self.policy._preprocess_images(b)  # noqa: SLF001
        images, img_masks = images[: len(self.present)], img_masks[: len(self.present)]
        tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
        n = int(masks.sum(1).max())
        return images, img_masks, tokens[:, :n], masks[:, :n]

    def _pi05(self, b: dict[str, Any]) -> Pooled:
        """pi0.5 and pi0: PaliGemma over [images, prompt] (pi0.5 writes the discretised state into the prompt)."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

        m = self.policy.model
        images, img_masks, tokens, masks = self._paligemma_inputs(b)
        embs, pad, att = m.embed_prefix(images, img_masks, tokens, masks)
        lm = m.paligemma_with_expert.paligemma.model.language_model
        lm.config._attn_implementation = "eager"  # noqa: SLF001  (as in PI05Pytorch.sample_actions)
        with self._tapped(lm.layers) as captured:
            (h, _), _ = m.paligemma_with_expert.forward(
                attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embs, None],
            )
        if "vision" in self.taps:
            captured["vision"] = embs
        n_lang = tokens.shape[1]
        return self._pool_prefix(
            h, pad, (embs.shape[1] - n_lang) // len(images), n_lang, captured, state_token=False
        )

    def _pi0fast(self, b: dict[str, Any]) -> Pooled:
        """pi0-FAST: PaliGemma over [images, prompt] (the prefix its FAST action tokens are decoded from)."""
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

        m = self.policy.model
        images, img_masks, tokens, masks = self._paligemma_inputs(b)
        embs, pad, att, n_img, _ = m.embed_prefix_fast(images, img_masks, tokens, masks)
        lm = m.paligemma_with_expert.paligemma.model.language_model
        if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            embs = embs.to(dtype=torch.bfloat16)
        with self._tapped(lm.layers) as captured:
            (h, _), _ = m.paligemma_with_expert.forward(
                attention_mask=prepare_attention_masks_4d(att, dtype=embs.dtype),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
        return self._pool_prefix(h, pad, n_img // len(images), tokens.shape[1], captured, state_token=False)

    def _xvla(self, b: dict[str, Any]) -> Pooled:
        """X-VLA: the Florence-2 encoder over [primary view's image tokens, prompt]; further views reach the policy as
        raw vision-tower tokens (``aux_visual_inputs``), so their key is those tokens."""
        m = self.policy.model
        inputs = self.policy._build_model_inputs(b)  # noqa: SLF001
        pixel = inputs["image_input"].to(dtype=m._get_target_dtype())  # noqa: SLF001
        with self._tapped(m.vlm.language_model.encoder.layers) as captured:
            enc = m.forward_vlm(inputs["input_ids"], pixel, inputs["image_mask"])
        h, aux = enc["vlm_features"], enc["aux_visual_inputs"]
        n_views = pixel.shape[1]
        per_view = aux.shape[1] // max(1, n_views - 1) if n_views > 1 else h.shape[1]
        views = [h[:, :per_view]] + [aux[:, i * per_view : (i + 1) * per_view] for i in range(n_views - 1)]
        tokens = {
            cam: _square_tokens(views[i].float()) for i, cam in enumerate(self.cameras) if cam in self.present
        }
        out: Pooled = {
            "img": {cam: t.mean(1) for cam, t in tokens.items()},
            "grid2": {cam: _grid2(t) for cam, t in tokens.items()},
            "lang": {VLA_CAMERA: h[:, per_view:].float().mean(1)},
            "all": {VLA_CAMERA: h.float().mean(1)},
        }
        primary = self.cameras[0]
        for (
            tap,
            hidden,
        ) in captured.items():  # encoder layers re-encode the primary view only; the others stay raw
            out[f"grid2@{tap}"] = dict(out["grid2"])
            if primary in tokens:
                out[f"grid2@{tap}"][primary] = _grid2(_square_tokens(hidden[:, :per_view].float()))
        return out

    def _vla_jepa(self, b: dict[str, Any]) -> Pooled:
        """VLA-JEPA: Qwen3-VL over [images, prompt]; its action head reads the last block's pre-norm output, which is
        what ``grid2`` pools (per view, on each image's merged-patch grid)."""
        m = self.policy.model
        inputs = self.policy._prepare_model_inputs(b, training=False)  # noqa: SLF001
        q = m.qwen.build_inputs(
            images=inputs["images"],
            instructions=inputs["instructions"],
            action_prompt=m.replace_prompt,
            embodied_prompt=m.embodied_replace_prompt,
        )
        with (
            self._tapped(m.qwen.model.model.language_model.layers, last=True) as captured,
            torch.autocast(device_type=self._device().type, dtype=torch.bfloat16),
        ):
            m.qwen.model.model(**q)
        cfg = m.qwen.model.config
        merge = cfg.vision_config.spatial_merge_size
        grid = q["image_grid_thw"]  # (B * views, 3), sample-major
        n_views = len(inputs["images"][0])
        is_image = q["input_ids"] == cfg.image_token_id
        views, start = [], 0
        for cam, g in zip(self.cameras[:n_views], grid[:n_views], strict=True):
            shape = (int(g[1]) // merge, int(g[2]) // merge)
            if cam in self.present:
                views.append((cam, start, shape))
            start += shape[0] * shape[1]
        return self._pool_views(
            captured.pop("last"),
            captured,
            is_image.nonzero(as_tuple=True)[1].view(len(grid) // n_views, -1),
            views,
            text_mask=~is_image & q["attention_mask"].bool(),
            valid_mask=q["attention_mask"],
        )

    def _groot(self, b: dict[str, Any]) -> Pooled:
        """GR00T N1.7: the Cosmos-Reason2 (Qwen3-VL) backbone over [images, prompt], truncated to the layer its action
        head reads (frozen in LIBERO fine-tuning); ``grid2`` pools that layer's image tokens per view."""
        model = self.policy._groot_model  # noqa: SLF001
        backbone = model.backbone
        inputs, _ = model.prepare_input(self.policy._filter_groot_inputs(b, include_action=False))  # noqa: SLF001
        with (
            self._tapped(backbone.language_model.layers) as captured,
            torch.autocast(device_type=self._device().type, dtype=torch.bfloat16),
        ):
            out = backbone(inputs)
        pack = next(step for step in self.pre.steps if hasattr(step, "_ordered_image_keys"))
        order = pack._ordered_image_keys(dict.fromkeys(self.present))  # noqa: SLF001  (the views' prompt order)
        merge = backbone.model.config.vision_config.spatial_merge_size
        grid = inputs["image_grid_thw"]  # (B * views, 3), sample-major
        views, start = [], 0
        for cam, g in zip(order, grid[: len(order)], strict=True):
            shape = (int(g[1]) // merge, int(g[2]) // merge)
            views.append((cam, start, shape))
            start += shape[0] * shape[1]
        valid = out["backbone_attention_mask"]
        return self._pool_views(
            out["backbone_features"],
            captured,
            out["image_mask"].nonzero(as_tuple=True)[1].view(len(grid) // len(order), -1),
            views,
            text_mask=valid & ~out["image_mask"],
            valid_mask=valid,
        )

    def _evo1(self, b: dict[str, Any]) -> Pooled:
        """Evo-1: InternVL3 (truncated to 14 layers) over [one image per camera, prompt]; its action head reads every
        token of the last layer's output, which ``grid2`` pools per view (each image: a square grid of tokens)."""
        pol = self.policy
        emb = pol.model.embedder
        images, masks = pol._collect_image_batches(b)  # noqa: SLF001
        prompts = pol._normalize_task_batch(b)  # noqa: SLF001
        input_ids: list[Tensor] = []
        record_ids = emb.model.register_forward_pre_hook(
            lambda _module, _args, kwargs: input_ids.append(kwargs["input_ids"]), with_kwargs=True
        )
        try:
            with (
                self._tapped(emb.model.language_model.layers) as captured,
                torch.autocast(device_type=self._device().type, dtype=torch.bfloat16),
            ):
                h, valid = emb.get_fused_image_text_embedding_batched(
                    images, masks, prompts, return_cls_only=False
                )
        finally:
            record_ids.remove()
        ids = input_ids[0]
        is_image = ids == emb.img_context_token_id
        n = emb.num_image_token
        side = int(round(n**0.5))
        model_cameras = list(pol._camera_keys or sorted(k for k in b if k.startswith(f"{OBS_IMAGES}.")))  # noqa: SLF001
        views = [(self._dataset_camera(cam), v * n, (side, side)) for v, cam in enumerate(model_cameras)]
        return self._pool_views(
            h,
            captured,
            is_image.nonzero(as_tuple=True)[1].view(len(ids), -1),
            [view for view in views if view[0] in self.present],
            text_mask=valid & ~is_image,
            valid_mask=valid,
        )

    def _molmoact2(self, b: dict[str, Any]) -> Pooled:
        """MolmoAct2: the Molmo2 decoder over [per camera: a low-res and a high-res grid of image patches, prompt]; its
        action expert attends to every layer's keys and values. ``grid2`` pools each camera's high-res patch grid
        (the low-res one if it has none) at the last layer's output."""
        pol = self.policy
        backbone = pol._backbone()  # noqa: SLF001
        inputs = pol._model_inputs(b)  # noqa: SLF001
        with self._tapped(backbone.transformer.blocks, last=True) as captured, pol._autocast_context():  # noqa: SLF001
            backbone(**inputs, use_cache=False)
        ids = inputs["input_ids"]
        is_patch = ids == backbone.config.image_patch_id
        grids = inputs["image_grids"].reshape(len(ids), -1, 4)  # per camera: low-res h, w, high-res h, w
        views, start = [], 0
        for v, model_camera in enumerate(pol.config.image_features):  # the camera order of the prompt
            low_h, low_w, high_h, high_w = (int(x) for x in grids[0, v])
            low, high = low_h * low_w, high_h * high_w
            cam = self._dataset_camera(model_camera)
            if cam in self.present:
                views.append((cam, start + low, (high_h, high_w)) if high else (cam, start, (low_h, low_w)))
            start += low + high
        if "attention_mask" in inputs:
            valid = inputs["attention_mask"].bool()
        else:
            valid = torch.ones_like(ids, dtype=torch.bool)
        return self._pool_views(
            captured.pop("last"),
            captured,
            is_patch.nonzero(as_tuple=True)[1].view(len(ids), -1),
            views,
            text_mask=valid & ~is_patch,
            valid_mask=valid,
        )


# ---------------------------------------------------------------------------------------------- rollouts
@dataclass
class _Slot:
    """Per-environment rollout state: queued actions, step, episode counter."""

    queue: deque = field(default_factory=deque)
    t: int = 0
    episode: int = -1


class VLARollout:
    """Wraps a LeRobot VLA for ``LiberoPool``: per-environment action queues, optional chunk recording.

    The pool hands over dataset-format observations; this class applies the VLA's own pre/post processors
    (so pass identity processors to ``LiberoPool.evaluate``) and keeps *unnormalised* actions queued.
    With ``record_init_states`` every re-planning step of those initial states stores the observation (uint8
    images, state, prompt, step) and the full predicted chunk for offline analysis.
    """

    def __init__(
        self,
        policy,
        pre,
        post,
        n_action_steps: int,
        record_init_states: set[int] | None = None,
        action_transform=None,
        camera_rename: dict[str, str] | None = None,
    ):
        """``action_transform``: (B, A) -> (B, 7) applied to every predicted action before it is queued (e.g. X-VLA's
        6D-rotation -> axis-angle LIBERO conversion, which the pool's queue-pop path would otherwise skip)."""
        self.policy, self.pre, self.post = policy, pre, post
        self.action_transform = action_transform
        self.camera_rename = camera_rename or {}  # dataset camera key -> the key the policy expects
        self.failed_predictions = 0
        self.n_action_steps = n_action_steps
        self.record_init_states = record_init_states
        self.records: list[dict[str, Any]] = []  # finished episodes of the recorded initial states
        self._open: dict[int, list[dict[str, Any]]] = {}  # episode -> its records so far
        self._slots: dict[int, _Slot] = {}
        self._next_episode = 0

    def reset(self) -> None:
        self._slots.clear()
        self.policy.reset()

    def _slot(self, env_id: int) -> _Slot:
        return self._slots.setdefault(env_id, _Slot())

    def reset_envs(self, env_ids: list[int]) -> None:
        for env_id in env_ids:
            slot = self._slot(env_id)
            slot.queue.clear()
            slot.t = 0
            slot.episode = self._next_episode
            self._next_episode += 1

    def episode_done(self, env_id: int, result: Any) -> None:
        records = self._open.pop(self._slot(env_id).episode, [])
        if self.record_init_states is not None and result.init_state in self.record_init_states:
            for record in records:
                record.update(
                    suite=result.suite,
                    task_id=result.task_id,
                    init_state=result.init_state,
                    success=result.success,
                    episode_steps=result.steps,
                )
            self.records.extend(records)

    def needs_observation(self, env_ids: list[int]) -> list[bool]:
        return [not self._slot(env_id).queue for env_id in env_ids]

    def _predict_batch(self, batch: dict[str, Any]) -> Tensor:
        self.policy.reset()  # clear any batch-level queues; chunks are managed here
        return self.post(self.policy.predict_action_chunk(self.pre(batch))).float().cpu()

    def _predict(self, sub: dict[str, Any]) -> Tensor:
        """Predict a chunk per row; a row whose prediction fails (e.g. an invalid FAST token sequence) holds still."""
        if self.camera_rename:
            sub = {self.camera_rename.get(k, k): v for k, v in sub.items()}
        try:
            return self._predict_batch(dict(sub))
        except Exception:  # noqa: BLE001 - retry row by row, then fall back to a zero (hold-still) chunk
            chunks: list[Tensor | None] = []
            for i in range(len(sub["task"])):
                row = {k: (v[i : i + 1] if torch.is_tensor(v) else [v[i]]) for k, v in sub.items()}
                try:
                    chunks.append(self._predict_batch(row))
                except Exception:  # noqa: BLE001
                    self.failed_predictions += 1
                    chunks.append(None)
            good = next((c for c in reversed(chunks) if c is not None), None)
            if good is None:
                raise
            return torch.cat([c if c is not None else torch.zeros_like(good) for c in chunks])

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], env_ids: list[int]) -> Tensor:
        slots = [self._slot(env_id) for env_id in env_ids]
        need = [i for i, slot in enumerate(slots) if not slot.queue]
        if need:
            sub = {
                k: (v[need] if torch.is_tensor(v) else [v[i] for i in need])
                for k, v in batch.items()
                if torch.is_tensor(v) or k == "task"
            }
            chunk = self._predict(sub)  # (B, C, A) unnormalised
            if self.action_transform is not None:
                chunk = self.action_transform(chunk.flatten(0, 1)).unflatten(0, chunk.shape[:2]).float().cpu()
            for row, i in enumerate(need):
                slots[i].queue.extend(chunk[row, : self.n_action_steps])
                if self.record_init_states is not None:
                    self._open.setdefault(slots[i].episode, []).append(
                        self._record(slots[i], sub, row, chunk)
                    )
        actions = []
        for slot in slots:
            actions.append(slot.queue.popleft())
            slot.t += 1
        return torch.stack(actions)

    @staticmethod
    def _record(slot: _Slot, sub: dict[str, Any], row: int, chunk: Tensor) -> dict[str, Any]:
        """One re-planning step: the observation (uint8 images) and the chunk predicted from it."""
        images = [(sub[cam][row].clamp(0, 1) * 255).round().byte().cpu().numpy() for cam in DATASET_CAMERAS]
        return {
            "episode": slot.episode,
            "t": slot.t,
            "task": sub["task"][row],
            "state": sub[OBS_STATE][row].float().cpu().numpy(),
            "images": np.stack(images),
            "chunk": chunk[row].numpy(),
        }


class Identity:
    """No-op processor with the ``reset`` hook ``LiberoPool.evaluate`` expects."""

    def __call__(self, x: Any) -> Any:
        return x

    def reset(self) -> None:
        pass
