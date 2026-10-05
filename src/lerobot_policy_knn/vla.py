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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .encoders import _grid_pool

logger = logging.getLogger(__name__)

VLA_CAMERA = "observation.vla"
DATASET_CAMERAS = ("observation.images.image", "observation.images.image2")
VLA_ROOT = Path("outputs/vla")


@dataclass(frozen=True)
class VLASpec:
    family: str  # "smolvla" | "pi05"
    weights: str | None  # hub repo or local dir; None = VLM initialised from its original (non-robot) weights
    processor: str  # where the LIBERO pre/post processors (normalisation stats, tokenizer) come from
    note: str = ""
    # each family's documented LIBERO evaluation settings (docs/source/<policy>.mdx)
    config_overrides: dict[str, Any] = field(default_factory=dict)  # policy config fields set before loading
    revision: str | None = None  # pinned Hub revision of weights and processors
    camera_rename: dict[str, str] = field(
        default_factory=dict
    )  # dataset camera key -> the key the policy expects
    observation_size: int = 256  # LIBERO render size its evaluation uses
    n_action_steps: int | None = (
        None  # its documented re-planning interval (None: the caller's; 0: its config's)
    )
    # config fields of an older checkpoint that current LeRobot renamed (old -> new) or removed (old -> None)
    legacy_config: dict[str, str | None] = field(default_factory=dict)


SMOLVLA_LIBERO = "lerobot/smolvla_libero"
VLA_SPECS: dict[str, VLASpec] = {
    "smolvla_base": VLASpec(
        "smolvla", "lerobot/smolvla_base", SMOLVLA_LIBERO, "SmolVLA pretrained, no LIBERO"
    ),
    "smolvla_libero": VLASpec("smolvla", SMOLVLA_LIBERO, SMOLVLA_LIBERO, "SmolVLA fine-tuned on LIBERO"),
    "pi05_base": VLASpec(
        "pi05",
        str(VLA_ROOT / "pi05_base_bf16"),
        "lerobot/pi05_libero_finetuned_v044",
        "pi0.5 base, no LIBERO",
    ),
    "pi05_libero": VLASpec(
        "pi05",
        "lerobot/pi05_libero_finetuned_v044",
        "lerobot/pi05_libero_finetuned_v044",
        "pi0.5 fine-tuned on LIBERO",
    ),
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
    # GR00T N1.7 (Cosmos-Reason2 / Qwen3-VL backbone): one LIBERO checkpoint per suite. These are LeRobot checkpoints
    # (weights + serialized processors holding the normalization statistics); their config's base_model_path is a
    # path on NVIDIA's machine, so the architecture is built from the checkpoint repo itself before its weights load.
    # 8 of the 16 predicted actions run before replanning, as in NVIDIA's LIBERO rollout wrapper.
    **{
        f"groot_{suite}": VLASpec(
            "groot",
            f"nvidia/gr00t17-lerobot-{suite}-640",
            f"nvidia/gr00t17-lerobot-{suite}-640",
            f"GR00T N1.7 fine-tuned on {suite}",
            config_overrides={"base_model_path": f"nvidia/gr00t17-lerobot-{suite}-640"},
            n_action_steps=8,
        )
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    },
    # the raw base checkpoint with a LIBERO checkpoint's saved processors: its VLM is bit-identical to the one every
    # LIBERO checkpoint above keeps (they froze it), so this is GR00T's key, on the images its LIBERO policies see
    "groot_base": VLASpec(
        "groot",
        "nvidia/GR00T-N1.7-3B",
        "nvidia/gr00t17-lerobot-libero_spatial-640",
        "GR00T N1.7 base",
        config_overrides={"embodiment_tag": "libero_sim"},
    ),
    # MolmoAct2 (LeRobot port): wrist camera as "wrist_image", continuous actions in bfloat16
    "molmoact2_libero": VLASpec(
        "molmoact2",
        "allenai/MolmoAct2-LIBERO-LeRobot",
        "allenai/MolmoAct2-LIBERO-LeRobot",
        "MolmoAct2 LIBERO",
        config_overrides={
            "inference_action_mode": "continuous",
            # the checkpoint is a full fine-tune (enable_lora_vlm=False): no LoRA wrapping, or its keys do not load
            "train_mode_vlm": "fft",
            "dtype": "bfloat16",
            "enable_inference_cuda_graph": False,
        },  # fmt: skip
        camera_rename={"observation.images.image2": "observation.images.wrist_image"},
        n_action_steps=0,
        # the Hub config predates #4249: training-only switches (all off) and the dtype field's old name
        legacy_config={
            "enable_lora_vlm": None,
            "enable_lora_action_expert": None,
            "train_action_expert_only": None,
            "model_dtype": "dtype",
        },  # fmt: skip
    ),
    # the released base, built from its HF checkpoint with the LIBERO fine-tune's processors and camera layout
    "molmoact2_base": VLASpec(
        "molmoact2",
        None,
        "allenai/MolmoAct2-LIBERO-LeRobot",
        "MolmoAct2 before LIBERO (allenai/MolmoAct2)",
        config_overrides={
            "inference_action_mode": "continuous",
            # the checkpoint is a full fine-tune (enable_lora_vlm=False): no LoRA wrapping, or its keys do not load
            "train_mode_vlm": "fft",
            "dtype": "bfloat16",
            "enable_inference_cuda_graph": False,
            "checkpoint_path": "allenai/MolmoAct2",  # the network is built from this HF checkpoint
        },  # fmt: skip
        camera_rename={"observation.images.image2": "observation.images.wrist_image"},
        n_action_steps=0,
        # the Hub config predates #4249: training-only switches (all off) and the dtype field's old name
        legacy_config={
            "enable_lora_vlm": None,
            "enable_lora_action_expert": None,
            "train_action_expert_only": None,
            "model_dtype": "dtype",
        },  # fmt: skip
    ),
    # Evo-1 (InternVL3-1B backbone): the verified Stage-2 checkpoint at 448 px renders, 14 actions per re-plan
    "evo1_libero": VLASpec(
        "evo1",
        "zuoxingdong/evo1_libero",
        "zuoxingdong/evo1_libero",
        "Evo-1 LIBERO",
        revision="515921f4a2c1d3f3ad523721eafa26fdf2af315b",
        config_overrides={
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
        },  # fmt: skip
        observation_size=448,
        n_action_steps=14,
    ),
    # no Evo-1 checkpoint before LIBERO: its VLM as published, built from evo1_libero's config (action head unused)
    "evo1_base": VLASpec(
        "evo1",
        None,
        "zuoxingdong/evo1_libero",
        "Evo-1's VLM before LIBERO: the raw InternVL3-1B (its first 14 layers)",
        revision="515921f4a2c1d3f3ad523721eafa26fdf2af315b",
        config_overrides={
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
        },  # fmt: skip
        observation_size=448,
        n_action_steps=14,
    ),
    "pi0_base": VLASpec(
        "pi0", str(VLA_ROOT / "pi0_base_bf16"), "lerobot/pi0_libero_finetuned_v044", "pi0 base"
    ),
    "pi0_libero": VLASpec(
        "pi0", "lerobot/pi0_libero_finetuned_v044", "lerobot/pi0_libero_finetuned_v044", "pi0 LIBERO"
    ),
}


def resolve_spec(name: str) -> VLASpec:
    """Registry name, or ``smolvla@<checkpoint dir>`` / ``pi05@<checkpoint dir>`` for local checkpoints."""
    if name in VLA_SPECS:
        return VLA_SPECS[name]
    if "@" in name:
        # <family>@<local checkpoint dir or Hub repo>; its own processors unless a local dir lacks them
        family, path = name.split("@", 1)
        local = Path(path).exists()
        processor = (
            path if (not local or (Path(path) / "policy_preprocessor.json").exists()) else SMOLVLA_LIBERO
        )
        return VLASpec(family, path, processor)
    raise KeyError(f"Unknown VLA embedder {name!r}; known: {sorted(VLA_SPECS)}")


def cache_tag(name: str) -> str:
    """Feature-cache directory name of a VLA embedder."""
    if "@" in name:
        family, path = name.split("@", 1)
        p = Path(path)
        # .../checkpoints/<step>/pretrained_model -> <run>_<step>
        parts = [x for x in p.parts if x not in ("pretrained_model", "checkpoints")]
        name = f"{family}_{'_'.join(parts[-2:])}"
    return "vla_" + name.replace("/", "_")


def load_vla(name: str, device: str = "cuda", dtype: str | None = None):
    """Return ``(policy, preprocessor, postprocessor)`` of a registered VLA."""
    from lerobot.policies.factory import get_policy_class

    spec = resolve_spec(name)
    if spec.family == "groot" and spec.weights == "nvidia/GR00T-N1.7-3B":  # a raw NVIDIA checkpoint
        return _load_groot(spec, device)
    proc_cfg = _pretrained_config(spec.processor, spec)
    cls = get_policy_class(spec.family)
    if spec.weights is None:
        cfg = proc_cfg
        cfg.load_vlm_weights = True  # the VLM's own pretrained weights; action expert stays random
        cfg.device = device
        for key, value in spec.config_overrides.items():
            setattr(cfg, key, value)
        policy = cls(cfg)
    else:
        cfg = _pretrained_config(spec.weights, spec)
        # the LIBERO camera/state layout and processors come from the processor checkpoint
        cfg.input_features = proc_cfg.input_features
        cfg.output_features = proc_cfg.output_features
        cfg.device = device
        if dtype is not None and hasattr(cfg, "dtype"):
            cfg.dtype = dtype
        for key, value in spec.config_overrides.items():
            setattr(cfg, key, value)
        if getattr(cfg, "compile_model", False):
            # some Hub checkpoints ship compile_model=True (max-autotune): it recompiles for every batch size the
            # evaluation pool sends and stalls for many minutes, so run eagerly
            cfg.compile_model = False
        kwargs: dict[str, Any] = {}
        if spec.revision is not None:
            kwargs["revision"] = spec.revision
        policy = cls.from_pretrained(spec.weights, config=cfg, **kwargs)
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

    path = (
        Path(repo) / "config.json"
        if Path(repo).exists()
        else Path(hf_hub_download(repo, "config.json", revision=spec.revision))
    )
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
    pretrained = (
        spec.processor if spec.processor != spec.weights else None
    )  # another checkpoint's saved processors
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=pretrained, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post


class XVLAFromDatasetFormat:
    """Dataset-format LIBERO observations -> X-VLA's own LIBERO convention.

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
            # the simulator's end-effector matrix (X-VLA's source) is its quaternion's (the dataset's) rotated by -90°
            # about z
            mat = _axis_angle_to_matrix(aa) @ _RZ_MINUS_90.to(aa)
            proprio = torch.cat([pos, self._rot6d(mat), torch.zeros_like(pos[..., :1])], dim=-1)
            obs[OBS_STATE] = torch.cat([proprio, torch.zeros_like(proprio)], dim=-1)
        return {**transition, TransitionKey.OBSERVATION: obs}

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {}

    def reset(self) -> None:
        pass


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


def _masked_mean(h: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class VLAEmbedder(nn.Module):
    """Pooled final-layer hidden states of a VLA's VLM prefix (see module docstring)."""

    def __init__(self, name: str, device: str = "cuda", taps: tuple[str, ...] = ()):
        """``taps``: extra hidden states pooled as ``grid2@<tap>``: ``"L<n>"`` = the residual stream after ``n`` layers,
        ``"vision"`` = the vision tower's image tokens as they enter the language model."""
        super().__init__()
        self.name = name
        # "L12+L15+last" style taps request a multi-layer key: tap each named layer and concatenate them
        self.combos = tuple(t for t in taps if "+" in t)
        singles = [x for t in taps for x in (t.split("+") if "+" in t else [t])]
        self.taps = tuple(dict.fromkeys(x for x in singles if x != "last"))
        self.combos = tuple(f"grid2@{c}" for c in self.combos)
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
        self.hidden_size = self._hidden_size()

    @property
    def poolings(self) -> tuple[str, ...]:
        base = ("img", "grid2", "lang", "all", *(f"grid2@{t}" for t in self.taps))
        return (*base, "state") if self.spec.family == "smolvla" else base

    def _hidden_size(self) -> int:
        if self.spec.family == "smolvla":
            return int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)
        if self.spec.family == "xvla":
            return int(self.policy.model.vlm.config.text_config.d_model)
        if self.spec.family == "vla_jepa":
            return int(self.policy.model.qwen.model.config.text_config.hidden_size)
        if self.spec.family == "groot":
            return int(self.policy._groot_model.backbone.model.config.text_config.hidden_size)  # noqa: SLF001
        if self.spec.family == "evo1":
            return int(self.policy.model.embedder.model.config.text_config.hidden_size)
        if self.spec.family == "molmoact2":
            return int(self.policy._backbone().config.text_config.hidden_size)  # noqa: SLF001
        return int(self.policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size)

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
    def forward(self, batch: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """``batch``: dataset-format observations (images in [0, 1], raw 8-D state, ``task`` prompts)."""
        b = self._preprocess(batch)
        if self.spec.family == "smolvla":
            out = self._smolvla(b)
        elif self.spec.family == "xvla":
            out = self._xvla(b)
        elif self.spec.family == "pi0_fast":
            out = self._pi0fast(b)
        elif self.spec.family == "vla_jepa":
            out = self._vla_jepa(b)
        elif self.spec.family == "groot":
            out = self._groot(b)
        elif self.spec.family == "evo1":
            out = self._evo1(b)
        elif self.spec.family == "molmoact2":
            out = self._molmoact2(b)
        else:  # pi0 shares the PaliGemma prefix path of pi0.5 (its state enters the suffix instead)
            out = self._pi05(b)
        for combo in self.combos:  # multi-layer keys, e.g. "grid2@L12+L15+last"
            from .features import layer_parts

            parts = layer_parts(combo)
            out[combo] = {cam: torch.cat([out[p][cam] for p in parts], dim=1) for cam in out[parts[0]]}
        return out

    def _pool(
        self,
        h: Tensor,
        pad: Tensor,
        n_img: list[int],
        n_lang: int,
        state_token: bool,
        taps: dict[str, Tensor] | None = None,
    ) -> dict[str, dict[str, Tensor]]:
        out: dict[str, dict[str, Tensor]] = {"img": {}, "grid2": {}, "lang": {}, "all": {}}
        start = 0
        for tap, ht in (taps or {}).items():
            out[f"grid2@{tap}"] = {}
            s0 = 0
            for cam, n in zip(self.present, n_img, strict=True):
                out[f"grid2@{tap}"][cam] = _grid_pool(ht[:, s0 : s0 + n].float(), 2).flatten(1)
                s0 += n
        for cam, n in zip(self.present, n_img, strict=True):
            tokens = h[:, start : start + n].float()
            out["img"][cam] = tokens.mean(1)
            # the action head attends to every token, so also keep the image layout: 2x2 cells of the token grid
            out["grid2"][cam] = _grid_pool(tokens, 2).flatten(1)
            start += n
        lang = slice(start, start + n_lang)
        out["lang"][VLA_CAMERA] = _masked_mean(h[:, lang].float(), pad[:, lang])
        out["all"][VLA_CAMERA] = _masked_mean(h.float(), pad)
        if state_token:
            out["state"] = {VLA_CAMERA: h[:, start + n_lang].float()}
        return out

    def _smolvla(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        pol = self.policy
        m = pol.model
        images, img_masks = pol.prepare_images(b)
        state = pol.prepare_state(b)
        tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
        embs, pad, att = m.embed_prefix(images, img_masks, tokens, masks, state=state)
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        # cache-filling mode runs plain self-attention over the prefix (as in VLAFlowMatching.sample_actions);
        # its custom layer loop is tapped by recording each layer's input residual stream
        vwe = m.vlm_with_expert
        want = {int(t[1:]): t for t in self.taps if t.startswith("L")}
        captured: dict[str, Tensor] = {}
        orig = vwe.forward_attn_layer

        def tapped(model_layers, inputs_embeds, layer_idx, *args, **kwargs):
            if layer_idx in want:
                captured[want[layer_idx]] = inputs_embeds[0]
            return orig(model_layers, inputs_embeds, layer_idx, *args, **kwargs)

        vwe.forward_attn_layer = tapped
        try:
            (h, _), _ = vwe.forward(
                attention_mask=att2d,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=True,
            )
        finally:
            del vwe.forward_attn_layer
        if "vision" in self.taps:
            captured["vision"] = embs
        n_lang = tokens.shape[1]
        per_img = (embs.shape[1] - n_lang - 1) // len(images)
        if per_img * len(images) + n_lang + 1 != embs.shape[1]:
            raise RuntimeError("Unexpected SmolVLA prefix layout (prefix padding is not supported)")
        return self._pool(h, pad, [per_img] * len(self.present), n_lang, state_token=True, taps=captured)

    def _pi05(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

        pol = self.policy
        m = pol.model
        images, img_masks = pol._preprocess_images(b)
        # empty cameras are fully masked: dropping them (and trailing prompt padding) leaves every valid
        # token's hidden state unchanged while saving most of the compute
        images, img_masks = images[: len(self.present)], img_masks[: len(self.present)]
        tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
        n = int(masks.sum(1).max())
        tokens, masks = tokens[:, :n], masks[:, :n]
        embs, pad, att = m.embed_prefix(images, img_masks, tokens, masks)
        att4d = prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1
        lm = m.paligemma_with_expert.paligemma.model.language_model
        lm.config._attn_implementation = "eager"  # noqa: SLF001  (as in PI05Pytorch.sample_actions)
        captured: dict[str, Tensor] = {}
        hooks = []
        for t in self.taps:
            if t.startswith("L"):  # output of layer n-1 = the residual stream after n layers

                def hook(_mod, _inp, out, t=t):
                    captured[t] = out[0] if isinstance(out, tuple) else out

                hooks.append(lm.layers[int(t[1:]) - 1].register_forward_hook(hook))
        try:
            (h, _), _ = m.paligemma_with_expert.forward(
                attention_mask=att4d, position_ids=pos, past_key_values=None, inputs_embeds=[embs, None]
            )
        finally:
            for hk in hooks:
                hk.remove()
        if "vision" in self.taps:
            captured["vision"] = embs
        per_img = (embs.shape[1] - n) // len(images)
        return self._pool(h, pad, [per_img] * len(self.present), n, state_token=False, taps=captured)

    def _xvla(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """X-VLA: the Florence-2 encoder over [primary view's image tokens, prompt]; further views reach the policy as
        raw vision-tower tokens (``aux_visual_inputs``), so their key is those tokens."""
        pol = self.policy
        m = pol.model
        inputs = pol._build_model_inputs(b)  # noqa: SLF001
        pixel = inputs["image_input"].to(dtype=m._get_target_dtype())  # noqa: SLF001
        captured: dict[str, Tensor] = {}
        hooks = []
        for t in self.taps:
            if t.startswith("L"):

                def hook(_mod, _inp, out, t=t):
                    captured[t] = out[0] if isinstance(out, tuple) else out

                hooks.append(m.vlm.language_model.encoder.layers[int(t[1:]) - 1].register_forward_hook(hook))
        try:
            enc = m.forward_vlm(inputs["input_ids"], pixel, inputs["image_mask"])
        finally:
            for hk in hooks:
                hk.remove()
        h, aux = enc["vlm_features"], enc["aux_visual_inputs"]
        n_views = pixel.shape[1]
        per_view = aux.shape[1] // max(1, n_views - 1) if n_views > 1 else h.shape[1]
        views = [h[:, :per_view]] + [aux[:, i * per_view : (i + 1) * per_view] for i in range(n_views - 1)]
        cam_view = {cam: views[self.model_cameras.index(model_cam)]
                    for cam, model_cam in zip(self.cameras, self.model_cameras, strict=True) if cam in self.present}  # fmt: skip
        out: dict[str, dict[str, Tensor]] = {"img": {}, "grid2": {}, "lang": {}, "all": {}}
        for cam, tokens in cam_view.items():
            tokens = _square_tokens(tokens.float())
            out["img"][cam] = tokens.mean(1)
            out["grid2"][cam] = _grid_pool(tokens, 2).flatten(1)
        for (
            tap,
            ht,
        ) in captured.items():  # encoder layers re-encode the primary view only; other views stay raw
            out[f"grid2@{tap}"] = {cam: out["grid2"][cam] for cam in cam_view}
            primary = self.cameras[0]
            if primary in cam_view:
                out[f"grid2@{tap}"][primary] = _grid_pool(
                    _square_tokens(ht[:, :per_view].float()), 2
                ).flatten(1)
        out["lang"][VLA_CAMERA] = h[:, per_view:].float().mean(1)
        out["all"][VLA_CAMERA] = h.float().mean(1)
        return out

    def _hook_layers(self, layers, captured: dict[str, Tensor]) -> list:
        """Forward hooks recording the residual stream after layer n for every ``L<n>`` tap."""
        hooks = []
        for t in self.taps:
            if t.startswith("L"):

                def hook(_mod, _inp, out, t=t):
                    captured[t] = out[0] if isinstance(out, tuple) else out

                hooks.append(layers[int(t[1:]) - 1].register_forward_hook(hook))
        return hooks

    def _pi0fast(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """pi0-FAST: PaliGemma over [images, prompt] (the prefix its FAST action tokens are decoded from)."""
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d

        pol = self.policy
        m = pol.model
        images, img_masks = pol._preprocess_images(b)  # noqa: SLF001
        images, img_masks = images[: len(self.present)], img_masks[: len(self.present)]
        tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
        n = int(masks.sum(1).max())
        tokens, masks = tokens[:, :n], masks[:, :n]
        embs, pad, att, n_img, _ = m.embed_prefix_fast(images, img_masks, tokens, masks)
        lm = m.paligemma_with_expert.paligemma.model.language_model
        if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            embs = embs.to(dtype=torch.bfloat16)
        captured: dict[str, Tensor] = {}
        hooks = self._hook_layers(lm.layers, captured)
        try:
            (h, _), _ = m.paligemma_with_expert.forward(
                attention_mask=prepare_attention_masks_4d(att, dtype=embs.dtype),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
        finally:
            for hk in hooks:
                hk.remove()
        return self._pool(
            h, pad, [n_img // len(images)] * len(self.present), n, state_token=False, taps=captured
        )

    def _vla_jepa(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """VLA-JEPA: Qwen3-VL over [images, prompt]; its action head reads the last block's pre-norm output, which is
        what ``grid2`` pools (per view, on each image's merged-patch grid)."""
        pol = self.policy
        m = pol.model
        inputs = pol._prepare_model_inputs(b, training=False)  # noqa: SLF001
        q = m.qwen.build_inputs(images=inputs["images"], instructions=inputs["instructions"],
                                action_prompt=m.replace_prompt, embodied_prompt=m.embodied_replace_prompt)  # fmt: skip
        layers = m.qwen.model.model.language_model.layers
        captured: dict[str, Tensor] = {}
        hooks = self._hook_layers(layers, captured)
        last: list[Tensor] = []
        hooks.append(
            layers[-1].register_forward_hook(
                lambda _m, _i, out: last.append(out[0] if isinstance(out, tuple) else out)
            )
        )
        try:
            with torch.autocast(device_type=self._device().type, dtype=torch.bfloat16):
                m.qwen.model.model(**q)
        finally:
            for hk in hooks:
                hk.remove()
        cfg = m.qwen.model.config
        merge = cfg.vision_config.spatial_merge_size
        grid = q["image_grid_thw"]  # (B * views, 3), sample-major
        n_views = len(inputs["images"][0])
        sides = [(int(g[1]) // merge, int(g[2]) // merge) for g in grid[:n_views]]
        img_pos = (
            (q["input_ids"] == cfg.image_token_id).nonzero(as_tuple=True)[1].view(len(grid) // n_views, -1)
        )
        txt = (q["input_ids"] != cfg.image_token_id) & q["attention_mask"].bool()

        def per_view(hs: Tensor) -> dict[str, Tensor]:
            toks = torch.gather(hs.float(), 1, img_pos[..., None].expand(-1, -1, hs.shape[-1]))
            out, s0 = {}, 0
            for cam, (gh, gw) in zip(self.cameras[:n_views], sides, strict=True):
                v = toks[:, s0 : s0 + gh * gw]
                s0 += gh * gw
                if cam in self.present:  # (B, 2*2 cells, D) flattened cell-major, as _grid_pool
                    fmap = v.transpose(1, 2).reshape(len(v), -1, gh, gw)
                    cells = torch.nn.functional.adaptive_avg_pool2d(fmap, 2).flatten(2).transpose(1, 2)
                    out[cam] = cells.flatten(1)
            return out

        h = last[0]
        res: dict[str, dict[str, Tensor]] = {"grid2": per_view(h), "img": {}}
        for tap, ht in captured.items():
            res[f"grid2@{tap}"] = per_view(ht)
        for cam, v in res["grid2"].items():
            res["img"][cam] = v.view(len(v), 4, -1).mean(1)
        res["lang"] = {VLA_CAMERA: _masked_mean(h.float(), txt)}
        res["all"] = {VLA_CAMERA: _masked_mean(h.float(), q["attention_mask"])}
        return res

    def _groot(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """GR00T N1.7: the Cosmos-Reason2 (Qwen3-VL) backbone over [images, prompt], truncated to the layer its action
        head reads (frozen in LIBERO fine-tuning); ``grid2`` pools that layer's image tokens per view."""
        pol = self.policy
        model = pol._groot_model  # noqa: SLF001
        backbone = model.backbone
        inputs, _ = model.prepare_input(pol._filter_groot_inputs(b, include_action=False))  # noqa: SLF001
        captured: dict[str, Tensor] = {}
        hooks = self._hook_layers(backbone.language_model.layers, captured)
        try:
            with torch.autocast(device_type=self._device().type, dtype=torch.bfloat16):
                out = backbone(inputs)
        finally:
            for hk in hooks:
                hk.remove()
        pack = next(step for step in self.pre.steps if hasattr(step, "_ordered_image_keys"))
        order = pack._ordered_image_keys(dict.fromkeys(self.present))  # noqa: SLF001  (the views' prompt order)
        merge = backbone.model.config.vision_config.spatial_merge_size
        grid = inputs["image_grid_thw"]  # (B * views, 3), sample-major
        sides = [(int(g[1]) // merge, int(g[2]) // merge) for g in grid[: len(order)]]
        img_pos = out["image_mask"].nonzero(as_tuple=True)[1].view(len(grid) // len(order), -1)

        def per_view(hs: Tensor) -> dict[str, Tensor]:
            toks = torch.gather(hs.float(), 1, img_pos[..., None].expand(-1, -1, hs.shape[-1]))
            res, s0 = {}, 0
            for cam, (gh, gw) in zip(order, sides, strict=True):
                v = toks[:, s0 : s0 + gh * gw]
                s0 += gh * gw
                fmap = v.transpose(1, 2).reshape(len(v), -1, gh, gw)  # (B, 2*2 cells, D) flattened cell-major
                res[cam] = (
                    torch.nn.functional.adaptive_avg_pool2d(fmap, 2).flatten(2).transpose(1, 2).flatten(1)
                )
            return res

        res: dict[str, dict[str, Tensor]] = {"grid2": per_view(out["backbone_features"]), "img": {}}
        for tap, ht in captured.items():
            res[f"grid2@{tap}"] = per_view(ht)
        for cam, v in res["grid2"].items():
            res["img"][cam] = v.view(len(v), 4, -1).mean(1)
        h, att = out["backbone_features"].float(), out["backbone_attention_mask"]
        res["lang"] = {VLA_CAMERA: _masked_mean(h, att & ~out["image_mask"])}
        res["all"] = {VLA_CAMERA: _masked_mean(h, att)}
        return res

    def _evo1(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """Evo-1: InternVL3 (truncated to 14 layers) over [one image per camera, prompt]; its action head reads every
        token of the last layer's output, which ``grid2`` pools per view (each image: a square grid of tokens)."""
        pol = self.policy
        emb = pol.model.embedder
        images, masks = pol._collect_image_batches(b)  # noqa: SLF001
        prompts = pol._normalize_task_batch(b)  # noqa: SLF001
        captured: dict[str, Tensor] = {}
        hooks = self._hook_layers(emb.model.language_model.layers, captured)
        ids: list[Tensor] = []
        hooks.append(
            emb.model.register_forward_pre_hook(
                lambda _m, _a, kwargs: ids.append(kwargs["input_ids"]), with_kwargs=True
            )
        )
        try:
            with torch.autocast(device_type=self._device().type, dtype=torch.bfloat16):
                h, valid = emb.get_fused_image_text_embedding_batched(
                    images, masks, prompts, return_cls_only=False
                )
        finally:
            for hk in hooks:
                hk.remove()
        n = emb.num_image_token
        side = int(round(n**0.5))
        img_pos = (ids[0] == emb.img_context_token_id).nonzero(as_tuple=True)[1].view(len(ids[0]), -1)
        views = list(pol._camera_keys or sorted(k for k in b if k.startswith(f"{OBS_IMAGES}.")))  # noqa: SLF001

        def per_view(hs: Tensor) -> dict[str, Tensor]:
            toks = torch.gather(hs.float(), 1, img_pos[..., None].expand(-1, -1, hs.shape[-1]))
            res = {}
            for v, model_cam in enumerate(views):
                cam = (
                    self.cameras[self.model_cameras.index(model_cam)]
                    if model_cam in self.model_cameras
                    else model_cam
                )
                if cam in self.present:
                    fmap = toks[:, v * n : (v + 1) * n].transpose(1, 2).reshape(len(toks), -1, side, side)
                    res[cam] = (
                        torch.nn.functional.adaptive_avg_pool2d(fmap, 2).flatten(2).transpose(1, 2).flatten(1)
                    )
            return res

        res: dict[str, dict[str, Tensor]] = {"grid2": per_view(h), "img": {}}
        for tap, ht in captured.items():
            res[f"grid2@{tap}"] = per_view(ht)
        for cam, v in res["grid2"].items():
            res["img"][cam] = v.view(len(v), 4, -1).mean(1)
        text = valid & (ids[0] != emb.img_context_token_id)
        res["lang"] = {VLA_CAMERA: _masked_mean(h.float(), text)}
        res["all"] = {VLA_CAMERA: _masked_mean(h.float(), valid)}
        return res

    def _molmoact2(self, b: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
        """MolmoAct2: the Molmo2 decoder over [per camera: a low-res and a high-res grid of image patches, prompt]; its
        action expert attends to every layer's keys and values. ``grid2`` pools each camera's high-res patch grid
        (the low-res one if it has none) at the last layer's output."""
        pol = self.policy
        backbone = pol._backbone()  # noqa: SLF001
        inputs = pol._model_inputs(b)  # noqa: SLF001
        layers = backbone.transformer.blocks
        captured: dict[str, Tensor] = {}
        hooks = self._hook_layers(layers, captured)
        last: list[Tensor] = []
        hooks.append(
            layers[-1].register_forward_hook(
                lambda _m, _i, out: last.append(out[0] if isinstance(out, tuple) else out)
            )
        )
        try:
            with pol._autocast_context():  # noqa: SLF001
                backbone(**inputs, use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        ids = inputs["input_ids"]
        n_rows = len(ids)
        grids = inputs["image_grids"].reshape(n_rows, -1, 4)  # per camera: low-res h, w, high-res h, w
        patch_pos = (ids == backbone.config.image_patch_id).nonzero(as_tuple=True)[1].view(n_rows, -1)
        views = list(pol.config.image_features)  # camera order of the prompt

        def per_view(hs: Tensor) -> dict[str, Tensor]:
            toks = torch.gather(hs.float(), 1, patch_pos[..., None].expand(-1, -1, hs.shape[-1]))
            res, s0 = {}, 0
            for v, model_cam in enumerate(views):
                rh, rw, hh, hw = (int(x) for x in grids[0, v])
                low, high = rh * rw, hh * hw
                gh, gw, start = (hh, hw, s0 + low) if high else (rh, rw, s0)
                s0 += low + high
                cam = self.cameras[self.model_cameras.index(model_cam)]
                if cam in self.present:
                    fmap = toks[:, start : start + gh * gw].transpose(1, 2).reshape(n_rows, -1, gh, gw)
                    res[cam] = (
                        torch.nn.functional.adaptive_avg_pool2d(fmap, 2).flatten(2).transpose(1, 2).flatten(1)
                    )
            return res

        res: dict[str, dict[str, Tensor]] = {"grid2": per_view(last[0]), "img": {}}
        for tap, ht in captured.items():
            res[f"grid2@{tap}"] = per_view(ht)
        for cam, v in res["grid2"].items():
            res["img"][cam] = v.view(len(v), 4, -1).mean(1)
        mask = (
            inputs["attention_mask"].bool()
            if "attention_mask" in inputs
            else torch.ones_like(ids, dtype=torch.bool)
        )
        res["lang"] = {
            VLA_CAMERA: _masked_mean(last[0].float(), mask & (ids != backbone.config.image_patch_id))
        }
        res["all"] = {VLA_CAMERA: _masked_mean(last[0].float(), mask)}
        return res


def _square_tokens(tokens: Tensor) -> Tensor:
    """Drop a leading summary token if that leaves a square patch grid."""
    p = tokens.shape[1]
    if int(round(p**0.5)) ** 2 != p and int(round((p - 1) ** 0.5)) ** 2 == p - 1:
        return tokens[:, 1:]
    return tokens


# ---------------------------------------------------------------------------------------------- rollouts
@dataclass
class _Slot:
    queue: deque
    t: int = 0
    episode: int = -1


class VLARollout:
    """Wraps a LeRobot VLA for ``LiberoPool``: per-environment action queues, optional chunk recording.

    The pool hands over dataset-format observations; this class applies the VLA's own pre/post processors
    (so pass identity processors to ``LiberoPool.evaluate``) and keeps *unnormalised* actions queued.
    With ``record=True`` every re-planning step stores the observation (uint8 images, state, prompt, step)
    and the full predicted chunk for offline analysis.
    """

    def __init__(self, policy, pre, post, n_action_steps: int, record_init_states: set[int] | None = None,
                 action_transform=None, camera_rename: dict[str, str] | None = None):  # fmt: skip
        """``action_transform``: (B, A) -> (B, 7) applied to every predicted action before it is queued (e.g. X-VLA's
        6D-rotation -> axis-angle LIBERO conversion, which the pool's queue-pop path would otherwise skip)."""
        self.policy, self.pre, self.post = policy, pre, post
        self.action_transform = action_transform
        self.camera_rename = camera_rename or {}  # dataset camera key -> the key the policy expects
        self.failed_predictions = 0
        self.n_action_steps = n_action_steps
        self.record_init_states = record_init_states
        self.records: list[dict[str, Any]] = []  # finished episodes of the recorded initial states
        self._open: dict[int, list[dict[str, Any]]] = {}
        self._slots: dict[int, _Slot] = {}
        self._next_episode = 0

    def reset(self) -> None:
        self._slots.clear()
        self.policy.reset()

    def _slot(self, env_id: int) -> _Slot:
        if env_id not in self._slots:
            self._slots[env_id] = _Slot(deque())
        return self._slots[env_id]

    def reset_envs(self, env_ids: list[int]) -> None:
        for w in env_ids:
            s = self._slot(w)
            s.queue.clear()
            s.t = 0
            s.episode = self._next_episode
            self._next_episode += 1

    def episode_done(self, env_id: int, result: Any) -> None:
        recs = self._open.pop(self._slot(env_id).episode, [])
        if self.record_init_states is not None and result.init_state in self.record_init_states:
            for r in recs:
                r.update(suite=result.suite, task_id=result.task_id, init_state=result.init_state,
                         success=result.success, episode_steps=result.steps)  # fmt: skip
            self.records.extend(recs)

    def needs_observation(self, env_ids: list[int]) -> list[bool]:
        return [not self._slot(w).queue for w in env_ids]

    def _predict(self, sub: dict[str, Any]) -> Tensor:
        """Predict a chunk per row; a row whose prediction fails (e.g. an invalid FAST token sequence) holds still."""
        if self.camera_rename:
            sub = {self.camera_rename.get(k, k): v for k, v in sub.items()}
        try:
            self.policy.reset()  # clear any batch-level queues; chunks are managed here
            return self.post(self.policy.predict_action_chunk(self.pre(dict(sub)))).float().cpu()
        except Exception:  # noqa: BLE001 - retry row by row, then fall back to a zero (hold-still) chunk
            n = len(sub["task"])
            rows, good = [], None
            for i in range(n):
                one = {k: (v[i : i + 1] if torch.is_tensor(v) else [v[i]]) for k, v in sub.items()}
                try:
                    self.policy.reset()
                    out = self.post(self.policy.predict_action_chunk(self.pre(one))).float().cpu()
                    good = out
                except Exception:  # noqa: BLE001
                    self.failed_predictions += 1
                    out = None
                rows.append(out)
            if good is None:
                raise
            return torch.cat([r if r is not None else torch.zeros_like(good) for r in rows])

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], env_ids: list[int]) -> Tensor:
        slots = [self._slot(w) for w in env_ids]
        need = [i for i, s in enumerate(slots) if not s.queue]
        if need:
            sub = {
                k: (v[need] if torch.is_tensor(v) else [v[i] for i in need])
                for k, v in batch.items()
                if torch.is_tensor(v) or k == "task"
            }
            chunk = self._predict(sub)  # (B, C, A) unnormalised
            if self.action_transform is not None:
                b_, c_ = chunk.shape[:2]
                chunk = self.action_transform(chunk.reshape(b_ * c_, -1)).reshape(b_, c_, -1).float().cpu()
            for row, i in enumerate(need):
                slots[i].queue.extend(chunk[row, : self.n_action_steps])
                if self.record_init_states is not None:
                    self._open.setdefault(slots[i].episode, []).append(
                        {
                            "episode": slots[i].episode,
                            "t": slots[i].t,
                            "task": sub["task"][row],
                            "state": sub[OBS_STATE][row].float().cpu().numpy(),
                            "images": np.stack(
                                [
                                    (sub[c][row].clamp(0, 1) * 255).round().byte().cpu().numpy()
                                    for c in DATASET_CAMERAS
                                ]
                            ),
                            "chunk": chunk[row].numpy(),
                        }
                    )
        out = []
        for s in slots:
            out.append(s.queue.popleft())
            s.t += 1
        return torch.stack(out)


class Identity:
    """No-op processor with the ``reset`` hook ``LiberoPool.evaluate`` expects."""

    def __call__(self, x: Any) -> Any:
        return x

    def reset(self) -> None:
        pass
