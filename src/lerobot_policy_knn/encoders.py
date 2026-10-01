"""Frozen, pretrained encoders that embed camera images and task prompts.

Every image encoder maps a float image batch ``(B, 3, H, W)`` in ``[0, 1]`` to a dict of *raw pooled*
features (``cls``, ``mean``, ``grid2``, ``grid4``, ...). Raw features are what the feature cache stores;
:func:`finalize_features` turns them into the unit-norm retrieval keys used by the policy. Running the
same two functions on dataset frames and on live observations keeps memory and queries consistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

logger = logging.getLogger(__name__)

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
_SIGLIP = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass(frozen=True)
class ImageEncoderSpec:
    kind: str  # "hf" | "torchvision" | "pixels"
    name: str | None
    norm: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    # Pool sizes worth caching for this encoder (grids are memory hungry for wide models).
    poolings: tuple[str, ...] = ("cls", "mean", "grid2")
    # torchvision ResNets: last residual stage kept (4 = full trunk, 3 = stop after layer3).
    layer: int = 4


IMAGE_ENCODERS: dict[str, ImageEncoderSpec] = {
    "dinov2_small": ImageEncoderSpec(
        "hf", "facebook/dinov2-small", _IMAGENET, ("cls", "mean", "grid2", "grid4")
    ),
    "dinov2_base": ImageEncoderSpec("hf", "facebook/dinov2-base", _IMAGENET),
    "dinov2reg_small": ImageEncoderSpec("hf", "facebook/dinov2-with-registers-small", _IMAGENET),
    "dinov3_small": ImageEncoderSpec(
        "hf", "facebook/dinov3-vits16-pretrain-lvd1689m", _IMAGENET, ("cls", "mean", "grid2", "grid4")
    ),
    "dinov3_base": ImageEncoderSpec("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m", _IMAGENET),
    "siglip2_base": ImageEncoderSpec("hf", "google/siglip2-base-patch16-224", _SIGLIP),
    "clip_b16": ImageEncoderSpec("hf", "openai/clip-vit-base-patch16", _CLIP),
    "resnet18": ImageEncoderSpec("torchvision", "resnet18", _IMAGENET, ("cls", "grid2", "grid4")),
    "resnet18_l3": ImageEncoderSpec(
        "torchvision", "resnet18", _IMAGENET, ("grid2", "grid4", "mean"), layer=3
    ),
    "pixels": ImageEncoderSpec("pixels", None, None, ("p16", "p32")),
}

TEXT_ENCODERS: dict[str, str] = {
    "clip_b16": "openai/clip-vit-base-patch16",
}


def _grid_pool(patches: Tensor, grid: int) -> Tensor:
    """(B, P, D) square patch tokens -> (B, grid*grid, D) average-pooled cells."""
    b, p, d = patches.shape
    side = int(round(p**0.5))
    if side * side != p:
        raise ValueError(f"Expected a square patch grid, got {p} tokens")
    fmap = patches.transpose(1, 2).reshape(b, d, side, side)
    return F.adaptive_avg_pool2d(fmap, grid).flatten(2).transpose(1, 2)


class ImageEncoder(nn.Module):
    """Frozen image backbone returning several raw poolings of its last feature map."""

    def __init__(self, name: str, resolution: int = 224):
        super().__init__()
        if name not in IMAGE_ENCODERS:
            raise ValueError(f"Unknown image encoder '{name}'. Available: {sorted(IMAGE_ENCODERS)}")
        self.name = name
        self.spec = IMAGE_ENCODERS[name]
        self.resolution = resolution
        self.n_prefix_tokens = 0
        self.backbone: nn.Module | None = None
        if self.spec.kind == "hf":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(self.spec.name)
            config = model.config
            model = getattr(model, "vision_model", model)
            model_type = getattr(config, "model_type", "")
            if model_type == "siglip":
                self.n_prefix_tokens = 0  # no CLS token; pooler_output is the MAP head
            else:
                self.n_prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0) or 0)
            self.backbone = model
        elif self.spec.kind == "torchvision":
            import torchvision

            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            resnet = torchvision.models.resnet18(weights=weights)
            # children: conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc
            self.backbone = nn.Sequential(*list(resnet.children())[: 4 + self.spec.layer])
        if self.spec.norm is not None:
            mean, std = self.spec.norm
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.eval().requires_grad_(False)

    @property
    def poolings(self) -> tuple[str, ...]:
        return self.spec.poolings

    @torch.no_grad()
    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """images: (B, 3, H, W) float in [0, 1]. Returns raw pooled features in float32."""
        images = images.float()
        if self.spec.kind == "pixels":
            out = {}
            for pooling in self.spec.poolings:
                side = int(pooling[1:])
                small = F.interpolate(images, size=(side, side), mode="area")
                out[pooling] = (small - 0.5).flatten(1)
            return out

        if images.shape[-1] != self.resolution or images.shape[-2] != self.resolution:
            images = F.interpolate(
                images, size=(self.resolution, self.resolution), mode="bilinear", antialias=True
            )
        x = (images - self.mean) / self.std
        assert self.backbone is not None
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            if self.spec.kind == "torchvision":
                fmap = self.backbone(x)  # (B, C, h, w)
                patches = fmap.flatten(2).transpose(1, 2)
                cls = patches.mean(1)
            else:
                out = self.backbone(pixel_values=x)
                hidden = out.last_hidden_state
                patches = hidden[:, self.n_prefix_tokens :]
                pooled = getattr(out, "pooler_output", None)
                cls = pooled if pooled is not None else hidden[:, 0]
        patches = patches.float()
        feats: dict[str, Tensor] = {}
        for pooling in self.spec.poolings:
            if pooling == "cls":
                feats["cls"] = cls.float()
            elif pooling == "mean":
                feats["mean"] = patches.mean(1)
            elif pooling.startswith("grid"):
                feats[pooling] = _grid_pool(patches, int(pooling[4:])).flatten(1)
            else:
                raise ValueError(f"Unknown pooling '{pooling}'")
        return feats


def finalize_features(raw: Tensor, pooling: str) -> Tensor:
    """Turn raw pooled features into unit-norm retrieval keys.

    Grid poolings normalise every cell independently (so each spatial cell contributes equally)
    before the whole vector is renormalised; the squared L2 distance between keys is then
    ``2 - 2 * mean_cell_cosine``.
    """
    raw = raw.float()
    if pooling.startswith("grid"):
        cells = int(pooling.split("@")[0][4:]) ** 2
        if "@" in pooling:  # "grid2@L12+L15+last": one block of cells per layer
            cells *= len(pooling.split("@", 1)[1].split("+"))
        b = raw.shape[0]
        per_cell = raw.view(b, cells, -1)
        per_cell = F.normalize(per_cell, dim=-1)
        return per_cell.flatten(1) / cells**0.5
    return F.normalize(raw, dim=-1)


class TextEncoder(nn.Module):
    """CLIP text tower used to embed task prompts (L2-normalised)."""

    def __init__(self, name: str = "clip_b16"):
        super().__init__()
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        repo = TEXT_ENCODERS.get(name, name)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = CLIPTextModelWithProjection.from_pretrained(repo).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        return F.normalize(self.model(**tokens).text_embeds.float(), dim=-1)
