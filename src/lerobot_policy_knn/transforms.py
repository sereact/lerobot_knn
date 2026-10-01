"""Query-side image transforms that make live camera frames look like the dataset's frames.

Dataset frames went through lossy video compression (AV1, yuv420p) before being embedded into the memory;
live simulator frames did not. Applying a cheap imitation of that pipeline to the *query* moves it into
the memory's domain. Specs compose with ``+``, e.g. ``"yuv420+blur:0.5"``:

- ``none``
- ``blur:<sigma>``: Gaussian blur (pixels)
- ``jpeg:<quality>``: JPEG round trip
- ``yuv420``: RGB -> YCbCr, 2x chroma subsampling, back to RGB
- ``av1[:<crf>]``: a real SVT-AV1 intra-frame encode/decode (LeRobot's default codec settings; slow,
  ~35 ms per frame, parallelised over a thread pool)
"""

import io
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def _blur(x: Tensor, sigma: float) -> Tensor:
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _yuv420(x: Tensor) -> Tensor:
    r, g, b = x.unbind(1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564
    cr = (r - y) * 0.713
    chroma = torch.stack([cb, cr], 1)
    h, w = x.shape[-2:]
    chroma = F.interpolate(F.avg_pool2d(chroma, 2), size=(h, w), mode="bilinear", align_corners=False)
    cb, cr = chroma.unbind(1)
    r = y + 1.403 * cr
    b = y + 1.773 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.stack([r, g, b], 1).clamp(0.0, 1.0)


def _jpeg(x: Tensor, quality: int) -> Tensor:
    from torchvision.io import decode_jpeg, encode_jpeg

    u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    out = [decode_jpeg(encode_jpeg(img, quality=quality)) for img in u8]
    return torch.stack(out).to(x.device, x.dtype) / 255.0


_POOL: ThreadPoolExecutor | None = None


def _av1_frame(img: np.ndarray, crf: int) -> np.ndarray:
    import av

    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    stream = out.add_stream("libsvtav1", rate=10)
    stream.width, stream.height, stream.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
    stream.options = {"crf": str(crf), "preset": "12", "g": "2", "svtav1-params": "lp=1"}
    for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
        out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    buf.seek(0)
    with av.open(buf) as inp:
        return next(inp.decode(video=0)).to_ndarray(format="rgb24")


def _av1(x: Tensor, crf: int) -> Tensor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=16)
    u8 = (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    out = list(_POOL.map(lambda img: _av1_frame(np.ascontiguousarray(img), crf), u8))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(x.device, x.dtype) / 255.0


def apply_query_transform(images: Tensor, spec: str) -> Tensor:
    """images: (B, 3, H, W) float in [0, 1]."""
    if not spec or spec == "none":
        return images
    x = images.float()
    for part in spec.split("+"):
        name, _, arg = part.partition(":")
        if name == "blur":
            x = _blur(x, float(arg))
        elif name == "jpeg":
            x = _jpeg(x, int(arg))
        elif name == "yuv420":
            x = _yuv420(x)
        elif name == "av1":
            x = _av1(x, int(arg) if arg else 30)
        elif name != "none":
            raise ValueError(f"Unknown query transform '{part}'")
    return x
