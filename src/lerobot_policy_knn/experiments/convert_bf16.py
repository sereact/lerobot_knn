"""Stream a float32 safetensors checkpoint from the Hub and write a bfloat16 copy, tensor by tensor.

    uv run python -m lerobot_policy_knn.experiments.convert_bf16 lerobot/pi05_base outputs/vla/pi05_base_bf16

``lerobot/pi05_base`` is published as a 14.5 GB float32 file. HTTP range requests fetch one tensor at a time,
so only the 7.2 GB bfloat16 result ever touches the disk (plus the small config and processor files).
"""

import argparse
import json
import struct
from pathlib import Path

import torch
from huggingface_hub import HfApi, get_session, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

CHUNK = 256 * 1024 * 1024


def _get_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of ``url``."""
    headers = {**build_hf_headers(), "Range": f"bytes={start}-{end - 1}"}
    r = get_session().get(url, headers=headers, timeout=600)
    r.raise_for_status()
    data = r.content
    if len(data) != end - start:
        raise OSError(f"Short read: {len(data)} of {end - start} bytes")
    return data


def convert(repo_id: str, out_dir: Path, filename: str = "model.safetensors") -> None:
    if (out_dir / filename).exists() and (out_dir / "config.json").exists():
        print(f"{out_dir} already converted")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id, filename)
    (n,) = struct.unpack("<Q", _get_range(url, 0, 8))
    header = json.loads(_get_range(url, 8, 8 + n))
    meta = header.pop("__metadata__", None)
    base = 8 + n
    names = sorted(header, key=lambda k: header[k]["data_offsets"][0])
    new_header, offset = {}, 0
    for name in names:
        info = header[name]
        dtype = "BF16" if info["dtype"] == "F32" else info["dtype"]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        size = size // 2 if info["dtype"] == "F32" else size
        new_header[name] = {"dtype": dtype, "shape": info["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    if meta is not None:
        new_header["__metadata__"] = meta
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = out_dir / (filename + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for i, name in enumerate(names):
            info = header[name]
            s, e = info["data_offsets"]
            parts = [_get_range(url, base + a, base + min(a + CHUNK, e)) for a in range(s, e, CHUNK)]
            data = b"".join(parts) if parts else b""
            if info["dtype"] == "F32":
                t = torch.frombuffer(bytearray(data), dtype=torch.float32).to(torch.bfloat16)
                data = t.view(torch.int16).numpy().tobytes()
            f.write(data)
            if i % 100 == 0:
                print(f"{i}/{len(names)} tensors", flush=True)
    tmp.rename(out_dir / filename)
    # small files: config (switched to bfloat16) and processors
    for sib in HfApi().model_info(repo_id).siblings:
        small = (sib.size or 0) < 50_000_000
        if sib.rfilename != filename and not sib.rfilename.endswith(".safetensors.index.json") and small:
            src = hf_hub_download(repo_id, sib.rfilename)
            (out_dir / sib.rfilename).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / sib.rfilename).write_bytes(Path(src).read_bytes())
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "dtype" in cfg:
            cfg["dtype"] = "bfloat16"
        cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"wrote {out_dir} ({offset / 1e9:.2f} GB of tensor data)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    convert(args.repo_id, Path(args.out_dir))


if __name__ == "__main__":
    main()
