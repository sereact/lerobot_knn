"""Fine-tuning dose-response: is the VLM a better lookup key the longer the VLA is fine-tuned?

Two sub-commands:

    # while lerobot-train runs: keep only the checkpoints we evaluate, shrink them to bf16, drop old optimizer state
    uv run python -m lerobot_policy_knn.experiments.dose_response prune outputs/vla/smolvla_own --keep=1000,3000,8000,25000

    # after the evaluations: assemble outputs/vla/dose_response.json for the poster
    uv run python -m lerobot_policy_knn.experiments.dose_response assemble outputs/vla/smolvla_own

A checkpoint's evaluations are found by tag: ``embed_quality.jsonl`` (tag ``dose``, held-out demo R²),
``outputs/knn/compare.jsonl`` (tag ``dose_knn``, closed-loop kNN on the checkpoint's VLM) and
``outputs/vla/results.jsonl`` (tag ``dose_vla``, the VLA itself). Step 0 is ``smolvla_base``.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

VLA = Path("outputs/vla")


def checkpoints(run: Path) -> dict[int, Path]:
    root = run / "checkpoints"
    return (
        {int(p.name): p for p in root.iterdir() if p.is_dir() and p.name.isdigit()} if root.exists() else {}
    )


def shrink(model_file: Path) -> None:
    tensors = load_file(str(model_file))
    if all(t.dtype != torch.float32 for t in tensors.values()):
        return
    tensors = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in tensors.items()}
    tmp = model_file.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    tmp.rename(model_file)


def prune(run: Path, keep: set[int], until_step: int, poll_s: float = 15.0) -> None:
    """Prune every checkpoint as soon as lerobot-train marks it complete (``checkpoints/last`` points at it).

    No run is ever resumed, so optimizer state is dropped immediately; kept steps are shrunk to bf16 and every
    other step's weights are deleted. This bounds disk use to one checkpoint being written plus the kept models.
    """
    done: set[int] = set()
    link = run / "checkpoints" / "last"
    while True:
        complete = int(link.resolve().name) if link.exists() else -1
        for step, path in sorted(checkpoints(run).items()):
            if step in done or step > complete:
                continue
            if (path / "training_state").exists():
                shutil.rmtree(path / "training_state")
            if step in keep:
                model = path / "pretrained_model" / "model.safetensors"
                if model.exists():
                    shrink(model)
            elif (path / "pretrained_model").exists():
                shutil.rmtree(path / "pretrained_model")
            done.add(step)
        if complete >= until_step:
            return
        time.sleep(poll_s)


def assemble(run: Path, accum: int = 1) -> None:
    """``accum``: gradient-accumulation factor (checkpoint names count micro-batches; the plot uses updates)."""

    def rows(path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

    def step_of(name: str) -> int | None:
        if name in ("smolvla_base", "vla:smolvla_base", "vla:smolvla_base/img"):
            return 0
        for part in Path(name.split("@", 1)[-1]).parts:
            if part.isdigit():
                return int(part)
        return None

    table: dict[int, dict] = {}
    for r in rows(VLA / "embed_quality.jsonl"):
        if (
            r.get("tag") == "dose"
            and r.get("k") == 8
            and (s := step_of(r["space"].rsplit("/", 1)[0])) is not None
        ):
            table.setdefault(s, {})["demo_r2"] = r["r2"]
    for r in rows(Path("outputs/knn/compare.jsonl")):
        if r.get("tag") == "dose_knn" and (s := step_of(r["config"]["image_encoder"])) is not None:
            table.setdefault(s, {})["knn_success"] = 100 * r["success"]
    for r in rows(VLA / "results.jsonl"):
        if r.get("tag") == "dose_vla" and (s := step_of(r["name"])) is not None:
            table.setdefault(s, {})["vla_success"] = 100 * r["success"]
    steps = sorted(s for s, v in table.items() if {"demo_r2", "knn_success", "vla_success"} <= set(v))
    out = {
        "run": str(run),
        "steps": [s // accum for s in steps],
        "demo_r2": [table[s]["demo_r2"] for s in steps],
        "knn_success": [table[s]["knn_success"] for s in steps],
        "vla_success": [table[s]["vla_success"] for s in steps],
    }
    (VLA / "dose_response.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prune", "assemble"])
    parser.add_argument("run")
    parser.add_argument("--keep", default="1000,3000,8000,25000")
    parser.add_argument("--until_step", type=int, default=25000)
    parser.add_argument("--accum", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prune":
        prune(Path(args.run), {int(x) for x in args.keep.split(",")}, args.until_step)
    else:
        assemble(Path(args.run), args.accum)


if __name__ == "__main__":
    main()
