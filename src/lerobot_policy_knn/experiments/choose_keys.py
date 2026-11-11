"""Pick each new backbone's lookup key from its layer pilot and write the plain-lookup configs.

For every family (its checkpoint before and after LIBERO fine-tuning) the tap with the best mean held-out action R²
over the two, z-scored, in ``outputs/vla/embed_pilot.json`` (``embed_pilot``) becomes the key, so before and after use
the same layer. A family whose LIBERO fine-tune froze the VLM has one checkpoint, which is both. Writes
``outputs/vla/configs/plain_<name>_<pooling>_z.json`` and ``outputs/vla/extra_keys.json``, which ``neighbor_metric``,
``plus_neighbor_metric`` and ``plus_language`` pick up.

    uv run python -m lerobot_policy_knn.experiments.choose_keys
"""

import json

from .common import VLA_OUT

FAMILIES = {  # family label -> (before, after) registry names, or (frozen,)
    "X-VLA": ("xvla_base", "xvla_libero"),
    "π0-FAST": ("pi0fast_base", "pi0fast_libero"),
    "VLA-JEPA": ("vla_jepa_pretrain", "vla_jepa_libero"),
    # tune_llm = tune_visual = False: every LIBERO checkpoint keeps the base's VLM
    "GR00T N1.7": ("groot_base",),
    # before: the raw InternVL3-1B (no Evo-1 checkpoint before LIBERO)
    "Evo-1": ("evo1_base", "evo1_libero"),
    "MolmoAct2": ("molmoact2_base", "molmoact2_libero"),
}
CONFIGS = VLA_OUT / "configs"
TEMPLATE = CONFIGS / "plain_pi05_base_grid2@L12_z.json"  # the plain lookup's settings
STAGE_LABELS = {"before": "base", "after": "LIBERO", "frozen": "frozen VLM"}


def main() -> None:
    pilot = json.loads((VLA_OUT / "embed_pilot.json").read_text())
    template = json.loads(TEMPLATE.read_text())
    keys = []
    for family, names in FAMILIES.items():
        runs = [pilot.get(f"vla_{n}_pilot", {}) for n in names]
        if not all(runs):
            print(f"skip {family}: pilot missing")
            continue
        taps = {k.split("|")[0] for k in runs[0] if k.endswith("|zscore")}
        score = {
            t: sum(r[f"{t}|zscore"]["r2"] for r in runs) / len(runs)
            for t in taps
            if all(f"{t}|zscore" in r for r in runs)
        }
        best = max(score, key=score.get)
        print(family, "best tap", best, {t: round(v, 3) for t, v in sorted(score.items())})
        stages = ("before", "after") if len(names) == 2 else ("frozen",)
        for name, stage in zip(names, stages, strict=True):
            stem = f"plain_{name}_{best}_z"
            config = {**template, "image_encoder": f"vla:{name}", "image_pooling": best, "key_norm": "zscore"}
            (CONFIGS / f"{stem}.json").write_text(json.dumps(config, indent=1))
            keys.append(
                {
                    "name": name,
                    "label": f"{family} {STAGE_LABELS[stage]} {best}",
                    "encoder": f"vla:{name}",
                    "pooling": best,
                    "norm": "zscore",
                    "stem": stem,
                    "tap": best.split("@")[1] if "@" in best else "last",
                    # the key never saw these demos (GR00T N1.7's base pretraining: DROID and real robots, no LIBERO)
                    "fair": stage in ("before", "frozen"),
                }
            )
    (VLA_OUT / "extra_keys.json").write_text(json.dumps({"keys": keys}, indent=1))
    print("->", [k["stem"] for k in keys])


if __name__ == "__main__":
    main()
