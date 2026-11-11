"""Shared code of the paper's figure notebooks.

Every function takes the results it reads: data (results/data.json) or maps (results/embedding_maps.json). Figures are
vector PDFs drawn at print size: the paper's 5.5 in text width, in its fonts (figure.mplstyle). Colour roles are fixed:
kNN (ours) = orange, learned models = blue, generic encoders = grey, and a VLM's LIBERO training (before / after
fine-tuning) = a light / dark blue.
"""

from enum import StrEnum

import numpy as np


class Colour(StrEnum):
    """The palette of every figure (a str: usable as any matplotlib colour)."""

    GRID = "#e1e0d9"
    AXIS = "#c3c2b7"
    KNN = "#eb6834"  # kNN (ours)
    LEARNED = "#2a78d6"  # learned models
    HUMAN = "#9a988f"
    HUMAN_DARK = "#5f5e59"
    BEFORE = "#86b6ef"  # a VLM before / after its LIBERO fine-tuning (validated ordinal pair)
    AFTER = "#1c5cab"


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


# ---- the policies compared, and their results ---------------------------------------------------------------
POLICIES = [  # label, role, LIBERO id in libero_full (official protocol, 2,000 episodes), LIBERO-plus spec
    ("k-NN (multimodal)", "knn", "knn:00_base", "knn:outputs/knn/v2_39.json"),
    ("k-NN (LAC key)", "knn", "knn:full_resnet18_l3_learned", "knn:outputs/vla/configs/full_resnet18_l3_learned.json"),
    ("k-NN (ResNet)", "knn2", "knn:plain_resnet18_l3", "knn:outputs/vla/configs/plain_resnet18_l3.json"),
    ("π0.5", "vla", "vla:pi05_libero", "vla:pi05_libero"),
    ("MolmoAct2", "vla", "vla:molmoact2_libero", "vla:molmoact2_libero"),
    ("VLA-JEPA", "vla", "vla:vla_jepa@lerobot/VLA-JEPA-LIBERO", "vla:vla_jepa@lerobot/VLA-JEPA-LIBERO"),
    ("Evo-1", "vla", "vla:evo1_libero", "vla:evo1_libero"),
    ("GR00T N1.7", "vla", "vla:groot_libero", "vla:groot_libero"),
    ("X-VLA", "vla", "vla:xvla@lerobot/xvla-libero", "vla:xvla@lerobot/xvla-libero"),
    ("π0-FAST", "vla", "vla:pi0_fast@lerobot/pi0fast-libero", "vla:pi0_fast@lerobot/pi0fast-libero"),
    ("SmolVLA", "vla", "vla:smolvla_libero", "vla:smolvla_libero"),
    ("π0", "vla", "vla:pi0_libero", "vla:pi0_libero"),
]  # fmt: skip


def libero_success(data: dict, run: str) -> list[float]:
    """[average, spatial, object, goal, long] in % on the official protocol (2,000 episodes)."""
    r = data["libero_full"][run]
    return [r["success"], *r["suites"]]


def plus_success(data: dict, spec: str) -> list[float]:
    """[average, spatial, object, goal, long] in % on the 420 LIBERO-plus variants."""
    r = data["libero_plus"][spec]
    return [100 * r["all"], *(100 * r[f"suite/{s}"] for s in SUITES)]


def ranked_policies(data: dict) -> list[tuple]:
    """POLICIES ordered by LIBERO-plus success, then LIBERO (best first): the order of every table and map."""
    return sorted(POLICIES, key=lambda p: (-plus_success(data, p[3])[0], -libero_success(data, p[2])[0]))


# ---- how hard is the task? -----------------------------------------------------------------------------------
def per_task(eps: dict[str, bool], task_of=None) -> dict[str, float]:
    """Success per task from per-episode results ("suite/task_id/init_state" keys)."""
    by: dict[str, list[bool]] = {}
    for k, v in eps.items():
        t = k.rsplit("/", 1)[0]
        by.setdefault(task_of.get(t, t) if task_of else t, []).append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def episodes(data: dict, bench: str, run: str, spec: str) -> dict[str, bool]:
    """Per-episode success of a policy on LIBERO (official protocol) or LIBERO-plus (the 420 variants)."""
    return data["libero_full"][run]["episodes"] if bench == "libero" else data["plus_episodes"][spec]


def task_matrix(data: dict, bench: str) -> tuple[list[str], np.ndarray]:
    """Rows: the policies (ranked); columns: the 40 tasks, easiest first."""
    task_of = data["plus_variant_task"] if bench == "libero_plus" else None
    rows = [lab for lab, *_ in ranked_policies(data)]
    mats = [per_task(episodes(data, bench, run, spec), task_of) for _lab, _role, run, spec in ranked_policies(data)]
    tasks = sorted(set().union(*mats))
    m = np.array([[mt[t] for t in tasks] for mt in mats])
    return rows, m[:, np.argsort(-m.mean(axis=0), kind="stable")]


PLUS_CATEGORIES = [("Camera Viewpoints", "Camera"), ("Robot Initial States", "Robot"), ("Language Instructions", "Language"),
                   ("Light Conditions", "Light"), ("Background Textures", "Background"), ("Sensor Noise", "Noise"),
                   ("Objects Layout", "Layout")]  # fmt: skip


LOOKUP_VLAS = [  # label, head_vs_knn "vla" name of its clean LIBERO recording (X-VLA: absolute poses, not comparable)
    ("π0.5", "pi05_libero"), ("VLA-JEPA", "vla_jepa@lerobot/VLA-JEPA-LIBERO"), ("Evo-1", "evo1_libero"),
    ("MolmoAct2", "molmoact2_libero"), ("GR00T N1.7†", "groot_libero_spatial"), ("π0-FAST", "pi0_fast@lerobot/pi0fast-libero"),
    ("SmolVLA", "smolvla_libero"), ("π0", "pi0_libero"),
]  # fmt: skip


def lookup_rows(data: dict) -> list[tuple]:
    """Per VLA (clean LIBERO rollouts, ResNet-18 key, k = 8): R² of the lookup's prediction of the VLA's chunks and of
    held-out human chunks, and the VLA's distance to the nearest training chunk relative to a human demo's. Ordered as
    ranked_policies, like every table and map."""
    rank = [lab for lab, *_ in ranked_policies(data)]
    out = []
    for label, name in sorted(LOOKUP_VLAS, key=lambda v: rank.index(v[0].rstrip("†"))):
        h = data["head_vs_knn"][name]
        sp, nn = h["spaces"]["resnet18_l3"]["8"], h["nearest_training_chunk"]
        out.append((label, sp["vla_r2"], sp["demo_r2"], nn["vla_median"] / nn["demo_median"]))
    return out


BACKBONES = [  # label, the VLA ("vla:<name>" in libero_full and libero_plus), key before fine-tuning, after
    ("π0.5", "pi05_libero", "pi05_base", "pi05_libero"),
    ("π0-FAST", "pi0_fast@lerobot/pi0fast-libero", "pi0fast_base", "pi0fast_libero"),
    ("X-VLA", "xvla@lerobot/xvla-libero", "xvla_base", "xvla_libero"),
    ("VLA-JEPA", "vla_jepa@lerobot/VLA-JEPA-LIBERO", "vla_jepa_pretrain", "vla_jepa_libero"),
    ("MolmoAct2", "molmoact2_libero", "molmoact2_base", "molmoact2_libero"),
    ("Evo-1", "evo1_libero", "evo1_base", "evo1_libero"),
    ("GR00T N1.7*", "groot_libero", "groot_base", "groot_base"),
    ("SmolVLA", "smolvla_libero", "smolvla_base", "smolvla_libero"),
    ("π0", "pi0_libero", "pi0_base", "pi0_libero"),
]


def backbone_rows(data: dict) -> list[tuple]:
    """Per VLA: (label, [clean, plus] of the lookup on its VLM before / after fine-tuning and of the VLA itself)."""

    def pair(run: str, spec: str) -> tuple[float, float]:
        return libero_success(data, run)[0], plus_success(data, spec)[0]

    fixed = {"pi05_base": "plain_pi05_base_grid2@L12_z", "pi05_libero": "plain_pi05_libero_grid2@L12_z",
             "pi0_base": "plain_pi0_base_grid2@L12_z", "pi0_libero": "plain_pi0_libero_grid2@L12_z",
             "smolvla_base": "plain_smolvla_base_grid2", "smolvla_libero": "plain_smolvla_libero_grid2"}  # fmt: skip

    def key(name: str) -> tuple[float, float]:
        """The lookup on a backbone key: config stem chosen by choose_keys, else the study's recipe (``fixed``)."""
        stem = fixed.get(name) or next(k["stem"] for k in data["extra_keys"] if k["name"] == name)
        return pair(f"knn:{stem}", f"knn:outputs/vla/configs/{stem}.json")

    out = [
        (label, key(before), key(after), pair(f"vla:{vla}", f"vla:{vla}"))
        for label, vla, before, after in BACKBONES
    ]
    return sorted(out, key=lambda r: (-r[3][1], -r[3][0]))  # as the results table




def suite_r2(maps: dict, space: str) -> float:
    """Share of the variance of a key's 2-D PCA map that the suite (the scene) explains."""
    xy, suite = np.array(maps["spaces"][space]["pca"]), np.array(maps["suite"])
    within = sum(((xy[suite == s] - xy[suite == s].mean(0)) ** 2).sum() for s in set(maps["suite"]))
    return float(1 - within / ((xy - xy.mean(0)) ** 2).sum())
