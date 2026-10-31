"""Shared code of the paper's figure notebooks.

Everything comes from results/data.json and results/embedding_maps.json. Figures are vector PDFs drawn at print
size: the paper's 5.5 in text width. Colour roles are fixed: kNN (ours) = orange, learned models = blue, generic encoders
= grey, and a VLM's LIBERO training (before / after fine-tuning) = a light / dark blue.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STYLE = Path(__file__).resolve().parent / "figure.mplstyle"
DATA, MAPS = {}, {}  # filled by load_data(data_path), at the top of each notebook


def load_data(data_path: Path) -> None:
    DATA.update(json.loads((data_path / "data.json").read_text()))
    MAPS.update(json.loads((data_path / "embedding_maps.json").read_text()))


INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KNN, LEARNED, HUMAN, HUMAN_DARK = "#eb6834", "#2a78d6", "#9a988f", "#5f5e59"
BEFORE, AFTER = "#86b6ef", "#1c5cab"  # a VLM before / after its LIBERO fine-tuning (validated ordinal pair)
W = 5.5  # inches: the CoRL template's text width
MAP_POINT_SIZE = 1.2  # embedding map scatter
FS, FS_SMALL = 9.5, 8.5  # 10 pt body text
mpl.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "grid.color": GRID,
    "grid.linewidth": 0.7, "lines.linewidth": 2, "pdf.fonttype": 3, "savefig.transparent": True,
})  # fmt: skip
# every figure height is scaled by 0.72, to suit the paper's small fonts
_subplots = plt.subplots


def _paper_subplots(*args, figsize=None, **kwargs):
    if figsize is not None:
        figsize = (figsize[0], figsize[1] * 0.72)
    return _subplots(*args, figsize=figsize, **kwargs)


plt.subplots = _paper_subplots

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def f1(v: float) -> str:
    """One decimal, rounding halves up (90.55 -> 90.6, as the papers report), not to even."""
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def hbar_axes(ax) -> None:
    ax.tick_params(axis="y", length=0, labelsize=FS_SMALL + 0.5, labelcolor=INK2)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


# ---- the policies compared, and their results ---------------------------------------------------------------
POLICIES = [  # label, role, LIBERO id in libero_full (official protocol, 2,000 episodes), LIBERO-plus spec
    ("kNN, full method", "knn", "knn:00_base", "knn:outputs/knn/v2_39.json"),
    ("kNN, + learnt key", "knn", "knn:full_resnet18_l3_learned", "knn:outputs/vla/configs/full_resnet18_l3_learned.json"),
    ("kNN, plain", "knn2", "knn:plain_resnet18_l3", "knn:outputs/vla/configs/plain_resnet18_l3.json"),
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


def libero_success(run: str) -> list[float]:
    """[average, spatial, object, goal, long] in % on the official protocol (2,000 episodes)."""
    r = DATA["libero_full"][run]
    return [r["success"], *r["suites"]]


def libero_episodes(run: str) -> dict[str, bool]:
    return DATA["libero_full"][run]["episodes"]


def plus_success(spec: str) -> list[float]:
    """[average, spatial, object, goal, long] in % on the 420 LIBERO-plus variants."""
    r = DATA["libero_plus"][spec]
    return [100 * r["all"], *(100 * r[f"suite/{s}"] for s in SUITES)]


def ranked_policies() -> list[tuple]:
    """POLICIES ordered by LIBERO-plus success, then LIBERO (best first): the order of every table and map."""
    return sorted(POLICIES, key=lambda p: (-plus_success(p[3])[0], -libero_success(p[2])[0]))


# ---- how hard is the task? -----------------------------------------------------------------------------------
def per_task(eps: dict[str, bool], task_of=None) -> dict[str, float]:
    """Success per task from per-episode results ("suite/task_id/init_state" keys)."""
    by: dict[str, list[bool]] = {}
    for k, v in eps.items():
        t = k.rsplit("/", 1)[0]
        by.setdefault(task_of.get(t, t) if task_of else t, []).append(v)
    return {t: float(np.mean(v)) for t, v in by.items()}


def episodes(bench: str, run: str, spec: str) -> dict[str, bool]:
    """Per-episode success of a policy on LIBERO (official protocol) or LIBERO-plus (the 420 variants)."""
    return libero_episodes(run) if bench == "libero" else DATA["plus_episodes"][spec]


def task_matrix(bench: str) -> tuple[list[str], np.ndarray]:
    """Rows: the policies (ranked); columns: the 40 tasks, easiest first."""
    task_of = DATA["plus_variant_task"] if bench == "libero_plus" else None
    rows = [lab for lab, *_ in ranked_policies()]
    mats = [per_task(episodes(bench, run, spec), task_of) for _lab, _role, run, spec in ranked_policies()]
    tasks = sorted(set().union(*mats))
    m = np.array([[mt[t] for t in tasks] for mt in mats])
    return rows, m[:, np.argsort(-m.mean(axis=0), kind="stable")]


def failure_overlap(bench: str) -> list[tuple[str, float, float]]:
    """Per VLA: P(the kNN fails an episode | the VLA fails it), and the kNN's overall failure rate."""
    ref = episodes(bench, *POLICIES[0][2:])
    out = []
    for lab, role, run, spec in ranked_policies():
        if role != "vla":
            continue
        e = episodes(bench, run, spec)
        keys = sorted(set(e) & set(ref))
        a, b = np.array([ref[k] for k in keys]), np.array([e[k] for k in keys])
        if (~b).sum():
            out.append((lab, float((~a & ~b).sum() / (~b).sum()), float((~a).mean())))
    return out


def task_corr(bench: str) -> list[float]:
    """Pearson correlation of per-task success between the kNN (full method) and each VLA."""
    rows, m = task_matrix(bench)
    i = rows.index("kNN, full method")
    roles = {lab: role for lab, role, _r, _s in POLICIES}
    return [float(np.corrcoef(m[i], m[j])[0, 1]) for j, lab in enumerate(rows) if roles[lab] == "vla"]


PLUS_CATEGORIES = [("Camera Viewpoints", "Camera"), ("Robot Initial States", "Robot"), ("Language Instructions", "Language"),
                   ("Light Conditions", "Light"), ("Background Textures", "Background"), ("Sensor Noise", "Noise"),
                   ("Objects Layout", "Layout")]  # fmt: skip


LOOKUP_VLAS = [  # label, head_vs_knn "vla" name of its clean LIBERO recording (X-VLA: absolute poses, not comparable)
    ("π0.5", "pi05_libero"), ("VLA-JEPA", "vla_jepa@lerobot/VLA-JEPA-LIBERO"), ("Evo-1", "evo1_libero"),
    ("MolmoAct2", "molmoact2_libero"), ("GR00T N1.7†", "groot_libero_spatial"), ("π0-FAST", "pi0_fast@lerobot/pi0fast-libero"),
    ("SmolVLA", "smolvla_libero"), ("π0", "pi0_libero"),
]  # fmt: skip


def lookup_rows() -> list[tuple]:
    """Per VLA (clean LIBERO rollouts, ResNet-18 key, k = 8): R² of the lookup's prediction of the VLA's chunks and of
    held-out human chunks, and the VLA's distance to the nearest training chunk relative to a human demo's."""
    out = []
    for label, name in LOOKUP_VLAS:
        h = DATA["head_vs_knn"][name]
        sp, nn = h["spaces"]["resnet18_l3"]["8"], h["nearest_training_chunk"]
        out.append((label, sp["vla_r2"], sp["demo_r2"], nn["vla_median"] / nn["demo_median"]))
    return out


def key_stem(name: str) -> str:
    """Config stem of a backbone key: chosen by choose_keys, else the study's recipe for the older families."""
    fixed = {"pi05_base": "plain_pi05_base_grid2@L12_z", "pi05_libero": "plain_pi05_libero_grid2@L12_z",
             "pi0_base": "plain_pi0_base_grid2@L12_z", "pi0_libero": "plain_pi0_libero_grid2@L12_z",
             "smolvla_base": "plain_smolvla_base_grid2", "smolvla_libero": "plain_smolvla_libero_grid2"}  # fmt: skip
    if name in fixed:
        return fixed[name]
    return next(k["stem"] for k in DATA["extra_keys"] if k["name"] == name)


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


def backbone_rows() -> list[tuple]:
    """Per VLA: (label, [clean, plus] of the lookup on its VLM before / after fine-tuning and of the VLA itself)."""

    def pair(run: str, spec: str) -> tuple[float, float]:
        return libero_success(run)[0], plus_success(spec)[0]

    def key(name: str) -> tuple[float, float]:
        stem = key_stem(name)
        return pair(f"knn:{stem}", f"knn:outputs/vla/configs/{stem}.json")

    out = [
        (label, key(before), key(after), pair(f"vla:{vla}", f"vla:{vla}"))
        for label, vla, before, after in BACKBONES
    ]
    return sorted(out, key=lambda r: (-r[3][1], -r[3][0]))  # as the results table


NEIGHBOR_KEYS = {  # one key per backbone (the recipe variants of π0.5 / SmolVLA are left out): point -> colour role
    "ResNet-18 grid": "generic", "DINOv2-S": "generic", "ResNet-18 grid (learnt key)": "learnt",
    "SmolVLA base grid": "before", "SmolVLA LIBERO grid": "after", "π0.5 base L12": "before", "π0.5 LIBERO L12": "after",
    "π0 base L12": "before", "π0 LIBERO L12": "after",
}  # fmt: skip


def suite_r2(space: str) -> float:
    """Share of the variance of a key's 2-D PCA map that the suite (the scene) explains."""
    xy, suite = np.array(MAPS["spaces"][space]["pca"]), np.array(MAPS["suite"])
    within = sum(((xy[suite == s] - xy[suite == s].mean(0)) ** 2).sum() for s in set(MAPS["suite"]))
    return float(1 - within / ((xy - xy.mean(0)) ** 2).sum())


# ---- embedding maps ------------------------------------------------------------------------------------------
PROGRESS_RAMP = ["#c9dcf4", "#5a9be3", "#1c5cab", "#0b2a55"]  # episode start -> end (sequential, light -> dark)
SUITE_COLOURS = {
    "libero_spatial": "#2a78d6",
    "libero_object": "#eb6834",
    "libero_goal": "#1baf7a",
    "libero_10": "#4a3aa7",
}  # fmt: skip  (validated all-pairs; aqua < 3:1, hence the labelled legend)


def map_panel(ax, space: str, method: str, colour_by: str, stats: bool = True) -> None:
    """One key space's 2-D map (method "tsne" or "pca"), points coloured by episode progress or by suite. t-SNE maps are
    labelled with the neighbour statistics (embedding_maps.py): same-task neighbours overall (``stats``) and on
    LIBERO-Goal, and the phase gap; PCA maps with the variance share of the suite and of the two components."""
    xy = np.array(MAPS["spaces"][space][method])
    if colour_by == "progress":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("progress", PROGRESS_RAMP)
        c = cmap(np.array(MAPS["progress"]))
    else:
        c = np.array([mpl.colors.to_rgba(SUITE_COLOURS[s]) for s in MAPS["suite"]])
    order = np.random.default_rng(0).permutation(len(xy))  # no suite / phase drawn last on top of the others
    ax.scatter(*xy[order].T, s=MAP_POINT_SIZE, c=c[order], linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    v = MAPS["spaces"][space]
    label = f"Goal {100 * v['task_nn/libero_goal']:.0f}% · {v['progress_gap']:.3f}"
    if stats:
        label = f"task {100 * v['task_nn']:.0f}% · " + label
    if method == "pca":
        label = f"suite R² {suite_r2(space):.2f} · PC1+2 {100 * sum(v['pca_variance']):.0f}% of variance"
    ax.set_xlabel(label, fontsize=FS_SMALL - (4 if stats else 3.5), color=INK2, labelpad=1)


def map_legend(fig, colour_by: str) -> None:
    if colour_by == "progress":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("progress", PROGRESS_RAMP)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=fig.axes, location="bottom", shrink=0.35, aspect=30, pad=0.02)
        cb.set_ticks([0, 1], labels=["episode start", "end"])
        cb.ax.tick_params(labelsize=FS_SMALL - 2.5, length=0)
        cb.outline.set_visible(False)
    else:
        names = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
        handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=6, label=names[s])
                   for s, c in SUITE_COLOURS.items()]  # fmt: skip
        fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4, fontsize=FS_SMALL - 2)


# ---- LaTeX tables --------------------------------------------------------------------------------------------
def tex_cell(v: float, fmt: str = "{:.0f}", bold: bool = False, colour: str | None = None) -> str:
    txt = fmt.format(v).replace("-", "−")
    if bold:
        txt = rf"\textbf{{{txt}}}"
    return rf"\textcolor{{{colour}}}{{{txt}}}" if colour else txt


def tex_label(label: str) -> str:
    return (label.replace("π0.5", r"$\pi_{0.5}$").replace("π0-FAST", r"$\pi_0$-FAST").replace("π0", r"$\pi_0$")
            .replace("kNN", r"\knnword{}"))  # fmt: skip
