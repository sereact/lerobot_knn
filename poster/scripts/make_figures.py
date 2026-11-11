"""Render the figures, tables and numbers.tex of the poster (and, with KNN_FIG_TARGET=paper, of the paper).

Everything comes from data/poster_data.json (prepare_data.py). Figures are vector PDFs drawn at print size: the poster's
180 mm columns, or the paper's 5.5 in text width. Colour roles are fixed: kNN (ours) = orange, learned models = blue,
generic encoders = grey, and a VLM's LIBERO training (before / after fine-tuning) = a light / dark blue.
"""

import json
import os
import re
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parents[1]
STYLE = Path(__file__).with_name(
    "figure.mplstyle"
)  # the paper's two-panel figures (serif, LaTeX-like labels)
DATA = json.loads((HERE / "data/poster_data.json").read_text())
MAPS = json.loads((HERE / "data/embedding_maps.json").read_text())
# KNN_FIG_TARGET=paper renders the same figures for the workshop paper (5.5 in text width, 9 pt) into paper/figures
PAPER = os.environ.get("KNN_FIG_TARGET") == "paper"
FIG = HERE.parent / "paper" / "figures" if PAPER else HERE / "figures"
FIG.mkdir(exist_ok=True)

for f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
    font_manager.fontManager.addfont(str(f))
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KNN, LEARNED, HUMAN, HUMAN_DARK = "#eb6834", "#2a78d6", "#9a988f", "#5f5e59"
BEFORE, AFTER = "#86b6ef", "#1c5cab"  # a VLM before / after its LIBERO fine-tuning (validated ordinal pair)
W = 6.45  # inches = 164 mm: a 180.7 mm column minus 2 x 8 mm padding
FS, FS_SMALL = 14, 12.5
if PAPER:  # CoRL template: 5.5 in text width, 10 pt body text
    W, FS, FS_SMALL = 5.5, 9.5, 8.5
mpl.rcParams.update({
    "font.family": "Inter", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "grid.color": GRID,
    "grid.linewidth": 0.7, "lines.linewidth": 2, "pdf.fonttype": 3, "savefig.transparent": True,
})  # fmt: skip
if PAPER:  # the poster's figure heights, scaled to the paper's smaller fonts
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


def save(fig, name: str, **kwargs) -> None:
    fig.savefig(FIG / f"{name}.pdf", **kwargs)  # fixed size: prints 1:1 in a column
    plt.close(fig)


def hbar_axes(ax) -> None:
    """Style the axes of a horizontal bar chart: no y ticks or left spine, vertical grid lines."""
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


def fig_task_map() -> None:
    """Every policy's success on each of the 40 tasks, tasks sorted from easiest to hardest (mean over policies):
    if the policies fail on the same tasks, the rows look alike."""
    panels = [(b, *task_matrix(b)) for b in ("libero", "libero_plus")]
    n = len(POLICIES)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(W, 0.6 + 0.18 * n * len(panels) + 0.4 * len(panels)),
        layout="constrained",
        squeeze=False,
    )
    axes = axes[:, 0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("succ", ["#ffffff", "#c9dcf4", "#5a9be3", "#104281"])
    for ax, (bench, rows, m) in zip(axes, panels, strict=True):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(rows)), rows, fontsize=FS_SMALL - 2.5)
        for tick, lab in zip(ax.get_yticklabels(), rows, strict=True):
            if lab.startswith("kNN"):
                tick.set_color(KNN)
                tick.set_fontweight("bold")
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(
            "LIBERO" if bench == "libero" else "LIBERO-plus (perturbed)",
            fontsize=FS_SMALL,
            color=INK2,
            loc="left",
        )
    axes[-1].set_xlabel(
        "40 tasks, from easiest to hardest (mean over policies)", fontsize=FS_SMALL - 1.5, color=INK2
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.01, aspect=25)
    cb.set_label("task success (%)", fontsize=FS_SMALL - 2)
    cb.ax.tick_params(labelsize=FS_SMALL - 3)
    cb.outline.set_visible(False)
    save(fig, "task_map")


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
# neutral sequential ramp, light -> dark
PLUS_RAMP = ["#f5f6f8", "#dcdee2", "#b9bcc2", "#8e9299", "#5f636b", "#353840"]


def fig_libero_plus() -> None:
    """Success per LIBERO-plus perturbation category (same 420 variants for every policy)."""
    plus = DATA["libero_plus"]
    rows = [(spec, label) for label, _role, _run, spec in ranked_policies()]
    cols = [*[short for _, short in PLUS_CATEGORIES], "all"]
    grid = np.full((len(rows), len(cols)), np.nan)
    for i, (spec, _label) in enumerate(rows):
        r = plus[spec]
        for j, (cat, _) in enumerate(PLUS_CATEGORIES):
            grid[i, j] = 100 * r[f"category/{cat}"]
        grid[i, -1] = 100 * r["all"]
    fig, ax = plt.subplots(figsize=(W, 0.85 + 0.27 * len(rows)), layout="constrained")
    cmap = mpl.colors.LinearSegmentedColormap.from_list("plus", PLUS_RAMP)
    xs = [j + (0.25 if j == len(cols) - 1 else 0) for j in range(len(cols))]
    for i in range(len(rows)):
        for j, x in enumerate(xs):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.add_patch(
                Rectangle(
                    (x - 0.47, i - 0.45), 0.94, 0.9, facecolor=cmap(v / 100), edgecolor="white", linewidth=1.5
                )
            )
            ax.text(
                x,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=FS_SMALL,
                color="white" if v >= 60 else INK,
                weight="bold" if j == len(cols) - 1 else "normal",
            )
    ax.set_xticks(xs, cols, rotation=35, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.tick_params(length=0, labelsize=FS_SMALL, labelcolor=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    save(fig, "libero_plus")


# ---- inside the VLAs -----------------------------------------------------------------------------------------
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


def lookup_bars(
    ax, values: Sequence[float], reference: float, reference_label: str, xmax: float, xlabel: str
) -> None:
    """One horizontal bar per VLA (first row on top), with a dashed line at the reference value."""
    for y, value in zip(np.arange(len(values))[::-1], values, strict=True):
        ax.barh(y, value, color=LEARNED, height=0.62, edgecolor="white")
        ax.text(value + 0.02, y, f"{value:.2f}", va="center", fontsize=FS_SMALL - 2.5, color=INK)
    ax.axvline(reference, color=KNN, linestyle=(0, (4, 3)), linewidth=1.6)
    ax.text(
        reference - 0.01,
        len(values) - 0.45,
        reference_label,
        ha="right",
        va="bottom",
        fontsize=FS_SMALL - 2.5,
        color=KNN,
    )
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(values) - 0.1)
    ax.set_xlabel(xlabel, fontsize=FS_SMALL - 1)
    hbar_axes(ax)
    ax.tick_params(axis="x", labelsize=FS_SMALL - 2.5)


def fig_vla_lookup() -> None:
    """Are VLAs secretly kNN? A lookup over the demos predicts their action chunks, and their chunks sit closer to the
    training data than a new human demonstration does."""
    rows = lookup_rows()
    human = float(np.mean([r[2] for r in rows]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 0.85 + 0.27 * len(rows)), sharey=True, layout="constrained")
    lookup_bars(a, [r[1] for r in rows], human, "human demos", 1.12, "R² of a lookup's prediction")
    lookup_bars(
        b, [r[3] for r in rows], 1.0, "human demos", 1.35, "distance to nearest training chunk (÷ human)"
    )
    a.set_yticks(np.arange(len(rows))[::-1], [r[0] for r in rows])
    save(fig, "vla_lookup")


def fig_vla_lookup_panels() -> None:
    """The two panels of ``fig_vla_lookup`` as separate figures of identical size, in the paper's style."""
    rows = lookup_rows()
    labels = [re.sub(r"π([\d.]+)", r"$\\pi_{\1}$", r[0]).replace("†", r"$^\dagger$") for r in rows]
    human = float(np.mean([r[2] for r in rows]))
    panels = (
        ("vla_lookup_r2", [r[1] for r in rows], human, 1.12, "$R^2$ of a lookup's prediction"),
        (
            "vla_lookup_distance",
            [r[3] for r in rows],
            1.0,
            1.35,
            "Distance to nearest training chunk\nrelative to ground-truth",
        ),
    )
    for name, values, reference, xmax, xlabel in panels:
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(3.4, 0.85 + 0.27 * len(rows)), layout="constrained")
            lookup_bars(ax, values, reference, "Ground-truth actions", xmax, xlabel)
            ax.set_yticks(np.arange(len(rows))[::-1], labels)
            save(fig, name)


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
NEIGHBOR_COLOUR = {"generic": HUMAN_DARK, "learnt": KNN, "before": BEFORE, "after": AFTER}


def fig_neighbors() -> None:
    """Local action consistency of a key (offline, held-out demos) vs the closed-loop success of a lookup with it:
    one point per backbone key, before and after fine-tuning."""
    pts = DATA["neighbor_metric"]["correlation"]["points"]
    roles = dict(NEIGHBOR_KEYS)
    for k in DATA["extra_keys"]:  # the chosen layer of every other backbone
        roles[k["label"]] = "after" if " LIBERO " in k["label"] else "before"
    fig, ax = plt.subplots(figsize=(W, 2.6), layout="constrained")
    fair = [
        (v["lac@8"], v["success"]) for n, v in pts.items() if v["fair"] and n in roles and v["lac@8"] >= 0.7
    ]
    xs, ys = np.array(fair).T
    slope, icpt = np.polyfit(xs, ys, 1)
    gx = np.array([min(xs) - 0.01, max(xs) + 0.01])
    ax.plot(gx, slope * gx + icpt, color=AXIS, linestyle=(0, (4, 3)), linewidth=1.4, zorder=1)
    off = []  # keys far off the scale of the others (a broken key), noted instead of plotted
    for name, role in roles.items():
        if name in pts:
            v = pts[name]
            if v["lac@8"] < 0.7:
                off.append((name, v))
                continue
            ax.scatter(
                [v["lac@8"]],
                [v["success"]],
                s=95,
                color=NEIGHBOR_COLOUR[role],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
    for i, (name, v) in enumerate(off):
        label = "π0 fine-tuned" if name.startswith("π0 LIBERO") else name
        ax.annotate(
            f"← {label}: LAC {v['lac@8']:.2f}, {v['success']:.0f}% (off scale)",
            xy=(0.45, 0.06 + 0.1 * i),
            xycoords="axes fraction",
            fontsize=FS_SMALL - 1.5,
            color=INK2,
        )
    for name, label, offset in (
        ("ResNet-18 grid", "ImageNet ResNet-18", (-8, 5)),
        ("ResNet-18 grid (learnt key)", "learnt key", (-8, 6)),
        ("DINOv2-S", "DINOv2", (8, -4)),
    ):
        ax.annotate(
            label,
            (pts[name]["lac@8"], pts[name]["success"]),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            fontsize=FS_SMALL - 1,
            color=INK2,
        )
    ax.set_xlabel("local action consistency LAC@8 (offline)")
    ax.set_ylabel("lookup on LIBERO (%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=NEIGHBOR_COLOUR[r], markersize=9, label=lab)
        for r, lab in (
            ("generic", "ImageNet / DINOv2"),
            ("before", "VLM before fine-tuning"),
            ("after", "after"),
            ("learnt", "learnt key"),
        )
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="outside upper center",
        ncol=4,
        fontsize=FS_SMALL - 1.5,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    save(fig, "neighbors")


# ---- embedding maps ------------------------------------------------------------------------------------------
# episode start -> end (sequential, light -> dark)
PROGRESS_RAMP = ["#c9dcf4", "#5a9be3", "#1c5cab", "#0b2a55"]
# validated for every pair; the aqua is below 3:1 against white, hence the labelled legend
SUITE_COLOURS = {
    "libero_spatial": "#2a78d6",
    "libero_object": "#eb6834",
    "libero_goal": "#1baf7a",
    "libero_10": "#4a3aa7",
}
SUITE_NAMES = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}
MAP_GRID = [  # (key space, panel title): generic keys and GR00T's frozen VLM, then each VLA before / after fine-tuning
    [("ResNet-18 grid", "ResNet-18"), ("ResNet-18 grid (learnt key)", "learnt key"), ("DINOv2-S", "DINOv2"),
     ("GR00T N1.7 frozen VLM grid2@L4", "GR00T (frozen)")],
    [("π0.5 base L12", "π0.5 before"), ("π0.5 LIBERO L12", "π0.5 after"),
     ("MolmoAct2 base grid2@L27", "MolmoAct2 before"), ("MolmoAct2 LIBERO grid2@L27", "MolmoAct2 after")],
    [("VLA-JEPA base grid2@L7", "VLA-JEPA before"), ("VLA-JEPA LIBERO grid2@L7", "VLA-JEPA after"),
     ("Evo-1 base grid2@L10", "Evo-1 before"), ("Evo-1 LIBERO grid2@L10", "Evo-1 after")],
    [("X-VLA base grid2@L8", "X-VLA before"), ("X-VLA LIBERO grid2@L8", "X-VLA after"),
     ("π0-FAST base grid2@L12", "π0-FAST before"), ("π0-FAST LIBERO grid2@L12", "π0-FAST after")],
    [("SmolVLA base grid", "SmolVLA before"), ("SmolVLA LIBERO grid", "SmolVLA after"),
     ("π0 base L12", "π0 before"), ("π0 LIBERO L12", "π0 after")],
]  # fmt: skip
MAP_PAIRS = [  # the compact figure: (VLA, key before fine-tuning, after)
    ("π0.5", "π0.5 base L12", "π0.5 LIBERO L12"),
    ("Evo-1", "Evo-1 base grid2@L10", "Evo-1 LIBERO grid2@L10"),
    ("MolmoAct2", "MolmoAct2 base grid2@L27", "MolmoAct2 LIBERO grid2@L27"),
    ("π0", "π0 base L12", "π0 LIBERO L12"),
]


def suite_r2(space: str) -> float:
    """Share of the variance of a key's 2-D PCA map that the suite (the scene) explains."""
    xy, suite = np.array(MAPS["spaces"][space]["pca"]), np.array(MAPS["suite"])
    within = sum(((xy[suite == s] - xy[suite == s].mean(0)) ** 2).sum() for s in set(MAPS["suite"]))
    return float(1 - within / ((xy - xy.mean(0)) ** 2).sum())


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
    ax.scatter(*xy[order].T, s=1.2 if PAPER else 2.2, c=c[order], linewidths=0, rasterized=True)
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
    """The colour scale of the maps: a progress colour bar, or one labelled dot per suite."""
    if colour_by == "progress":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("progress", PROGRESS_RAMP)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=fig.axes, location="bottom", shrink=0.35, aspect=30, pad=0.02)
        cb.set_ticks([0, 1], labels=["episode start", "end"])
        cb.ax.tick_params(labelsize=FS_SMALL - 2.5, length=0)
        cb.outline.set_visible(False)
    else:
        handles = [
            Line2D([], [], marker="o", linestyle="", color=c, markersize=6, label=SUITE_NAMES[s])
            for s, c in SUITE_COLOURS.items()
        ]
        fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4, fontsize=FS_SMALL - 2)


def fig_embedding_grid(method: str, colour_by: str) -> None:
    """Every key's map: the generic keys and GR00T's frozen VLM, then each VLA's VLM before / after fine-tuning."""
    fig, axes = plt.subplots(len(MAP_GRID), 4, figsize=(W, W * 1.32), layout="constrained")
    if PAPER:  # undo the paper's height scaling: these panels are square
        fig.set_size_inches(W, W * 1.32)
    for row, panels in zip(axes, MAP_GRID, strict=True):
        for ax, (space, title) in zip(row, panels, strict=True):
            map_panel(ax, space, method, colour_by)
            ax.set_title(title, fontsize=FS_SMALL - 2, color=INK2, pad=2)
    map_legend(fig, colour_by)
    save(fig, f"embedding_{method}_{colour_by}", dpi=300)  # the scatter layers are rasterised


def fig_embedding_pairs() -> None:
    """t-SNE of a few VLMs before (top) and after (bottom) LIBERO fine-tuning, coloured by episode progress, next to
    the ImageNet ResNet-18 (top) and the key learnt for LAC (bottom)."""
    cols = [("ResNet-18", "ResNet-18 grid", "ResNet-18 grid (learnt key)"), *MAP_PAIRS]
    fig, axes = plt.subplots(2, len(cols), figsize=(W, W * 0.5), layout="constrained")
    if PAPER:  # a little flatter, to fit the main text
        fig.set_size_inches(W, W * 0.31)
    for j, (name, before, after) in enumerate(cols):
        for i, space in enumerate((before, after)):
            map_panel(axes[i, j], space, "tsne", "progress", stats=False)
        axes[0, j].set_title(name, fontsize=FS_SMALL - 1, color=INK2, pad=2)
    axes[0, 0].set_ylabel("before fine-tuning", fontsize=FS_SMALL - 2, color=INK2)
    axes[1, 0].set_ylabel("after", fontsize=FS_SMALL - 2, color=INK2)
    axes[1, 0].set_title("learnt key", fontsize=FS_SMALL - 2, color=KNN, pad=2)
    if not PAPER:  # the paper's caption describes the colour scale instead
        map_legend(fig, "progress")
    save(fig, "embedding_pairs", dpi=300)


# ---- the kNN policy: tuning and ablations -------------------------------------------------------------------
OPTUNA_NAMES = {"k": "k (neighbours)", "one_per_episode": "≤ 1 neighbour per demo", "chunk_size": "chunk length",
                "use_consistency": "trajectory stickiness", "w_wrist": "wrist-camera weight",
                "temperature": "softmax temperature", "temporal_ensemble": "temporal ensembling",
                "distance_normalization": "distance normalisation", "w_agentview": "agent-view weight",
                "norm_preserving": "norm preserving", "use_history": "history matching", "use_action_mode": "mode seeking",
                "task_conditioning": "prompt conditioning", "state_weight": "state weight", "time_weight": "time weight",
                "use_time": "time feature", "memory_stride": "memory stride", "image_encoder": "image encoder",
                "pooling_resnet18": "feature pooling", "ensemble_coeff": "ensembling decay",
                "use_ensemble_mode": "ensemble mode seeking", "gripper_mode": "gripper averaging"}  # fmt: skip


def fig_optuna() -> None:
    """The hyper-parameter search: fANOVA importance of each parameter, and every trial's success against k."""
    op = DATA["optuna"]
    imp = list(op["importance"].items())[:8][::-1]
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(W, 3.1), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]}
    )
    a.barh(range(len(imp)), [100 * v for _k, v in imp], color=KNN, height=0.62, edgecolor="white")
    a.set_yticks(range(len(imp)), [OPTUNA_NAMES.get(k, k) for k, _v in imp], fontsize=FS_SMALL - 2)
    for i, (_k, v) in enumerate(imp):
        a.text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", fontsize=FS_SMALL - 2.5, color=INK2)
    a.set_xlabel("importance (fANOVA)", fontsize=FS_SMALL - 1)
    a.set_xlim(0, 75)
    hbar_axes(a)
    trials = [t for t in op["trials"] if "k" in t["params"]]
    for one, colour, lab in ((True, KNN, "≤ 1 per demo"), (False, HUMAN, "any frames")):
        sel = [t for t in trials if bool(t["params"].get("one_per_episode", True)) == one]
        b.scatter(
            [t["params"]["k"] for t in sel],
            [100 * t["value"] for t in sel],
            s=34,
            color=colour,
            edgecolor="white",
            linewidth=0.8,
            label=lab,
            zorder=3,
        )
    b.set_xlabel("k", fontsize=FS_SMALL - 1)
    b.set_ylabel("trial success (%)", fontsize=FS_SMALL - 1)
    b.grid(True)
    b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=FS_SMALL - 2.5, loc="lower right")
    save(fig, "optuna")


# ---- LaTeX tables --------------------------------------------------------------------------------------------
def _cell(v: float, fmt: str = "{:.0f}", bold: bool = False, colour: str | None = None) -> str:
    txt = fmt.format(v).replace("-", "−")
    if bold:
        txt = rf"\textbf{{{txt}}}"
    return rf"\textcolor{{{colour}}}{{{txt}}}" if colour else txt


def _tex_label(label: str) -> str:
    return (
        label.replace("π0.5", r"$\pi_{0.5}$")
        .replace("π0-FAST", r"$\pi_0$-FAST")
        .replace("π0", r"$\pi_0$")
        .replace("kNN", r"\knnword{}")
    )


def table_results() -> None:
    """LIBERO and LIBERO-plus success of every policy, average and per suite, as a LaTeX table (figures/*.tex)."""
    rows = [
        (lab, role, libero_success(run), plus_success(spec)) for lab, role, run, spec in ranked_policies()
    ]
    cols = [[r[2][j] for r in rows] for j in range(5)] + [[r[3][j] for r in rows] for j in range(5)]
    best = [max(c) for c in cols]
    min_drop = min(r[2][0] - r[3][0] for r in rows)
    num = r">{\raggedleft\arraybackslash}X"
    lines = [rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}l*{{5}}{{{num}}}@{{\hspace{{3mm}}}}*{{5}}{{{num}}}@{{\hspace{{3mm}}}}{num}@{{}}}}",
             r"\toprule",
             r" & \multicolumn{5}{c}{LIBERO} & \multicolumn{5}{c}{LIBERO-plus} & \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             r" & avg & Spat & Obj & Goal & Long & avg & Spat & Obj & Goal & Long & drop \\", r"\midrule"]  # fmt: skip
    for i, (lab, role, clean, plus) in enumerate(rows):
        cells = [
            _cell(cols[j][i], "{:.1f}" if j in (0, 5) else "{:.0f}", bold=cols[j][i] >= best[j] - 1e-9)
            for j in range(10)
        ]
        drop = clean[0] - plus[0]
        cells.append(_cell(-drop, "{:+.1f}", bold=drop <= min_drop + 1e-9))
        prefix = r"\rowcolor{tint}" if role != "vla" else ""
        lines.append(prefix + _tex_label(lab) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (FIG / "results_table.tex").write_text("\n".join(lines) + "\n")


ABLATIONS = {  # config stem in outputs/knn/ablations -> label, in the table's order
    "no_language": "prompt ignored", "encoder_pixels": "raw 16×16 pixels", "demos_10": "10% of demos",
    "demos_25": "25% of demos", "demos_50": "50% of demos", "k1": "k = 1 (replay nearest demo)",
    "encoder_dinov2": "DINOv2-S instead of ResNet-18", "no_wrist": "agent-view camera only",
    "no_agentview": "wrist camera only", "no_mode": "plain weighted mean", "no_ensemble": "no temporal ensembling",
    "no_history": "no history matching", "no_state": "no proprioceptive state", "no_time": "no time-step feature",
    "hard_filter": "hard task filter",
}  # fmt: skip


def table_ablations() -> None:
    """One change to the full method at a time, on the official protocol (2,000 episodes; LaTeX table)."""
    abf, full = DATA["ablations_full"], libero_success("knn:00_base")

    def row(label: str, stem: str) -> str:
        r = abf[stem]
        cells = [_cell(v, "{:.1f}") for v in (r["success"], *r["suites"])]
        return f"{label} & " + " & ".join(cells) + f" & {_cell(r['success'] - full[0], '{:+.1f}')}" + r" \\"

    lines = [r"\begin{tabularx}{\linewidth}{@{}X*{6}{r}@{}}", r"\toprule",
             r"Change to the full method & avg & Spatial & Object & Goal & Long & $\Delta$ \\", r"\midrule",
             r"none (full method, \texttt{lerobot-eval}) & " + " & ".join(_cell(v, "{:.1f}") for v in full) + r" & \\"]  # fmt: skip
    lines += [
        row(label.replace("%", r"\%").replace("×", r"$\times$"), stem) for stem, label in ABLATIONS.items()
    ]
    lines.append(r"\midrule")
    lines += [
        row(f"$k$ = {int(k[1:])} (13 in the full method)", k) for k in ("k02", "k04", "k08", "k24", "k32")
    ]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (FIG / "ablations_table.tex").write_text("\n".join(lines) + "\n")


def table_backbones() -> None:
    """The lookup keyed on each VLA's own VLM before / after LIBERO fine-tuning, next to the VLA (LaTeX table).
    Orange: the lookup on the fine-tuned VLM beats the VLA itself."""
    lines = [r"\begin{tabularx}{\linewidth}{@{}l*{6}{>{\raggedleft\arraybackslash}X}@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{LIBERO} & \multicolumn{3}{c}{LIBERO-plus} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r" & \knnword{} before & \knnword{} after & VLA & \knnword{} before & \knnword{} after & VLA \\", r"\midrule"]  # fmt: skip
    for label, before, after, vla in backbone_rows():
        cells = []
        for col in (0, 1):
            b, a, v = before[col], after[col], vla[col]
            beats = a >= v
            cells += [
                _cell(b, "{:.1f}"),
                _cell(a, "{:.1f}", bold=beats, colour="knn" if beats else None),
                _cell(v, "{:.1f}", colour="learned"),
            ]
        lines.append(_tex_label(label) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (FIG / "backbones_table.tex").write_text("\n".join(lines) + "\n")


# ---- numbers.tex ---------------------------------------------------------------------------------------------
PLUS_VLAS = {"Pi": "vla:pi05_libero", "Xvla": "vla:xvla@lerobot/xvla-libero", "Smol": "vla:smolvla_libero",
             "Fast": "vla:pi0_fast@lerobot/pi0fast-libero", "PiZero": "vla:pi0_libero",
             "Jepa": "vla:vla_jepa@lerobot/VLA-JEPA-LIBERO"}  # fmt: skip


def policy_numbers() -> dict[str, str]:
    """Success of the kNN and the VLAs on LIBERO and LIBERO-plus, drops under perturbation and ranks."""
    plus, control, abf = DATA["libero_plus"], DATA["libero_plus_control"], DATA["ablations_full"]
    knn_plus = plus["knn:outputs/knn/v2_39.json"]
    m = {
        "KnnAvg": f1(libero_success("knn:00_base")[0]),
        "KnnPlainResnet": f1(libero_success("knn:plain_resnet18_l3")[0]),
        "KnnLearnt": f1(libero_success("knn:plain_resnet18_l3_learned")[0]),
        "KnnFullLearnt": f1(libero_success("knn:full_resnet18_l3_learned")[0]),
        "KnnPiZeroBaseLt": f1(libero_success("knn:plain_pi0_base_grid2@L12_z")[0]),
        "KnnPiZeroLt": f1(libero_success("knn:plain_pi0_libero_grid2@L12_z")[0]),
        "AblNoLang": f1(abf["no_language"]["success"]),
        "AblPixels": f1(abf["encoder_pixels"]["success"]),
        "DemosTen": f1(abf["demos_10"]["success"]),
        "PlusKnn": f1(100 * knn_plus["all"]),
        "PlusN": str(knn_plus["n_episodes"]),
        "PlusPerCat": str(knn_plus["n_episodes"] // len(PLUS_CATEGORIES)),
        "PlusXvlaCamera": f"{100 * plus[PLUS_VLAS['Xvla']]['category/Camera Viewpoints']:.0f}",
    }
    for tag in ("Smol", "Fast", "PiZero"):  # reworded instructions vs the original wording, same scenes
        m[f"Plus{tag}Language"] = f"{100 * plus[PLUS_VLAS[tag]]['category/Language Instructions']:.0f}"
        m[f"Plus{tag}Ctrl"] = f"{100 * control[PLUS_VLAS[tag]]:.0f}"

    # drop under perturbation, and the kNN's rank among the VLAs
    drops = {lab: libero_success(run)[0] - plus_success(spec)[0] for lab, _role, run, spec in POLICIES}
    vla_drops = sorted(drops[lab] for lab, role, *_ in POLICIES if role == "vla")
    m.update(
        KnnDrop=f1(drops["kNN, full method"]),
        MinVlaDrop=f1(vla_drops[0]),
        MaxVlaDrop=f1(vla_drops[-1]),
        VlaN=str(len(vla_drops)),
    )
    entrants = [p for p in POLICIES if p[1] == "vla" or p[0] == "kNN, full method"]
    for macro, score in (
        ("KnnRankLibero", lambda p: libero_success(p[2])[0]),
        ("KnnRankPlus", lambda p: plus_success(p[3])[0]),
    ):
        names = [p[0] for p in sorted(entrants, key=score, reverse=True)]
        m[macro] = str(1 + names.index("kNN, full method"))
    m["RankN"] = str(len(entrants))
    best_plus = 0  # LIBERO-plus suites where a kNN row is the best policy
    for j in range(1, 5):
        vals = [(plus_success(spec)[j], role) for _lab, role, _run, spec in POLICIES]
        top = max(v for v, _ in vals)
        best_plus += any(v >= top - 1e-9 and role != "vla" for v, role in vals)
    m["KnnBestPlusSuites"] = ["no", "one", "two", "three", "all four"][best_plus]
    return m


def failure_numbers() -> dict[str, str]:
    """Do the policies fail on the same episodes and tasks?"""
    m: dict[str, str] = {}
    fo = failure_overlap("libero")
    lifts = [c / base for _lab, c, base in fo]
    tc = sorted(task_corr("libero"))
    m.update(
        FailBase=f"{100 * fo[0][2]:.0f}",
        FailMin=f"{100 * min(c for _l, c, _b in fo):.0f}",
        FailMax=f"{100 * max(c for _l, c, _b in fo):.0f}",
        LiftMin=f"{min(lifts):.1f}",
        LiftMax=f"{max(lifts):.1f}",
        TaskCorrMedian=f"{tc[len(tc) // 2]:.2f}",
    )
    return m


def backbone_numbers() -> dict[str, str]:
    """Backbone and action head: the lookup on each VLA's VLM before / after fine-tuning vs the VLA."""
    m: dict[str, str] = {}
    rows = backbone_rows()
    tuned = [r for r in rows if r[1] != r[2]]  # GR00T froze its VLM: before = after
    up = [after[1] - before[1] for _l, before, after, _v in tuned if after[1] > before[1]]
    subtract = [after[1] - vla[1] for _l, _b, after, vla in rows if after[1] >= vla[1]]
    m.update(
        KeyN=str(len(rows)),
        BeatenClean=str(sum(after[0] >= vla[0] for _l, _b, after, vla in rows)),
        KeyNPlus=str(len(rows)),
        BeatenPlus=str(len(subtract)),
        FtN=str(len(tuned)),
        FtUp=str(len(up)),
        FtUpMin=f"{min(up):.0f}",
        FtUpMax=f"{max(up):.0f}",
        FtCleanUp=str(sum(after[0] > before[0] + 1 for _l, before, after, _v in tuned)),
        SubtractMax=f"{max(subtract):.0f}",
    )
    return m


def lookup_numbers() -> dict[str, str]:
    """How well a lookup predicts the VLAs' action chunks."""
    m: dict[str, str] = {}
    lookups = {lab: (r2, demo) for lab, r2, demo, _ratio in lookup_rows()}
    m["LookupPi"] = f"{lookups['π0.5'][0]:.2f}"
    m["LookupHuman"] = f"{np.mean([demo for _r2, demo in lookups.values()]):.2f}"
    # the best lookup in the VLA's own key spaces on LIBERO-plus rollouts
    for tag in ("Pi", "Jepa", "Smol", "Fast"):
        r2 = DATA["head_vs_knn_plus"][PLUS_VLAS[tag][4:]]["r2_k8"]
        m[f"HeadPlus{tag}"] = f"{max(v for k, v in r2.items() if k != 'resnet18_l3'):.2f}"
    return m


def neighbor_numbers() -> dict[str, str]:
    """Local action consistency of the keys and its correlation with closed-loop success."""
    m: dict[str, str] = {}
    nm = DATA["neighbor_metric"]
    fair = nm["correlation"]["fair"]
    # one key per backbone, as in fig_neighbors
    shown = [*NEIGHBOR_KEYS, *(k["label"] for k in DATA["extra_keys"])]
    others = [nm["spaces"][k]["lac@8"] for k in shown if k != "π0 LIBERO L12"]
    m.update(NmOtherMin=f"{min(others):.2f}", NmOtherMax=f"{max(others):.2f}")
    m.update(
        NmFairRho=f"{fair['spearman_lac@8']:.2f}",
        NmFairR=f"{fair['pearson_lac@8']:.2f}",
        NmFairN=str(fair["n"]),
        NmPiZeroLibero=f"{nm['spaces']['π0 LIBERO L12']['lac@8']:.2f}",
    )
    return m


def map_numbers() -> dict[str, str]:
    """Embedding maps: same-task neighbours on LIBERO-Goal (one scene, ten instructions), the phase gap and the
    share of the map's variance the suite explains."""
    m: dict[str, str] = {}
    spaces = MAPS["spaces"]
    goal = {name: 100 * v["task_nn/libero_goal"] for name, v in spaces.items()}
    # bidirectional prefix attention over image + text
    sees_prompt = [n for n in goal if n.startswith(("π0", "X-VLA"))]
    causal = [n for n in goal if spaces[n]["stage"] in ("before", "after") and n not in sees_prompt]
    pairs = [
        (b, a)
        for b in spaces
        for a in spaces
        if spaces[b]["family"] == spaces[a]["family"]
        and spaces[b]["stage"] == "before"
        and spaces[a]["stage"] == "after"
    ]
    phase = {n: v["progress_gap"] for n, v in spaces.items()}
    m.update(
        MapN=f"{len(MAPS['progress']):,}",
        MapGoalResnet=f"{goal['ResNet-18 grid']:.0f}",
        MapGoalPromptMin=f"{min(goal[n] for n in sees_prompt):.0f}",
        MapGoalCausalMin=f"{min(goal[n] for n in causal):.0f}",
        MapGoalCausalMax=f"{max(goal[n] for n in causal):.0f}",
        MapPhaseN=str(len(pairs)),
        MapPhaseDown=str(sum(phase[a] < phase[b] for b, a in pairs)),
        MapPhaseEvoBefore=f"{phase['Evo-1 base grid2@L10']:.3f}",
        MapPhaseEvoAfter=f"{phase['Evo-1 LIBERO grid2@L10']:.3f}",
        MapPhasePiZeroBefore=f"{phase['π0 base L12']:.3f}",
        MapPhasePiZeroAfter=f"{phase['π0 LIBERO L12']:.3f}",
        MapGoalPiZeroAfter=f"{goal['π0 LIBERO L12']:.0f}",
    )
    r2 = {n: suite_r2(n) for n in spaces}
    before, after = [r2[b] for b, _a in pairs], [r2[a] for _b, a in pairs]
    m.update(
        MapSuiteResnet=f"{100 * r2['ResNet-18 grid']:.0f}",
        MapSuiteBeforeMin=f"{100 * min(before):.0f}",
        MapSuiteBeforeMax=f"{100 * max(before):.0f}",
        MapSuiteAfterMin=f"{100 * min(after):.0f}",
        MapSuiteAfterMax=f"{100 * max(after):.0f}",
        MapSuiteDown=str(sum(r2[a] < r2[b] for b, a in pairs)),
    )
    return m


def search_numbers() -> dict[str, str]:
    """The kNN policy: hyper-parameter search and retry."""
    m: dict[str, str] = {}
    op = DATA["optuna"]
    m.update(
        OptunaN=str(op["n_trials"]),
        OptunaAll=str(op["n_trials_all_studies"]),
        OptunaK=f"{100 * op['importance']['k']:.0f}",
        OptunaOne=f"{100 * op['importance']['one_per_episode']:.0f}",
    )
    retry = DATA["retry"]
    others = [v for k, v in retry.items() if k != "base"]
    m.update(
        RetryBase=f1(100 * retry["base"]), RetryMin=f1(100 * min(others)), RetryMax=f1(100 * max(others))
    )
    return m


def numbers() -> None:
    """Every number the poster and the paper quote in their text, as LaTeX macros."""
    m = {
        **policy_numbers(),
        **failure_numbers(),
        **backbone_numbers(),
        **lookup_numbers(),
        **neighbor_numbers(),
        **map_numbers(),
        **search_numbers(),
    }
    lines = ["% generated by poster/scripts/make_figures.py -- do not edit"]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in m.items()]
    (FIG.parent if PAPER else HERE).joinpath("numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_task_map()
    fig_libero_plus()
    fig_vla_lookup()
    fig_vla_lookup_panels()
    fig_neighbors()
    fig_optuna()
    fig_embedding_pairs()
    if PAPER:  # appendix figures
        fig_embedding_grid("tsne", "progress")
        fig_embedding_grid("pca", "suite")
    table_results()
    table_backbones()
    table_ablations()
    numbers()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
