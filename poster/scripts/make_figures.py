"""Render the poster's figures (vector PDF, at print size) and numbers.tex from data/poster_data.json.

A1 portrait poster, three 180 mm columns: every figure is drawn at the column width, so its text prints at the
sizes set below, next to 15.5 pt body text. Colour roles are fixed across the poster: kNN (ours) = orange,
learned models = blue, human demonstrations / generic encoders = grey, and the "how much robot / LIBERO training
has the VLM seen" ladder = an ordinal blue ramp. Figures and table rows appear only for experiments that have
run; numbers.tex carries \\ifhave... flags so the poster adds the matching blocks automatically.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parents[1]
DATA = json.loads((HERE / "data/poster_data.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

for f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
    font_manager.fontManager.addfont(str(f))
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KNN, KNN_LIGHT, LEARNED, HUMAN, HUMAN_DARK = "#eb6834", "#f3a07c", "#2a78d6", "#9a988f", "#5f5e59"
LADDER = [
    "#86b6ef",
    "#2a78d6",
    "#104281",
]  # generic VLM -> robot-pretrained -> LIBERO-tuned (validated ordinal)
LIBERO_STAGE = [
    "#86b6ef",
    "#3987e5",
    "#1c5cab",
    "#0d366b",
]  # VLM untouched by LIBERO -> ... -> longest (validated)
W = 6.45  # inches = 164 mm: a 180.7 mm column minus 2 x 8 mm padding
FS, FS_SMALL = 14, 12.5
mpl.rcParams.update({
    "font.family": "Inter", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "grid.color": GRID,
    "grid.linewidth": 0.7, "lines.linewidth": 2, "pdf.fonttype": 3, "savefig.transparent": True,
})  # fmt: skip
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_NAMES = ["Spatial", "Object", "Goal", "Long"]
VLM_POOL = "grid2"  # the head attends to every token, so the main VLM key keeps their 2x2 layout


def f1(v: float) -> str:
    """One decimal, rounding halves up (90.55 -> 90.6, as the papers report), not to even."""
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")  # fixed size: prints 1:1 in a column
    plt.close(fig)


def hbar_axes(ax) -> None:
    ax.tick_params(axis="y", length=0, labelsize=FS_SMALL + 0.5, labelcolor=INK2)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def best_vla(name: str) -> float | None:
    """The VLA's own closed-loop success (%), best of the execution horizons tried."""
    hits = [100 * v["success"] for v in DATA.get("vla_success", {}).values() if v["name"] == name]
    return max(hits) if hits else None


def vla_run(name: str) -> dict | None:
    runs = [v for v in DATA.get("vla_success", {}).values() if v["name"] == name]
    return max(runs, key=lambda v: v["success"]) if runs else None


# ---- middle: leaderboard -------------------------------------------------------------------------------------
def fig_leaderboard() -> None:
    short = {
        "Diffusion Policy (from scratch)": "Diffusion Policy",
        "Octo (fine-tuned)": "Octo",
        "OpenVLA 7B (fine-tuned)": "OpenVLA (7B)",
        "Multi-task DiT (diffusion, LeRobot)": "Multi-task DiT",
        "π0 (fine-tuned)": "π0",
        "OpenVLA-OFT": "OpenVLA-OFT",
        "π0.5 (LeRobot)": "π0.5",
    }
    rows = [(short.get(b["name"], b["name"]), b["avg"], LEARNED) for b in DATA["baselines"]]
    rows.append(("kNN lookup (ours)", DATA["knn_official"]["avg"], KNN))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(W, 2.75), layout="constrained")
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.7, edgecolor="white", linewidth=1)
    for yi, (_name, v, c) in zip(y, rows, strict=True):
        ax.text(
            v + 0.5,
            yi,
            f1(v),
            va="center",
            fontsize=FS_SMALL,
            color=INK,
            weight="bold" if c == KNN else "normal",
        )
    ax.set_yticks(y, [r[0] for r in rows])
    hbar_axes(ax)
    for lbl, (_, _, c) in zip(ax.get_yticklabels(), rows, strict=True):
        lbl.set_color(INK if c == KNN else INK2)
        lbl.set_fontweight("bold" if c == KNN else "normal")
    ax.set_xlim(50, 101)
    ax.set_xlabel("LIBERO success (%), mean of 4 suites")
    save(fig, "leaderboard")


# ---- middle: which episodes need more than a lookup? --------------------------------------------------------
WIN_VLAS = [  # (vla_success key "tag:name", label): each VLA at its best execution horizon
    ("pi05_n10:pi05_libero", "π0.5 (LeRobot)"),
    ("hub_other_n10:xvla@lerobot/xvla-libero", "X-VLA"),
    ("hub_other_n10:pi0_fast@lerobot/pi0fast-libero", "π0-FAST"),
    ("smolvla_n10:smolvla_libero", "SmolVLA"),
    ("hub_other_n10:pi0_libero", "π0"),
    ("vla_jepa_n7:vla_jepa@lerobot/VLA-JEPA-LIBERO", "VLA-JEPA"),
    ("evo1:evo1_libero", "Evo-1"),
    ("molmoact2:molmoact2_libero", "MolmoAct2"),
    ("groot_n17:groot_libero", "GR00T N1.7"),
]


def fig_episode_wins() -> bool:
    """Per VLA, the episodes it solves that a training-free ImageNet lookup does not (right) and vice versa (left)."""
    ov = DATA.get("episode_overlap", {}).get("vlas", {})
    rows = [(label, ov[k]["plain"]) for k, label in WIN_VLAS if k in ov and "plain" in ov[k]]
    if len(rows) < 3:
        return False
    rows.sort(key=lambda r: r[1]["vla_only"] - r[1]["lookup_only"])
    fig, ax = plt.subplots(figsize=(W, 0.95 + 0.34 * len(rows)), layout="constrained")
    y = np.arange(len(rows))
    lo = [-r[1]["lookup_only"] for r in rows]
    hi = [r[1]["vla_only"] for r in rows]
    ax.barh(y, lo, color=KNN, height=0.66, edgecolor="white", linewidth=1)
    ax.barh(y, hi, color=LEARNED, height=0.66, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK2, linewidth=1)
    for yi, (_label, c) in zip(y, rows, strict=True):
        ax.text(
            -c["lookup_only"] - 2,
            yi,
            str(c["lookup_only"]),
            ha="right",
            va="center",
            fontsize=FS_SMALL,
            color=INK,
        )
        ax.text(
            c["vla_only"] + 2, yi, str(c["vla_only"]), ha="left", va="center", fontsize=FS_SMALL, color=INK
        )
        ok = c["both"] + c["vla_only"]
        ax.text(1.02, yi, f"{100 * c['both'] / ok:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=FS_SMALL, color=INK2)  # fmt: skip
    ax.text(1.02, len(rows) - 0.35, "shared", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=FS_SMALL - 1, color=MUTED)  # fmt: skip
    labels = [f"{label} {f1(100 * (c['both'] + c['vla_only']) / 400)}" for label, c in rows]
    ax.set_yticks(y, labels)
    hbar_axes(ax)
    lim = max(max(hi), -min(lo)) + 22
    ax.set_xlim(-lim, max(hi) + 22)
    ticks = [t for t in (-120, -80, -40, 0, 40) if -lim <= t <= max(hi) + 22]
    ax.set_xticks(ticks, [str(abs(t)) for t in ticks])
    ax.set_xlabel("episodes (of 400)")
    ax.set_ylim(-0.6, len(rows) + 0.15)
    ax.text(
        -3, len(rows) - 0.35, "only the lookup solves", ha="right", va="bottom", fontsize=FS_SMALL, color=INK2
    )
    ax.text(3, len(rows) - 0.35, "only the VLA solves", ha="left", va="bottom", fontsize=FS_SMALL, color=INK2)
    save(fig, "episode_wins")
    return True


# ---- middle: the diffusion transformer's chunks ----------------------------------------------------------------
def fig_dit() -> None:
    m = DATA["dit_memorization"]
    rows = [
        ("DiT chunk → nearest training chunk", m["dit_to_train"], LEARNED),
        ("new human demo → nearest other demo", m["human_to_other_demos"], HUMAN),
        ("DiT chunk → kNN answer, same state", m["dit_to_knn"], LEARNED),
        ("kNN's own neighbours → kNN answer", m["knn_spread"], KNN),
    ]
    fig, ax = plt.subplots(figsize=(W, 2.45), layout="constrained")
    ys = [3.15, 2.35, 0.95, 0.15]
    for (_label, vals, c), y in zip(rows, ys, strict=True):
        v = np.asarray(vals)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        lo, hi = np.percentile(v, [5, 95])
        ax.plot([lo, hi], [y, y], color=c, linewidth=1.5, solid_capstyle="round")
        ax.add_patch(
            Rectangle((q1, y - 0.25), q3 - q1, 0.5, facecolor=c, alpha=0.25, edgecolor=c, linewidth=1.2)
        )
        ax.plot([med, med], [y - 0.25, y + 0.25], color=c, linewidth=2.5)
        ax.text(hi * 1.15, y, f"{med:.4f}", va="center", fontsize=FS_SMALL, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 1.0)
    ax.set_yticks(ys, [r[0] for r in rows])
    hbar_axes(ax)
    ax.set_ylim(-0.4, 3.6)
    ax.axhline(1.65, color=GRID, linewidth=1)
    ax.set_xlabel("8-step motion chunk difference (squared, log)")
    save(fig, "dit_vs_knn")


# ---- middle: when does the lookup fail? ------------------------------------------------------------------------
def fig_failures() -> bool:
    eps = DATA.get("failure_diag")
    if not eps:
        return False
    fig, ax = plt.subplots(figsize=(W, 1.55), layout="constrained")
    rng = np.random.default_rng(0)
    # all are kNN episodes: outcome is encoded by fill (and labelled), not by another hue
    for row, (ok, label) in enumerate(((False, "failed"), (True, "succeeded"))):
        v = np.array([e["peak_distance"] for e in eps if e["success"] == ok])
        ax.scatter(v, row + rng.uniform(-0.12, 0.12, len(v)), s=36, facecolor=KNN if ok else "white", edgecolor=KNN,
                   linewidth=1.4, zorder=3)  # fmt: skip
        ax.text(v.max() + 0.15, row, f"{label}: {len(v)}", va="center", fontsize=FS_SMALL, color=INK2)
    ax.axvline(3.0, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(1.5, 10)
    ax.set_xlabel("largest distance to the nearest demo frame in the episode (× typical)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    save(fig, "failures")
    return True


# ---- left: ablations ------------------------------------------------------------------------------------------
def fig_ablations() -> None:
    keep = ["no_language",
        "demos_10",
        "demos_50", "encoder_pixels", "k1", "encoder_dinov2", "no_wrist", "no_agentview", "no_mode",
               "no_history", "no_ensemble"]  # fmt: skip
    ab = DATA["ablations"]
    rows = sorted([(ab[k]["label"], ab[k]["success"]) for k in keep if k in ab], key=lambda r: r[1])
    base = DATA["ablation_base"]
    fig, ax = plt.subplots(figsize=(W, 3.3), layout="constrained")
    y = np.arange(len(rows))
    ax.axvline(base, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        base + 0.4, -0.95, f"full method {f1(base)}", ha="left", va="center", fontsize=FS_SMALL, color=INK2
    )
    for yi, (_label, v) in zip(y, rows, strict=True):
        ax.plot([base, v], [yi, yi], color=AXIS, linewidth=2, zorder=2, solid_capstyle="round")
        ax.scatter([v], [yi], s=46, color=KNN, edgecolor="white", linewidth=1, zorder=3)
        left = v < base
        ax.text(v + (-0.8 if left else 0.8), yi, f1(v), va="center", ha="right" if left else "left",
                fontsize=FS_SMALL, color=INK)  # fmt: skip
    ax.set_yticks(y, [r[0] for r in rows])
    hbar_axes(ax)
    ax.set_xlim(50, 96)
    ax.set_ylim(-1.4, len(rows) - 0.4)
    ax.set_xlabel("success (%) with one ingredient changed")
    save(fig, "ablations")


# ---- right: backbone vs action expert ---------------------------------------------------------------------
def best_lookup(stem: str) -> float | None:
    """Best closed-loop success (%) of the plain lookup on a VLM over the keys tried (layer, normalisation, codec)."""
    runs = DATA.get("knn_with_vla_embedders", {})
    hits = [
        100 * v["success"] for k, v in runs.items() if k == stem or k.startswith((stem + "_", stem + "@"))
    ]
    base = stem.removesuffix("_grid2")
    hits += [100 * v["success"] for k, v in runs.items() if k.startswith(base + "_grid2@")]
    return max(hits) if hits else None


def first_run(stem: str | tuple[str, ...]) -> str | None:
    """A config stem, or of a tuple of stems the first that has run."""
    runs = DATA.get("knn_with_vla_embedders", {})
    if isinstance(stem, str):
        return stem
    return next((x for x in stem if x in runs), stem[0])


DIAG_POINTS = [  # the VLA, the plain lookup on its own VLM, label, marker
    ("smolvla_libero", "plain_smolvla_libero_grid2", "SmolVLA", "o"),
    ("pi05_libero", "plain_pi05_libero_grid2@L12_z", "π0.5", "s"),
    ("pi0_libero", "plain_pi0_libero_grid2@L12_z", "π0", "^"),
    ("xvla@lerobot/xvla-libero", "plain_xvla_libero_grid2@L8_z", "X-VLA", "D"),
    ("pi0_fast@lerobot/pi0fast-libero", "plain_pi0fast_libero_grid2@L12_z", "π0-FAST", "v"),
    ("vla_jepa@lerobot/VLA-JEPA-LIBERO", "plain_vla_jepa_libero_grid2@L7_z", "VLA-JEPA", "P"),
    # its LIBERO checkpoints froze the VLM, so the base's is its own; the layer is the pilot's choice (choose_keys)
    ("groot_libero", tuple(f"plain_groot_base_grid2{t}_z" for t in ("@L4", "@L8", "@L12", "")), "GR00T N1.7", "X"),
]  # fmt: skip


def fig_diagnostic() -> bool:
    """Backbone vs action expert: each VLA against a plain lookup keyed on its own VLM, on the same 400 LIBERO
    episodes and on the same 420 LIBERO-plus variants (one row per VLA, sorted by the clean gain)."""
    runs = DATA.get("knn_with_vla_embedders", {})
    plus = DATA.get("libero_plus", {})
    rows = []
    for vla, stem, label, _marker in DIAG_POINTS:
        stem = first_run(stem)
        best = best_vla(vla)
        if best is None or stem not in runs:
            continue
        lookup_plus = plus.get(f"knn:outputs/vla/configs/{stem}.json")
        vla_plus = plus.get(f"vla:{vla}")
        rows.append((label, (100 * runs[stem]["success"], best),
                     (100 * lookup_plus["all"], 100 * vla_plus["all"]) if lookup_plus and vla_plus else None))  # fmt: skip
    if len(rows) < 2:
        return False
    rows.sort(key=lambda r: r[1][1] - r[1][0])
    panels = [("LIBERO (clean)", 1)] + ([("LIBERO-plus (perturbed)", 2)] if any(r[2] for r in rows) else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(W, 1.05 + 0.36 * len(rows)), sharey=True, layout="constrained",
                             squeeze=False)  # fmt: skip
    y = np.arange(len(rows))
    for ax, (title, col) in zip(axes[0], panels, strict=True):
        for yi, row in zip(y, rows, strict=True):
            if row[col] is None:
                ax.text(50, yi, "pending", ha="center", va="center", fontsize=FS_SMALL - 1, color=MUTED)
                continue
            lookup, vla = row[col]
            ax.plot([lookup, vla], [yi, yi], color=AXIS, linewidth=2.5, solid_capstyle="round", zorder=1)
            ax.scatter([lookup], [yi], s=70, color=KNN, edgecolor="white", linewidth=1, zorder=3)
            ax.scatter([vla], [yi], s=70, color=LEARNED, edgecolor="white", linewidth=1, zorder=3)
            gain = Decimal(f1(vla)) - Decimal(f1(lookup))  # the difference of the rounded numbers the text quotes
            ax.text(104, yi, f"{'+' if gain > 0 else '−' if gain < 0 else ''}{abs(gain):.0f}", ha="left", va="center",
                    fontsize=FS_SMALL, color=INK if gain > 0 else KNN, fontweight="bold" if gain < 0 else "normal")  # fmt: skip
        ax.set_xlim(-2, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=FS_SMALL + 0.5, color=INK2, loc="left")
        hbar_axes(ax)
    axes[0][0].set_yticks(y, [r[0] for r in rows])
    axes[0][0].set_ylim(-0.6, len(rows) - 0.4)
    fig.supxlabel("success (%)", fontsize=FS, color=INK2)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=9, label=lab)
               for c, lab in ((KNN, "plain lookup on the VLA's own VLM"), (LEARNED, "the VLA"))]  # fmt: skip
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncol=2)
    save(fig, "diagnostic")
    return True


KEY_RECIPES = [  # label, lookup before LIBERO fine-tuning, after, the VLA itself (closed loop, same 400 episodes)
    ("SmolVLA · last layer", "plain_smolvla_base_grid2", "plain_smolvla_libero_grid2", "smolvla_libero"),
    ("π0.5 · last layer, cosine", "plain_pi05_base_grid2", "plain_pi05_libero_grid2", "pi05_libero"),
    ("π0.5 · last layer, z-scored", "plain_pi05_base_grid2_z", "plain_pi05_libero_grid2_z", "pi05_libero"),
    ("π0.5 · layer 12 of 18, z-scored", "plain_pi05_base_grid2@L12_z", "plain_pi05_libero_grid2@L12_z", "pi05_libero"),
    ("π0.5 · layers 12+15+18, PCA", "plain_pi05_base_grid2@L12+L15+last_pca", "plain_pi05_libero_grid2@L12+L15+last_pca",
     "pi05_libero"),
]  # fmt: skip


def fig_vlm_keys() -> bool:
    """Closed-loop success of the plain lookup keyed on a VLM, per key recipe, before vs after LIBERO fine-tuning."""
    runs = DATA.get("knn_with_vla_embedders", {})
    rows = [(label, first_run(b), first_run(a), v) for label, b, a, v in KEY_RECIPES]
    rows = [r for r in rows if r[1] in runs or r[2] in runs]
    if len(rows) < 3:
        return False
    fig, ax = plt.subplots(figsize=(W, 1.0 + 0.36 * len(rows)), layout="constrained")
    y = np.arange(len(rows))[::-1]
    resnet = runs.get("plain_resnet18_l3")
    if resnet:
        ax.axvline(100 * resnet["success"], color=HUMAN, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(100 * resnet["success"], len(rows) - 0.35, "ImageNet\nResNet-18", ha="center", va="bottom",
                fontsize=FS_SMALL - 1, color=INK2)  # fmt: skip
    for yi, (_label, before, after, vla) in zip(y, rows, strict=True):
        vals = [
            (100 * runs[k]["success"], c)
            for k, c in ((before, LIBERO_STAGE[0]), (after, LIBERO_STAGE[2]))
            if k in runs
        ]
        if len(vals) == 2:
            ax.plot(
                [vals[0][0], vals[1][0]],
                [yi, yi],
                color=AXIS,
                linewidth=2.2,
                zorder=2,
                solid_capstyle="round",
            )
        for v, c in vals:
            ax.scatter([v], [yi], s=70, color=c, edgecolor="white", linewidth=1, zorder=3)
        lo, hi = min(v for v, _ in vals), max(v for v, _ in vals)
        if len(vals) == 2:
            ax.text(lo - 0.9, yi, f1(lo), ha="right", va="center", fontsize=FS_SMALL, color=INK)
        ax.text(hi + 0.9, yi, f1(hi), ha="left", va="center", fontsize=FS_SMALL, color=INK)
        best = best_vla(vla)
        if best is not None:
            ax.scatter([best], [yi], marker="D", s=40, color=INK, zorder=4)
    ax.set_yticks(y, [r[0] for r in rows])
    hbar_axes(ax)
    ax.set_xlim(60, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("plain lookup success (%)")
    handles = [Line2D([], [], marker="o", linestyle="", color=LIBERO_STAGE[0], markersize=8, label="VLM before LIBERO"),
               Line2D([], [], marker="o", linestyle="", color=LIBERO_STAGE[2], markersize=8, label="after fine-tuning"),
               Line2D([], [], marker="D", linestyle="", color=INK, markersize=6, label="the VLA itself")]  # fmt: skip
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncol=3)
    save(fig, "vlm_keys")
    return True


NEIGHBOR_STYLE = [  # key-space name prefix -> colour (how much LIBERO training its weights saw)
    ("ResNet-18 grid (learnt key)", KNN),
    ("ResNet", HUMAN_DARK),
    ("DINOv2", HUMAN),
    ("SmolVLA base", LIBERO_STAGE[0]),
    ("π0.5 base", LIBERO_STAGE[0]),
    ("SmolVLA", LIBERO_STAGE[2]),
    ("π0.5", LIBERO_STAGE[2]),
]
NEIGHBOR_LABELS = {  # key space -> (label, text offset in points)
    "ResNet-18 grid": ("ResNet-18", (-8, 6)), "DINOv2-S": ("DINOv2", (-8, 6)), "SmolVLA base grid": ("SmolVLA base", (7, -4)),
    "SmolVLA LIBERO grid": ("SmolVLA LIBERO", (6, 5)), "π0.5 base L12": ("π0.5 base L12", (6, -13)),
    "π0.5 LIBERO L12": ("π0.5 LIBERO L12", (7, -12)), "ResNet-18 grid (learnt key)": ("learnt key", (7, 3)),
}  # fmt: skip
NEIGHBOR_ARROWS = [("ResNet-18 grid", "ResNet-18 grid (learnt key)", KNN), ("SmolVLA base grid", "SmolVLA LIBERO grid", AXIS),
                   ("π0.5 base L12", "π0.5 LIBERO L12", AXIS)]  # fmt: skip


def fig_neighbors() -> bool:
    """Local action consistency of a key (offline, held-out demos) vs the closed-loop success of a lookup with it."""
    nm = DATA.get("neighbor_metric_heldout", {}).get("correlation")
    if not nm or nm.get("fair", {}).get("n", 0) < 4:
        return False
    pts = nm["points"]
    fig, ax = plt.subplots(figsize=(W, 3.1), layout="constrained")
    fair = [(v["lac@8"], v["success"]) for v in pts.values() if v["fair"]]
    xs, ys = np.array(fair).T
    slope, icpt = np.polyfit(xs, ys, 1)
    gx = np.array(
        [min(v["lac@8"] for v in pts.values()) - 0.005, max(v["lac@8"] for v in pts.values()) + 0.005]
    )
    ax.plot(gx, slope * gx + icpt, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    for a, b, c in NEIGHBOR_ARROWS:
        if a in pts and b in pts:
            ax.annotate(
                "",
                (pts[b]["lac@8"], pts[b]["success"]),
                (pts[a]["lac@8"], pts[a]["success"]),
                arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.6, "shrinkA": 6, "shrinkB": 6},
                zorder=2,
            )
    for name, v in pts.items():
        colour = next(c for prefix, c in NEIGHBOR_STYLE if name.startswith(prefix))
        ax.scatter([v["lac@8"]], [v["success"]], s=64, facecolor=colour if v["fair"] else "white", edgecolor=colour,
                   linewidth=1.8, zorder=3)  # fmt: skip
        if name in NEIGHBOR_LABELS:
            label, offset = NEIGHBOR_LABELS[name]
            ax.annotate(label, (v["lac@8"], v["success"]), xytext=offset, textcoords="offset points",
                        ha="right" if offset[0] < 0 else "left", fontsize=FS_SMALL - 1, color=INK2)  # fmt: skip
    ax.text(0.98, 0.04, f"Spearman ρ = {nm['fair']['spearman_lac@8']:.2f}, filled (n = {nm['fair']['n']})\n"
            f"ρ = {nm['all']['spearman_lac@8']:.2f} with the hollow ones (n = {nm['all']['n']})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_SMALL, color=INK)  # fmt: skip
    ax.set_xlabel("local action consistency LAC@8 (held-out demos)")
    ax.set_ylabel("plain lookup success (%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=c, markeredgecolor=c, markersize=8, label=lab)
               for c, lab in ((HUMAN_DARK, "ImageNet"), (LIBERO_STAGE[0], "VLM before LIBERO"),
                              (LIBERO_STAGE[2], "fine-tuned (hollow: saw these demos)"), (KNN, "learnt key"))]  # fmt: skip
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncol=2, fontsize=FS_SMALL - 0.5)
    save(fig, "neighbors")
    return True


# ---- middle: LIBERO-plus ----------------------------------------------------------------------------------
PLUS_CATEGORIES = [("Camera Viewpoints", "Camera"), ("Robot Initial States", "Robot"), ("Language Instructions", "Language"),
                   ("Light Conditions", "Light"), ("Background Textures", "Background"), ("Sensor Noise", "Noise"),
                   ("Objects Layout", "Layout")]  # fmt: skip
PLUS_POLICIES = [  # LIBERO-plus policy spec, label, how to find its unperturbed success on the same-400 LIBERO episodes
    ("knn:outputs/knn/v2_39.json", "kNN, full method", ("ablation", None)),
    ("knn:outputs/vla/configs/plain_resnet18_l3.json", "kNN, plain (ResNet-18)", ("knn", "plain_resnet18_l3")),
    ("vla:pi05_libero", "π0.5 (LeRobot)", ("vla", "pi05_libero")),
    ("vla:xvla@lerobot/xvla-libero", "X-VLA", ("vla", "xvla@lerobot/xvla-libero")),
    ("vla:pi0_fast@lerobot/pi0fast-libero", "π0-FAST", ("vla", "pi0_fast@lerobot/pi0fast-libero")),
    ("vla:smolvla_libero", "SmolVLA", ("vla", "smolvla_libero")),
    ("vla:pi0_libero", "π0", ("vla", "pi0_libero")),
    ("vla:vla_jepa@lerobot/VLA-JEPA-LIBERO", "VLA-JEPA", ("vla", "vla_jepa@lerobot/VLA-JEPA-LIBERO")),
    ("vla:evo1_libero", "Evo-1", ("vla", "evo1_libero")),
    ("vla:molmoact2_libero", "MolmoAct2", ("vla", "molmoact2_libero")),
    ("vla:groot_libero", "GR00T N1.7", ("vla", "groot_libero")),
]  # fmt: skip
PLUS_RAMP = [
    "#f5f6f8",
    "#dcdee2",
    "#b9bcc2",
    "#8e9299",
    "#5f636b",
    "#353840",
]  # neutral sequential, light -> dark


def clean_success(how: tuple[str, str | None]) -> float | None:
    kind, key = how
    if kind == "ablation":
        return DATA["ablation_base"]
    if kind == "knn":
        run = DATA.get("knn_with_vla_embedders", {}).get(key)
        return 100 * run["success"] if run else None
    return best_vla(key)


def fig_libero_plus() -> bool:
    """Success per LIBERO-plus perturbation category (same 420 variants for every policy) next to clean LIBERO."""
    plus = DATA.get("libero_plus", {})
    rows = [p for p in PLUS_POLICIES if p[0] in plus]
    if len(rows) < 2:
        return False
    cols = ["LIBERO", *[short for _, short in PLUS_CATEGORIES], "all"]
    grid = np.full((len(rows), len(cols)), np.nan)
    for i, (spec, _label, how) in enumerate(rows):
        r = plus[spec]
        clean = clean_success(how)
        grid[i, 0] = clean if clean is not None else np.nan
        for j, (cat, _) in enumerate(PLUS_CATEGORIES):
            grid[i, 1 + j] = 100 * r.get(f"category/{cat}", np.nan)
        grid[i, -1] = 100 * r["all"]
    fig, ax = plt.subplots(figsize=(W, 0.85 + 0.29 * len(rows)), layout="constrained")
    cmap = mpl.colors.LinearSegmentedColormap.from_list("plus", PLUS_RAMP)
    xs = [j + (0.25 if j == len(cols) - 1 else 0) + (0 if j == 0 else 0.15) for j in range(len(cols))]
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
            ax.text(x, i, f"{v:.0f}", ha="center", va="center", fontsize=FS_SMALL,
                    color="white" if v >= 60 else INK, weight="bold" if j == len(cols) - 1 else "normal")  # fmt: skip
    ax.set_xticks(xs, cols, rotation=35, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.tick_params(length=0, labelsize=FS_SMALL, labelcolor=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    save(fig, "libero_plus")
    return True


# ---- numbers.tex ------------------------------------------------------------------------------------------
def numbers(flags: dict[str, bool]) -> None:
    k = DATA["knn_official"]
    base = {b["name"]: b for b in DATA["baselines"]}
    m = DATA["dit_memorization"]
    ab = DATA["ablations"]
    pend = r"\pending"
    macros = {
        "KnnAvg": f1(k["avg"]),
        "KnnSpatial": f1(k["suites"][0]), "KnnObject": f1(k["suites"][1]),
        "KnnGoal": f1(k["suites"][2]), "KnnLong": f1(k["suites"][3]),
        "KnnEpisodes": f"{k['n']:,}",
        "DitDocsAvg": f1(base["Multi-task DiT (diffusion, LeRobot)"]["avg"]),
        "PiFiveAvg": f1(base["π0.5 (LeRobot)"]["avg"]),
        "OpenVlaAvg": f1(base["OpenVLA 7B (fine-tuned)"]["avg"]),
        "AblBase": f1(DATA["ablation_base"]),
        "AblNoLang": f1(ab["no_language"]["success"]),
        "AblNoLangGoal": f"{ab['no_language']['suites'][2]:.0f}",
        "AblPixels": f1(ab["encoder_pixels"]["success"]),
        "AblKOne": f1(ab["k1"]["success"]),
        "AblDino": f1(ab["encoder_dinov2"]["success"]),
        "AblOpenLoop": f1(ab["no_ensemble"]["success"]),
        "DemosTen": f1(DATA["demo_scaling"]["10"]),
        "DitOurs": f1(DATA["dit_ours"]["avg"]) if "dit_ours" in DATA else pend,
        "DitTrainMed": f"{np.median(m['dit_to_train']):.4f}",
        "HumanMed": f"{np.median(m['human_to_other_demos']):.4f}",
        "DitKnnMed": f"{np.median(m['dit_to_knn']):.4f}",
        "KnnSpreadMed": f"{np.median(m['knn_spread']):.4f}",
        "GripAgree": f"{100 * m['gripper_agreement']:.0f}",
        "DitChunks": str(m["n_chunks"]),
    }  # fmt: skip
    diag = DATA.get("failure_diag", [])
    ok = [e["peak_distance"] for e in diag if e["success"]]
    bad = [e["peak_distance"] for e in diag if not e["success"]]
    macros["DiagN"] = str(len(diag))
    macros["DiagOkMax"] = f"{max(ok):.1f}" if ok else pend
    macros["DiagBadMin"] = f"{min(bad):.1f}" if bad else pend
    macros["DiagBad"] = str(len(bad))
    for bname, tag in (("Multi-task DiT (diffusion, LeRobot)", "Dit"), ("π0.5 (LeRobot)", "PiFive")):
        for suite, v in zip(SUITE_NAMES, base[bname]["suites"], strict=True):
            macros[f"{tag}{suite}"] = f1(v)
    rob = {r["name"]: r["overall"] for r in k.get("robustness", [])}
    macros["KnnSeedTwo"] = f1(rob["evaluation seed 2000"]) if "evaluation seed 2000" in rob else pend
    hires = "360×360 renders (lerobot-eval default)"
    macros["KnnHiRes"] = f1(rob[hires]) if hires in rob else pend
    runs = DATA.get("knn_with_vla_embedders", {})
    for stem, name in [("plain_smolvla_libero_img", "KnnSmolLibero"), ("plain_smolvla_base_img", "KnnSmolBase"),
                       ("plain_smolvla_libero_grid2", "KnnSmolLiberoGrid"), ("plain_smolvla_base_grid2", "KnnSmolBaseGrid"),
                       ("plain_resnet18_l3", "KnnPlainResnet"), ("plain_resnet18_l3_mean", "KnnPlainResnetMean"),
                       ("plain_dinov2", "KnnPlainDino"), ("plain_pi05_libero_grid2", "KnnPiLibero"),
                       ("plain_pi05_base_grid2", "KnnPiBase"), ("plain_pi05_libero_grid2_av1", "KnnPiLiberoAvOne")]:  # fmt: skip
        macros[name] = f1(100 * runs[stem]["success"]) if stem in runs else pend
    for name, macro in (("smolvla_libero", "VlaSmol"), ("pi05_libero", "VlaPi")):
        best = best_vla(name)
        macros[macro] = f1(best) if best is not None else pend
    for stem, macro in (("smolvla_libero", "SwapSmol"), ("pi05_libero", "SwapPi")):
        best = best_vla(stem)
        for pool, suffix in (("grid2", ""), ("img", "Mean")):
            run = runs.get(f"plain_{stem}_{pool}")
            macros[macro + suffix] = f"{100 * 100 * run['success'] / best:.0f}" if run and best else pend
    heads = DATA.get("head_vs_knn", {})
    for vla, tag, before in (("smolvla_libero", "Smol", "smolvla_base"), ("pi05_libero", "Pi", "pi05_base")):
        h = heads.get(vla, {})
        sp = h.get("spaces", {})
        own, bsp = sp.get(f"vla:{vla}/{VLM_POOL}", {}), sp.get(f"vla:{before}/{VLM_POOL}", {})
        rn = sp.get("resnet18_l3", {})
        macros[f"Head{tag}Own"] = f"{own['8']['vla_r2']:.2f}" if "8" in own else pend
        macros[f"Head{tag}OwnOne"] = f"{own['1']['vla_r2']:.2f}" if "1" in own else pend
        macros[f"Head{tag}Before"] = f"{bsp['8']['vla_r2']:.2f}" if "8" in bsp else pend
        macros[f"Head{tag}BeforeOne"] = f"{bsp['1']['vla_r2']:.2f}" if "1" in bsp else pend
        macros[f"Head{tag}Resnet"] = f"{rn['8']['vla_r2']:.2f}" if "8" in rn else pend
        macros[f"Head{tag}ResnetOne"] = f"{rn['1']['vla_r2']:.2f}" if "1" in rn else pend
        nf, nn = h.get("noise_floor"), h.get("nearest_training_chunk")
        macros[f"Head{tag}Self"] = f"{nf['vla_r2']:.2f}" if nf else pend
        macros[f"Head{tag}NnVla"] = f"{nn['vla_median']:.4f}" if nn else pend
        macros[f"Head{tag}NnDemo"] = f"{nn['demo_median']:.4f}" if nn else pend
    pi = vla_run("pi05_libero")
    runs_pi = runs.get("plain_pi05_libero_grid2")
    for i, (sname, name) in enumerate(zip(SUITES, SUITE_NAMES, strict=True)):
        macros[f"VlaPi{name}"] = f1(100 * pi[sname]) if pi else pend
        macros[f"KnnPiLibero{name}"] = f1(100 * runs_pi["suites"][i]) if runs_pi else pend
    for stem, macro in (("plain_pi05_libero_grid2_z", "KnnPiLiberoZ"), ("plain_pi05_base_grid2_z", "KnnPiBaseZ"),
                        ("plain_pi05_libero_grid2@L15_z", "KnnPiLiberoLf"), ("plain_pi05_base_grid2@L15_z", "KnnPiBaseLf"),
                        ("plain_pi05_libero_grid2@L12_z", "KnnPiLiberoLt"), ("plain_pi05_base_grid2@L12_z", "KnnPiBaseLt")):  # fmt: skip
        macros[macro] = f1(100 * runs[stem]["success"]) if stem in runs else pend
    for stem, macro in (
        ("plain_pi05_libero_grid2", "BestLookupPi"),
        ("plain_smolvla_libero_grid2", "BestLookupSmol"),
    ):
        best = best_lookup(stem)
        macros[macro] = f1(best) if best is not None else pend
    smol = vla_run("smolvla_libero")
    for s, name in zip(SUITES, SUITE_NAMES, strict=True):
        macros[f"VlaSmol{name}"] = f1(100 * smol[s]) if smol else pend
    for stem, macro in (("plain_pi05_base_grid2@L12+L15+last_pca", "KnnPiBaseMulti"),
                        ("plain_pi05_libero_grid2@L12+L15+last_pca", "KnnPiLiberoMulti"),
                        ("plain_pi05_base_grid2@L12_pca", "KnnPiBasePca"), ("plain_pi05_libero_grid2@L12_pca", "KnnPiLiberoPca"),
                        ("plain_resnet18_l3_learned", "KnnLearnt")):  # fmt: skip
        macros[macro] = f1(100 * runs[stem]["success"]) if stem in runs else pend
    full_learnt = [100 * r["success"] for r in DATA.get("knn_learned_key", [])]
    macros["KnnFullLearnt"] = f1(full_learnt[-1]) if full_learnt else pend
    nm = DATA.get("neighbor_metric_heldout", {})
    corr, spaces = nm.get("correlation", {}), nm.get("spaces", {})
    for subset, tag in (("fair", "Fair"), ("all", "All")):
        c = corr.get(subset, {})
        macros[f"Nm{tag}Rho"] = f"{c['spearman_lac@8']:.2f}" if "spearman_lac@8" in c else pend
        macros[f"Nm{tag}GlobalRho"] = (
            f"{c['spearman_global_rho']:.2f}" if "spearman_global_rho" in c else pend
        )
        macros[f"Nm{tag}N"] = str(c.get("n", 0))
    for space, macro in (("SmolVLA base grid", "NmSmolBase"), ("SmolVLA LIBERO grid", "NmSmolLibero"),
                         ("π0.5 base L12", "NmPiBase"), ("π0.5 LIBERO L12", "NmPiLibero"),
                         ("ResNet-18 grid", "NmResnet"), ("ResNet-18 grid (learnt key)", "NmLearnt")):  # fmt: skip
        macros[macro] = f"{spaces[space]['lac@8']:.2f}" if space in spaces else pend
    lk = DATA.get("learned_key", {})
    macros["LkBefore"] = f"{lk['before']['lac@8']:.2f}" if lk else pend
    macros["LkAfter"] = f"{lk['after']['lac@8']:.2f}" if lk else pend
    plus = DATA.get("libero_plus", {})
    sample = [p for p in PLUS_POLICIES if p[0] in plus]
    macros["PlusN"] = str(plus[sample[0][0]]["n_episodes"]) if sample else pend
    macros["PlusPerCat"] = str(plus[sample[0][0]]["n_episodes"] // len(PLUS_CATEGORIES)) if sample else pend
    # per policy: LIBERO-plus success overall and per perturbation, its clean LIBERO success and the drop
    for spec, macro in (("knn:outputs/knn/v2_39.json", "PlusKnn"), ("knn:outputs/vla/configs/plain_resnet18_l3.json", "PlusPlain"),
                        ("knn:outputs/vla/configs/plain_pi05_base_grid2@L12_z.json", "PlusPlainPi"),
                        ("vla:pi05_libero", "PlusPi"),
                        ("vla:xvla@lerobot/xvla-libero", "PlusXvla"), ("vla:smolvla_libero", "PlusSmol"),
                        ("vla:pi0_fast@lerobot/pi0fast-libero", "PlusFast"), ("vla:pi0_libero", "PlusPiZero")):  # fmt: skip
        r = plus.get(spec)
        macros[macro] = f1(100 * r["all"]) if r else pend
        for cat, short in PLUS_CATEGORIES:
            macros[macro + short] = (
                f"{100 * r[f'category/{cat}']:.0f}" if r and f"category/{cat}" in r else pend
            )
        how = next((h for sp, _, h in PLUS_POLICIES if sp == spec), None)
        clean = clean_success(how) if how else None
        macros[macro + "Clean"] = f1(clean) if clean is not None else pend
        macros[macro + "Drop"] = (
            f"{Decimal(f1(clean)) - Decimal(f1(100 * r['all']))}" if r and clean is not None else pend
        )
        ctrl = DATA.get("libero_plus_control", {}).get(spec)
        macros[macro + "Ctrl"] = f"{100 * ctrl:.0f}" if ctrl is not None else pend
    # how much of each VLA's success a training-free ImageNet lookup already has (same 400 episodes)
    ov = DATA.get("episode_overlap", {})
    wins = {
        label: ov["vlas"][k]["plain"] for k, label in WIN_VLAS if "plain" in ov.get("vlas", {}).get(k, {})
    }
    shared = [100 * c["both"] / (c["both"] + c["vla_only"]) for c in wins.values()]
    macros["SharedMin"] = f"{min(shared):.0f}" if shared else pend
    macros["SharedMax"] = f"{max(shared):.0f}" if shared else pend
    for label, tag in (("π0.5 (LeRobot)", "Pi"), ("SmolVLA", "Smol"), ("π0", "PiZero"), ("X-VLA", "Xvla"),
                       ("π0-FAST", "Fast")):  # fmt: skip
        c = wins.get(label)
        macros[f"Wins{tag}"] = str(c["vla_only"]) if c else pend
        macros[f"Loses{tag}"] = str(c["lookup_only"]) if c else pend
        macros[f"Rescue{tag}"] = f"{100 * c['vla_only'] / (c['vla_only'] + c['neither']):.0f}" if c else pend
    for name, tag in (("plain", "Plain"), ("full", "Full")):
        per = ov.get("lookups", {}).get(name, {}).get("per_task", {})
        macros[f"TasksNinety{tag}"] = str(sum(v >= 0.9 for v in per.values())) if per else pend
        macros[f"TasksPerfect{tag}"] = str(sum(v >= 1.0 for v in per.values())) if per else pend
    macros["NTasks"] = str(len(ov.get("lookups", {}).get("plain", {}).get("per_task", {})))
    for vla, stem, tag in (("smolvla_libero", "plain_smolvla_libero_grid2", "Smol"),
                           ("pi05_libero", "plain_pi05_libero_grid2@L12_z", "Pi"),
                           ("pi0_libero", "plain_pi0_libero_grid2@L12_z", "PiZero")):  # fmt: skip
        best = best_vla(vla)
        gain = (
            Decimal(f1(best)) - Decimal(f1(100 * runs[stem]["success"]))
            if best is not None and stem in runs
            else None
        )
        macros[f"Gain{tag}"] = f"{'+' if gain > 0 else '$-$'}{abs(gain)}" if gain is not None else pend
    for vla, stem, _label, _marker in DIAG_POINTS:  # the same gap on LIBERO-plus (whole points, as the figure)
        tag = {"SmolVLA": "Smol", "π0.5": "Pi", "π0": "PiZero", "X-VLA": "Xvla", "π0-FAST": "Fast",
               "VLA-JEPA": "Jepa", "GR00T N1.7": "Groot"}[_label]  # fmt: skip
        a = DATA.get("libero_plus", {}).get(f"vla:{vla}")
        b = DATA.get("libero_plus", {}).get(f"knn:outputs/vla/configs/{first_run(stem)}.json")
        gain = Decimal(f1(100 * a["all"])) - Decimal(f1(100 * b["all"])) if a and b else None
        macros[f"PlusGain{tag}"] = f"{'+' if gain > 0 else '$-$'}{abs(gain):.0f}" if gain is not None else pend
    for vla, macro in (("xvla@lerobot/xvla-libero", "VlaX"), ("pi0_fast@lerobot/pi0fast-libero", "VlaFast"),
                       ("pi0_libero", "VlaPiZero")):  # fmt: skip
        best = best_vla(vla)
        macros[macro] = f1(best) if best is not None else pend
    for stem, macro in (("plain_pi0_libero_grid2@L12_z", "KnnPiZeroLt"), ("plain_pi0_base_grid2@L12_z", "KnnPiZeroBaseLt")):  # fmt: skip
        macros[macro] = f1(100 * runs[stem]["success"]) if stem in runs else pend
    lines = ["% generated by scripts/make_figures.py -- do not edit"]
    for flag, on in flags.items():
        lines += [f"\\newif\\if{flag}", f"\\{flag}{'true' if on else 'false'}"]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in macros.items()]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_leaderboard()
    have_wins = fig_episode_wins()
    fig_dit()
    fig_failures()
    fig_ablations()
    have_diag = fig_diagnostic()
    have_keys = fig_vlm_keys()
    have_neighbors = fig_neighbors()
    have_plus = fig_libero_plus()
    have_pi = "pi05_libero" in DATA.get("head_vs_knn", {}) or best_vla("pi05_libero") is not None
    numbers(
        {
            "havewins": have_wins,
            "havepi": have_pi,
            "havediag": have_diag,
            "havekeys": have_keys,
            "haveneighbors": have_neighbors,
            "haveplus": have_plus,
        }
    )
    for stale, have in (("neighbors", have_neighbors), ("libero_plus", have_plus)):
        if not have:
            (FIG / f"{stale}.pdf").unlink(missing_ok=True)
    for old in ("same_episodes", "head_r2", "scaling"):
        (FIG / f"{old}.pdf").unlink(missing_ok=True)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
