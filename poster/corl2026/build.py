# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "segno"]
# ///
"""Build the CoRL 2026 workshop poster "How Much of Your VLA is kNN" (16:9, for an online poster session).

    make                      # build/poster.html -> poster.pdf -> preview.png
    uv run --script build.py  # only build/poster.html and the generated figures

`template.html` holds the layout and the text. This script fills its ``{{PLACEHOLDER}}`` slots with inline SVG charts and
writes the generated images (t-SNE maps, LaTeX equations, QR codes) next to the page in ``build/``.

Where the numbers come from:
  * Tables 1 and 2 below are copied from the camera-ready paper.
  * LAC values, the LAC-vs-success scatter and the t-SNE maps are read from ``../data/*.json``, which
    ``../scripts/prepare_data.py`` collects from the experiment logs.

Chart geometry is in millimetres: one SVG user unit is 1 mm on the page, times the chart's ``scale``.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import segno
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
DATA = json.loads((HERE / "../data/poster_data.json").read_text())
MAPS = json.loads((HERE / "../data/embedding_maps.json").read_text())

PROJECT_URL = "https://sereact.github.io/lerobot_knn/"
CODE_URL = "https://github.com/sereact/lerobot_knn"

# Colour roles, shared with the project page: orange = k-NN (ours), blue = VLAs, grey = generic encoders.
KNN, KNN_SOFT, KNN_TINT = "#eb6834", "#f6b79b", "#fdf1ea"
VLA, VLA_SOFT = "#2a78d6", "#a9c9ef"
INK, INK2, MUTED, GRID, GREY, STRIPE = "#0f1720", "#47505c", "#7a828d", "#e3e6ea", "#8b9099", "#f5f6f8"
PROGRESS_RAMP = ["#c9dcf4", "#5a9be3", "#1c5cab", "#0b2a55"]  # episode start -> end


# ---- labels ---------------------------------------------------------------------------------------------------------
# Policy names are written as plain text; "_" starts a numeric subscript ("π_0.5", "π_0-FAST").
_SUBSCRIPT = re.compile(r"_([\d.]+)")


def svg_label(name: str) -> str:
    name = name.replace("k-NN", '<tspan font-style="italic">k</tspan>-NN')
    head, underscore, rest = name.partition("_")
    if not underscore:
        return name
    sub = re.match(r"[\d.]+", rest).group()
    return f'{head}<tspan dy="1.6" font-size="72%">{sub}</tspan><tspan dy="-1.6">{rest[len(sub) :]}</tspan>'


def html_label(name: str) -> str:
    return _SUBSCRIPT.sub(r"<sub>\1</sub>", name)


# ---- a minimal SVG builder ------------------------------------------------------------------------------------------
class Svg:
    """Collects SVG markup for one chart. ``uid`` keeps the marker ids of different charts apart in the page."""

    def __init__(self, uid: str, width: float, height: float, scale: float = 1.0) -> None:
        self.uid, self.width, self.height, self.scale = uid, width, height, scale
        self.parts: list[str] = []

    def line(self, x1, y1, x2, y2, stroke, width, dash: str = "", cap: str = "", arrow: str = "") -> None:
        extra = (f' stroke-dasharray="{dash}"' if dash else "") + (f' stroke-linecap="{cap}"' if cap else "")
        if arrow:  # `arrow` is the head's colour
            extra += f' marker-end="url(#{self._arrowhead(arrow)})"'
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
                          f'stroke-width="{width}"{extra}/>')  # fmt: skip

    def circle(self, cx, cy, r, fill, stroke="#fff", stroke_width=0.8) -> None:
        self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{stroke_width}"/>')  # fmt: skip

    def rect(self, x, y, width, height, fill, radius=0.0) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{radius}" '
                          f'fill="{fill}"/>')  # fmt: skip

    def diamond(self, cx, cy, half, fill) -> None:
        self.parts.append(f'<path d="M{cx:.2f} {cy - half:.2f} l{half} {half} l{-half} {half} l{-half} {-half}z" '
                          f'fill="{fill}" stroke="#fff" stroke-width="0.8"/>')  # fmt: skip

    def text(self, x, y, content, size, anchor="start", weight=400, fill=INK) -> None:
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" fill="{fill}">{content}</text>')  # fmt: skip

    def _arrowhead(self, colour: str) -> str:
        marker_id = f"{self.uid}-arrow-{colour.lstrip('#')}"
        marker = (f'<defs><marker id="{marker_id}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" '
                  f'markerHeight="5" orient="auto"><path d="M0 1 L9 5 L0 9z" fill="{colour}"/></marker></defs>')  # fmt: skip
        if marker not in self.parts:
            self.parts.append(marker)
        return marker_id

    def render(self) -> str:
        w, h, k = self.width, self.height, self.scale
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * k:.1f}mm" height="{h * k:.1f}mm" '
                f'viewBox="0 0 {w} {h}" font-family="Inter" fill="{INK}">{"".join(self.parts)}</svg>')  # fmt: skip


# ---- Result 1: LIBERO vs LIBERO-Plus (paper, Table 1) ---------------------------------------------------------------
# (policy, LIBERO, LIBERO-Plus, drop in %pt computed before rounding, is k-NN), sorted by LIBERO-Plus
BENCHMARK = [
    ("π_0.5", 97.7, 85.5, -12.2, False),
    ("MolmoAct2", 97.5, 82.6, -14.9, False),
    ("k-NN (multimodal)", 91.7, 82.4, -9.3, True),
    ("k-NN (LAC key)", 93.0, 81.7, -11.3, True),
    ("GR00T N1.7", 94.9, 80.2, -14.6, False),
    ("k-NN (ResNet)", 88.7, 75.5, -13.2, True),
    ("VLA-JEPA", 97.2, 74.0, -23.1, False),
    ("Evo-1", 95.4, 71.4, -24.0, False),
    ("X-VLA", 94.5, 64.5, -30.0, False),
    ("π_0", 71.9, 42.9, -29.0, False),
    ("SmolVLA", 82.2, 27.9, -54.3, False),
    ("π_0-FAST", 90.6, 25.7, -64.8, False),
]


def chart_benchmark(width: float, scale: float, row_height: float) -> str:
    """One row per policy: LIBERO-Plus (filled dot) to LIBERO (ring), with the drop in the right-hand column."""
    # the policy names sit left of x_min, the drop column right of x_max
    top, bottom, x_min, x_max = 14.0, 13.0, 70.0, width - 45.0
    height = top + row_height * len(BENCHMARK) + bottom
    svg = Svg("benchmark", width, height, scale)

    def x(success: float) -> float:  # the axis runs from 10 to 100 %
        return x_min + (success - 10) / 90 * (x_max - x_min)

    svg.circle(x_min + 3, 5, 3.4, INK2, stroke_width=0)
    svg.text(x_min + 9, 7.6, "LIBERO-Plus", 7, fill=INK2)
    svg.circle(x_min + 66, 5, 2.9, "#fff", INK2, 1.4)
    svg.text(x_min + 72, 7.6, "LIBERO", 7, fill=INK2)
    svg.text(width, 7.6, "Δ %pt", 7, "end", 600, INK2)
    for tick in range(20, 101, 20):
        svg.line(x(tick), top, x(tick), height - 12, GRID, 0.5)
        svg.text(x(tick), height - 4.5, str(tick), 6.6, "middle", fill=MUTED)
    svg.text(x_min - 4, height - 4.5, "Success rate (%)", 6.6, "end", fill=MUTED)

    for i, (policy, libero, plus, drop, is_knn) in enumerate(BENCHMARK):
        y = top + row_height * (i + 0.5)
        colour, soft, weight = (KNN, KNN_SOFT, 700) if is_knn else (VLA, VLA_SOFT, 500)
        if is_knn:  # highlight our rows
            svg.rect(0, y - row_height / 2 + 0.8, width, row_height - 1.6, KNN_TINT, 3)
            svg.circle(5, y, 2.4, KNN, stroke_width=0)
        svg.text(10.5 if is_knn else 4, y + 2.6, svg_label(policy), 7.6, weight=weight)
        svg.line(x(plus), y, x(libero), y, soft, 3.2, cap="round")
        svg.circle(x(plus), y, 4.1, colour)
        svg.circle(x(libero), y, 3.4, "#fff", colour, 1.6)
        svg.text(x(plus) - 6.5, y + 2.5, f"{plus:.1f}", 6.8, "end", weight)
        svg.text(x(libero) + 6.2, y + 2.5, f"{libero:.1f}", 6.8, fill=INK2)
        svg.text(width - 3, y + 2.6, f"−{abs(drop):.1f}", 7.4, "end", weight)
    return svg.render()


# ---- Result 2: k-NN on each VLA's backbone vs the full VLA (paper, Table 2) -----------------------------------------
# (VLA, {benchmark: (k-NN before fine-tuning, k-NN after, full VLA)}); GR00T's VLM is frozen, so before = after
BACKBONES = [
    ("π_0.5", {"LIBERO": (85.4, 81.8, 97.7), "LIBERO-Plus": (60.5, 65.7, 85.5)}),
    ("MolmoAct2", {"LIBERO": (78.6, 91.4, 97.5), "LIBERO-Plus": (52.6, 81.0, 82.6)}),
    ("GR00T N1.7*", {"LIBERO": (87.6, 87.6, 94.9), "LIBERO-Plus": (66.4, 66.4, 80.2)}),
    ("VLA-JEPA", {"LIBERO": (84.0, 83.0, 97.2), "LIBERO-Plus": (49.3, 65.7, 74.0)}),
    ("Evo-1", {"LIBERO": (62.1, 87.1, 95.4), "LIBERO-Plus": (35.5, 72.6, 71.4)}),
    ("X-VLA", {"LIBERO": (83.3, 88.6, 94.5), "LIBERO-Plus": (62.4, 78.1, 64.5)}),
    ("π_0", {"LIBERO": (86.9, 17.9, 71.9), "LIBERO-Plus": (56.7, 2.4, 42.9)}),
    ("SmolVLA", {"LIBERO": (76.3, 86.5, 82.2), "LIBERO-Plus": (45.0, 69.3, 27.9)}),
    ("π_0-FAST", {"LIBERO": (86.6, 95.0, 90.6), "LIBERO-Plus": (58.3, 74.0, 25.7)}),
]


def chart_backbones(width: float, scale: float, row_height: float, panels=("LIBERO-Plus", "LIBERO")) -> str:
    """Per benchmark: ring (k-NN before fine-tuning) -> dot (after), a diamond for the full VLA, and k-NN minus VLA."""
    top, bottom, names, diff_column, gap = 19.0, 13.0, 58.0, 27.0, 10.0
    plot = (width - names - gap * (len(panels) - 1) - diff_column * len(panels)) / len(panels)
    height = top + row_height * len(BACKBONES) + bottom
    svg = Svg("backbones", width, height, scale)

    for i, (vla, _) in enumerate(BACKBONES):
        y = top + row_height * (i + 0.5)
        if i % 2 == 0:
            svg.rect(0, y - row_height / 2, width, row_height, STRIPE, 3)
        svg.text(4, y + 2.6, svg_label(vla), 7.6, weight=500)

    for p, benchmark in enumerate(panels):
        left = names + p * (plot + diff_column + gap)
        right = left + plot + diff_column

        def x(success: float, left: float = left) -> float:
            return left + 5 + success / 100 * (plot - 10)

        svg.text(left + plot / 2, 8, benchmark, 8.4, "middle", 700)
        svg.text(right - 1, 8, "vs VLA", 6.6, "end", 600, INK2)
        svg.line(left, 12.5, right, 12.5, INK, 0.6)
        for tick in range(0, 101, 25):
            svg.line(x(tick), top, x(tick), height - 12, GRID, 0.5)
            svg.text(x(tick), height - 4.5, str(tick), 6.4, "middle", fill=MUTED)

        for i, (_, results) in enumerate(BACKBONES):
            before, after, full_vla = results[benchmark]
            y = top + row_height * (i + 0.5)
            if before != after:
                direction = 1 if after > before else -1
                svg.line(x(before) + 3.2 * direction, y, x(after) - 4.2 * direction, y, KNN, 1.5, arrow=KNN)
                svg.circle(x(before), y, 2.9, "#fff", KNN, 1.4)
            svg.circle(x(after), y, 3.7, KNN)
            svg.diamond(x(full_vla), y, 4.2, VLA)
            diff = after - full_vla
            if diff > 0:  # the k-NN on the backbone beats the full VLA
                svg.rect(right - 24, y - 5, 24, 10, KNN, 5)
                svg.text(right - 12, y + 2.5, f"+{diff:.1f}", 6.8, "middle", 700, "#fff")
            else:
                svg.text(right - 3, y + 2.5, f"−{abs(diff):.1f}", 6.8, "end", fill=MUTED)
    return svg.render()


# ---- Result 5: LAC@8 vs closed-loop success of the image-only k-NN with each key ------------------------------------
# keys in ../data/poster_data.json: (VLM before LIBERO fine-tuning, after)
VLM_KEYS = [
    ("SmolVLA base grid", "SmolVLA LIBERO grid"),
    ("π0.5 base L12", "π0.5 LIBERO L12"),
    ("X-VLA base grid2@L8", "X-VLA LIBERO grid2@L8"),
    ("π0-FAST base grid2@L12", "π0-FAST LIBERO grid2@L12"),
    ("VLA-JEPA base grid2@L7", "VLA-JEPA LIBERO grid2@L7"),
    ("Evo-1 base grid2@L10", "Evo-1 LIBERO grid2@L10"),
    ("MolmoAct2 base grid2@L27", "MolmoAct2 LIBERO grid2@L27"),
    ("π0 base L12", "π0 LIBERO L12"),
]
FROZEN_VLM_KEY = "GR00T N1.7 frozen VLM grid2@L4"
RESNET_KEY, LAC_KEY, DINO_KEY = "ResNet-18 grid", "ResNet-18 grid (learnt key)", "DINOv2-S"
EVO1_KEYS, PI0_FINETUNED_KEY = VLM_KEYS[5], "π0 LIBERO L12"


def chart_lac(width: float, height: float, scale: float) -> str:
    """Scatter of offline LAC@8 against closed-loop success; the fine-tuned pi0 key is off scale and only noted."""
    points = DATA["neighbor_metric"]["correlation"]["points"]
    left, right, bottom, top = 22.0, width - 4, height - 16, 19.0
    lac_min, lac_max, success_min, success_max = 0.735, 0.89, 58.0, 100.0
    svg = Svg("lac", width, height, scale)

    def xy(key: str) -> tuple[float, float]:
        return x(points[key]["lac@8"]), y(points[key]["success"])

    def x(lac: float) -> float:
        return left + (lac - lac_min) / (lac_max - lac_min) * (right - left)

    def y(success: float) -> float:
        return bottom + (success - success_min) / (success_max - success_min) * (top - bottom)

    def arrow(
        start: str, end: str, start_radius: float, end_radius: float, colour: str, stroke: float
    ) -> None:
        """A straight arrow between two keys that stops short of both markers along the line itself."""
        (ax, ay), (bx, by) = xy(start), xy(end)
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        ux, uy = (bx - ax) / length, (by - ay) / length
        svg.line(ax + ux * start_radius, ay + uy * start_radius, bx - ux * end_radius, by - uy * end_radius,
                 colour, stroke, arrow=colour)  # fmt: skip

    for tick in (60, 70, 80, 90, 100):
        svg.line(left, y(tick), right, y(tick), GRID, 0.5)
        svg.text(left - 3, y(tick) + 2.2, str(tick), 6.2, "end", fill=MUTED)
    for tick in (0.76, 0.80, 0.84, 0.88):
        svg.text(x(tick), bottom + 8, f"{tick:.2f}", 6.2, "middle", fill=MUTED)
    svg.text(right, height - 0.8, "LAC@8 (offline) →", 6.6, "end", 600, INK2)
    svg.text(0, 6.5, svg_label("k-NN success on LIBERO (%)"), 6.6, weight=600, fill=INK2)

    # least-squares line over the "fair" keys, i.e. those that never saw the held-out demonstrations
    fair = [(p["lac@8"], p["success"]) for p in points.values() if p["fair"]]
    mean_lac, mean_success = (sum(values) / len(fair) for values in zip(*fair, strict=True))
    slope = sum((a - mean_lac) * (b - mean_success) for a, b in fair) / sum(
        (a - mean_lac) ** 2 for a, _ in fair
    )
    fit = [(lac, mean_success + slope * (lac - mean_lac)) for lac in (0.75, 0.835)]
    svg.line(x(fit[0][0]), y(fit[0][1]), x(fit[1][0]), y(fit[1][1]), GREY, 0.9, dash="2.6 2")

    arrow(*EVO1_KEYS, 3.4, 4.2, KNN, 1.1)
    svg.circle(*xy(DINO_KEY), 3.1, GREY, stroke_width=1.2)
    svg.circle(*xy(FROZEN_VLM_KEY), 3.1, "#fff", KNN, 1.2)
    for before, after in VLM_KEYS:
        svg.circle(*xy(before), 2.7, "#fff", KNN, 1.2)
        if points[after]["lac@8"] >= lac_min:
            svg.circle(*xy(after), 3.1, KNN, stroke_width=1.2)
    arrow(RESNET_KEY, LAC_KEY, 4.2, 5.0, INK, 1.0)
    svg.circle(*xy(RESNET_KEY), 3.4, GREY, stroke_width=1.2)
    svg.circle(*xy(LAC_KEY), 3.8, KNN, INK, 1.2)

    resnet, lac_key, dino, evo1 = xy(RESNET_KEY), xy(LAC_KEY), xy(DINO_KEY), xy(EVO1_KEYS[1])
    svg.text(resnet[0] - 9, resnet[1] - 5, "ResNet-18", 6.4, "end", 600)
    svg.text(lac_key[0], lac_key[1] - 6.5, "LAC key", 6.6, "middle", 700)
    svg.text(dino[0] - 5, dino[1] + 2.2, "DINOv2", 6.2, "end", fill=INK2)
    svg.text(evo1[0] - 5, evo1[1] - 4, "Evo-1", 6.4, "end", 600)
    pi0 = points[PI0_FINETUNED_KEY]
    note = f"← π_0 fine-tuned: LAC {pi0['lac@8']:.2f}, {pi0['success']:.1f} % (off scale)"
    svg.text(right, bottom - 3, svg_label(note), 6.2, "end", 500, INK2)
    return svg.render()


# ---- Result 3: t-SNE maps of the retrieval keys, with LAC@8 under each ----------------------------------------------
# (column title, key before fine-tuning, key after); the first column is the ResNet-18 before / after LAC training
TSNE_COLUMNS = [
    ("ResNet-18", RESNET_KEY, LAC_KEY),
    ("π_0.5", "π0.5 base L12", "π0.5 LIBERO L12"),
    ("Evo-1", "Evo-1 base grid2@L10", "Evo-1 LIBERO grid2@L10"),
    ("MolmoAct2", "MolmoAct2 base grid2@L27", "MolmoAct2 LIBERO grid2@L27"),
    ("π_0", "π0 base L12", "π0 LIBERO L12"),
]


def progress_colour(progress: float) -> tuple[int, int, int]:
    stops = [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in PROGRESS_RAMP]
    position = min(max(progress, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(position), len(stops) - 2)
    return tuple(round(a + (position - i) * (b - a)) for a, b in zip(stops[i], stops[i + 1], strict=True))


def render_tsne(key: str, path: Path, size: int = 700, dot_radius: float = 3.5, supersample: int = 2) -> None:
    """Draw one t-SNE map as a square PNG, points coloured by episode progress and plotted in a fixed random order."""
    points = MAPS["spaces"][key]["tsne"]
    xs, ys = zip(*points, strict=True)
    centre_x, centre_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.04
    canvas, radius = size * supersample, dot_radius * supersample
    image = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(image)
    order = list(range(len(points)))
    random.Random(0).shuffle(order)  # so that no phase of the episode is systematically drawn on top
    for i in order:
        px = (xs[i] - centre_x) / span * canvas + canvas / 2
        py = canvas / 2 - (ys[i] - centre_y) / span * canvas
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius), fill=progress_colour(MAPS["progress"][i])
        )
    image.resize((size, size), Image.LANCZOS).save(path, optimize=True)


def tsne_grid() -> str:
    """HTML for the before / after grid; writes build/tsne_<column>_<stage>.png."""
    lac = DATA["neighbor_metric"]["spaces"]
    cells = ['<div class="row-label"></div>'] + [
        f'<div class="column-title">{html_label(title)}</div>' for title, _, _ in TSNE_COLUMNS
    ]
    for stage, row in (("before", 1), ("after", 2)):
        cells.append(f'<div class="row-label"><span>{stage.capitalize()}</span></div>')
        for column, keys in enumerate(TSNE_COLUMNS):
            key = keys[row]
            image = f"tsne_{column}_{stage}.png"
            render_tsne(key, BUILD / image)
            tag = "<b>LAC key</b><br>" if key == LAC_KEY else ""
            cells.append(f'<figure><img src="build/{image}" alt=""><figcaption>{tag}LAC {lac[key]["lac@8"]:.2f}'
                         f"</figcaption></figure>")  # fmt: skip
    return '<div class="tsne">' + "".join(cells) + "</div>"


# ---- equations and QR codes -----------------------------------------------------------------------------------------
EQUATIONS = {  # name -> LaTeX, rendered to build/eq_<name>.svg
    "policy": r"\pi(q)=\sum_{i\in\mathcal{N}_k(q)}\alpha_i\,\mathbf{a}_{t_i:t_i+H},\qquad "
              r"\alpha_i\propto\exp\!\big(-d(q,i)/\tau\big)",
    "lac": r"\mathrm{LAC@}k(q)=1-\frac{\bar{d}_{\mathcal{N}_k(q)}}{\bar{d}_{\mathcal{C}(q)}},\qquad "
           r"\bar{d}_{\mathcal{S}}=\frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}}\lVert\mathbf{a}_q-\mathbf{a}_i\rVert_2^2",
}  # fmt: skip
LATEX_DOCUMENT = r"""\documentclass[border=1pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}$\displaystyle %s$\end{document}
"""


def render_equations() -> None:
    """Typeset each equation with pdfLaTeX and convert it to SVG with pdftocairo (poppler)."""
    for name, tex in EQUATIONS.items():
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "eq.tex").write_text(LATEX_DOCUMENT % tex)
            subprocess.run(
                ["pdflatex", "-interaction=batchmode", "eq.tex"], cwd=tmp, check=True, capture_output=True
            )
            subprocess.run(
                ["pdftocairo", "-svg", "eq.pdf", str(BUILD / f"eq_{name}.svg")], cwd=tmp, check=True
            )


def render_qr(url: str, name: str) -> None:
    segno.make(url, error="m").save(str(BUILD / f"qr_{name}.svg"), border=0, dark=INK, light=None)


# ---- the page -------------------------------------------------------------------------------------------------------
def main() -> None:
    BUILD.mkdir(exist_ok=True)
    render_equations()
    render_qr(PROJECT_URL, "page")
    render_qr(CODE_URL, "code")
    placeholders = {
        "CHART_BENCHMARK": chart_benchmark(width=242.0, scale=1.22, row_height=15.4),
        "CHART_BACKBONES": chart_backbones(width=294.0, scale=1.15, row_height=11.8),
        "CHART_LAC": chart_lac(width=252.0, height=96.0, scale=1.2),
        "TSNE": tsne_grid(),
    }
    page = (HERE / "template.html").read_text()
    for name, markup in placeholders.items():
        page = page.replace("{{" + name + "}}", markup)
    if leftover := re.findall(r"\{\{\w+\}\}", page):
        raise SystemExit(f"template.html has placeholders that build.py does not fill: {leftover}")
    (BUILD / "poster.html").write_text(page)
    print(f"wrote {BUILD / 'poster.html'}")


if __name__ == "__main__":
    main()
