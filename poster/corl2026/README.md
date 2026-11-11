# Poster: How Much of Your VLA is kNN

Poster for the CoRL 2026 Workshop on The Action Gap. It is a 16:9 page (1360 × 765 mm) made for an online poster
session: `poster.pdf`, with a low-resolution `preview.png`.

## Build

```bash
make          # build/poster.html -> poster.pdf -> preview.png
make clean    # remove build/
```

Requirements: [uv](https://docs.astral.sh/uv/), Chromium (`make CHROME=google-chrome` for another binary), pdfLaTeX
with `standalone` and `amsmath`, poppler (`pdftocairo`, `pdftoppm`), and the Inter font installed on the system.
`build.py` declares its Python dependencies inline, so `uv run --script build.py` needs no environment.

## Files

| File | Role |
|---|---|
| `template.html` | Layout, styles and all the text of the poster. Edit this to change wording or layout. |
| `build.py` | Draws the charts as inline SVG, renders the t-SNE maps, the equations and the QR codes, and fills the template. Edit this to change a chart. |
| `assets/pipeline.svg`, `assets/lac.svg` | Figures 1 and 2 of the paper, cropped from the camera-ready PDF as vector graphics. |
| `assets/logo_*.svg` | Sereact and KIT logos. |
| `build/` | Everything generated (git-ignored): `poster.html`, `tsne_*.png`, `eq_*.svg`, `qr_*.svg`. |

## Where the numbers come from

- The benchmark and backbone charts use Tables 1 and 2 of the camera-ready paper, copied into `build.py`.
- LAC values, the LAC-versus-success scatter and the t-SNE maps are read from `../data/poster_data.json` and
  `../data/embedding_maps.json`, which `../scripts/prepare_data.py` collects from the experiment logs.
- The headline numbers, the ablation bars and the figures quoted in the text are typed in `template.html` and match
  the paper.

Colours are fixed across the poster and the project page: orange for k-NN (ours), blue for VLAs, grey for generic
encoders.
