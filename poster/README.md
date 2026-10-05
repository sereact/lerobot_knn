# Poster: "Your VLA is Secretly kNN"

A1 portrait workshop poster (594 × 841 mm; LuaLaTeX, tcolorbox `poster` library, Inter + Fira Math).

```bash
make          # collect results -> render figures, tables and numbers.tex -> poster.pdf
```

- `scripts/prepare_data.py` gathers every number the poster and the paper use from the experiment results under
  `outputs/` (`outputs/knn`, `outputs/vla`, `outputs/libero_plus`) into `data/poster_data.json`, and the 2-D maps of
  every key into `data/embedding_maps.json`. Both are tracked, so the figures can be rebuilt without the (local)
  result files.
- `scripts/make_figures.py` renders `figures/*.pdf` (task map, LIBERO-plus per perturbation, VLA-as-lookup, LAC vs
  closed loop, Optuna, embedding maps before / after fine-tuning), the tables `figures/{results,backbones,ablations}_table.tex`,
  and `numbers.tex`. No number in `poster.tex` is typed by hand.
- `knn_diagram.tex` is the TikZ diagram of the kNN policy.

The paper uses the same scripts: `KNN_FIG_TARGET=paper` renders the figures at the paper's column width and smaller
fonts into `paper/figures/` (plus the appendix grids of every key's t-SNE and PCA map) and writes `paper/numbers.tex`;
`make` in `paper/` does this and runs pdfLaTeX.

Colour roles are fixed across the poster and the paper: orange = kNN (ours), blue = learned models (VLAs), grey =
human demonstrations and generic encoders, and an ordinal blue ramp = how much robot/LIBERO training a VLM has seen.
They were checked with the dataviz palette validator.
