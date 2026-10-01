# Poster: "Your VLA is Secretly kNN"

A0 landscape workshop poster (LuaLaTeX, tcolorbox `poster` library, Inter + Fira Math).

```bash
make          # collect results -> render figures + numbers.tex -> poster.pdf
```

- `scripts/prepare_data.py` gathers every number from the experiment logs (`outputs/knn`, `outputs/vla`)
  into `data/poster_data.json`.
- `scripts/make_figures.py` renders `figures/*.pdf` at print size and writes `numbers.tex`. No number in
  `poster.tex` is typed by hand.
- Panels whose experiment has not run yet render as hatched "EXPERIMENT RUNNING" boxes, and their numbers
  render as **running**. They fill in on the next `make` once the data exists:
  - §4 `head_r2`: `experiments/vla_rollouts.py --record` + `experiments/head_vs_knn.py`
  - §5 `embedders`: `experiments/compare.py outputs/vla/configs/plain_*.json --tag=vla_plain`
  - §5 `dose`: the own SmolVLA fine-tune (`scripts/train_smolvla_libero.sh`), writing
    `outputs/vla/dose_response.json`

Colour roles are fixed across the poster: orange = kNN (ours), blue = learned models, grey = human demos and
references, and an ordinal blue ramp = how much robot/LIBERO training the VLM has seen. They were checked with the
dataviz palette validator.
