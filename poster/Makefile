# Rebuild the poster from the experiment logs: `make` (or `make figures` / `make pdf`).
all: pdf

data:
	uv run python scripts/prepare_data.py

figures: data
	uv run python scripts/make_figures.py

pdf: figures
	latexmk -lualatex -interaction=nonstopmode -halt-on-error poster.tex

clean:
	latexmk -C poster.tex

.PHONY: all data figures pdf clean
