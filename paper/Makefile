# Build the workshop paper: `make` (figures, tables and numbers.tex come from the poster's scripts in paper mode).
all: pdf

figures:
	cd .. && uv run python poster/scripts/prepare_data.py && KNN_FIG_TARGET=paper uv run python poster/scripts/make_figures.py

pdf: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

clean:
	latexmk -C paper.tex

.PHONY: all figures pdf clean
