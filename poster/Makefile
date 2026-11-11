# Rebuild the figures, tables and numbers from the experiment logs: `make` (or `make data` / `make figures`).
# The poster itself is built in corl2026/.
all: figures

data:
	uv run python scripts/prepare_data.py

figures: data
	uv run python scripts/make_figures.py

.PHONY: all data figures
