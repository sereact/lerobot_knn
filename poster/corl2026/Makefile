# CoRL 2026 workshop poster (16:9, 1360 x 765 mm): `make` -> poster.pdf and preview.png.
# Needs uv, Chromium, pdfLaTeX and poppler (pdftocairo, pdftoppm), and the Inter font installed.
CHROME ?= chromium

all: poster.pdf preview.png

build/poster.html: build.py template.html ../data/poster_data.json ../data/embedding_maps.json
	uv run --script build.py

poster.pdf: build/poster.html $(wildcard assets/*)
	$(CHROME) --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
	  --print-to-pdf=$(CURDIR)/poster.pdf file://$(CURDIR)/build/poster.html

preview.png: poster.pdf
	pdftoppm -r 36 -png -singlefile poster.pdf preview

clean:
	rm -rf build

.PHONY: all clean
