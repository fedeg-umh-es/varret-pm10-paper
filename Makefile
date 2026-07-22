.PHONY: paper figures test clean

paper: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper_a.tex

figures:
	python3 src/plotting/plot_master_figure.py

test:
	pytest -q

clean:
	latexmk -C paper_a.tex
