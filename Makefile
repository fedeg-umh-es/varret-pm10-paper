.PHONY: paper figures data reproduce test clean

paper: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper_a.tex

figures:
	python3 scripts/render_paper_a_results.py
	python3 src/plotting/plot_master_figure.py

data:
	python3 scripts/recover_madrid_pm10.py

reproduce: data
	python3 scripts/run_paper_a_empirical.py --protocol rolling_origin
	python3 scripts/run_paper_a_empirical.py --protocol holdout
	python3 src/plotting/plot_master_figure.py

test:
	pytest -q

clean:
	latexmk -C paper_a.tex
