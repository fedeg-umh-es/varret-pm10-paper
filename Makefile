.PHONY: paper figures data reproduce test editorial-check clean

paper: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper_a.tex

figures:
	MPLCONFIGDIR=tmp/matplotlib python3 scripts/render_paper_a_results.py
	MPLCONFIGDIR=tmp/matplotlib python3 src/plotting/plot_master_figure.py

data:
	python3 scripts/recover_madrid_pm10.py

reproduce: data
	python3 scripts/run_paper_a_empirical.py --protocol rolling_origin
	python3 scripts/run_paper_a_empirical.py --protocol holdout
	python3 src/plotting/plot_master_figure.py

test:
	python3 -m pytest -q

editorial-check:
	python3 scripts/check_paper_a_consistency.py

clean:
	latexmk -C paper_a.tex
