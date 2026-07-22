# Paper A: variance retention in multi-horizon PM10 forecasting

This repository contains the manuscript, data-recovery code, row-level
predictions, aggregate results, and figures for an hourly PM10 case study at
Casa de Campo, Madrid.  The study examines whether persistence-relative RMSE
skill and forecast variance retention convey different horizon-wise
information under leakage-free rolling-origin evaluation.

## Canonical empirical scope

- Source: Madrid Open Data, Casa de Campo station 024, PM10 magnitude 10.
- Period: calendar year 2023 on a regular 8,760-point hourly grid.
- Valid observations: 8,465; missing or provider-invalid values: 295.
- Horizons: 1, 6, 24, and 48 hours.
- Models: fixed LightGBM and SARIMA configurations.
- Required baseline: persistence.
- Primary protocol: five-fold expanding rolling-origin evaluation.
- Secondary sensitivity analysis: one chronological 80/20 holdout.
- P75 event thresholds: estimated separately from each training fold.

The empirical source of truth is the CSV and Parquet material under
`outputs/reproduction/`.  `paper_a.tex` includes the canonical LaTeX table from
`outputs/tables/paper_a_rolling_results.tex`; that table is regenerated from the
rolling-origin CSV rather than maintained by hand.

## Non-negotiable safeguards

- preprocessing and threshold calibration use training information only;
- unavailable inputs may be forward-filled causally, never backward-filled;
- no full-series normalization is used in the Paper A reproduction;
- invalid or absent verification targets are excluded, not imputed;
- every model is compared with persistence on the same valid verification rows;
- metrics and diagnostics are reported separately by horizon;
- Skill_VP is an auxiliary diagnostic, not a replacement accuracy metric.

The repository also retains earlier daily P33/E1-RR modules for provenance.
Those legacy modules, their documentation, and their outputs are not empirical
sources for `paper_a.tex` and must not be mixed with the hourly Madrid case.

## Reproduce and validate

Create an environment and install the pinned empirical dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-reproduction.txt
```

Run the complete workflow:

```bash
make data          # recover and checksum the official 2023 archive
make reproduce     # rerun rolling-origin and holdout predictions
make figures       # regenerate the table and the two manuscript figures
make paper         # compile paper_a.pdf with BibTeX
make test          # run protocol, artifact, metric, and editorial checks
make editorial-check
```

The SARIMA reproduction updates state at every hourly origin and may take about
15 minutes on a laptop.  Exact run settings are recorded in
`outputs/reproduction/run_manifest_rolling_origin.json` and
`outputs/reproduction/run_manifest_holdout.json`.

## Key files

- `paper_a.tex`, `paper_a.pdf`: generic manuscript source and compiled PDF.
- `scripts/run_paper_a_empirical.py`: canonical empirical rerun.
- `scripts/recover_madrid_pm10.py`: official data recovery and checksum.
- `scripts/render_paper_a_results.py`: canonical LaTeX table generator.
- `src/plotting/plot_master_figure.py`: two manuscript figures.
- `scripts/check_paper_a_consistency.py`: manuscript/artifact divergence guard.
- `docs/empirical_reproducibility_audit.md`: recovery and leakage audit.
- `docs/reference_audit.md`: cited-reference verification.

## Data and licensing

Madrid Open Data distributes the hourly air-quality dataset under CC BY 4.0.
The exact annual archive URL and SHA-256 checksum are recorded in
`data/processed/casa_de_campo_pm10_2023.manifest.json`.

## Citation

If you use this software, cite:

> Garcia Crespi, F. F. (2026). *varret-pm10-paper: Variance Retention and
> Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting* (v1.0.0).
> Zenodo. https://doi.org/10.5281/zenodo.20185328
