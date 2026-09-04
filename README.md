> [!WARNING]
> **REPOSITORIO ARCHIVADO / LEGACY:** Este repositorio ha sido consolidado en [`../varret-pm10-paper`](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper). Ver [`README_ARCHIVED.md`](README_ARCHIVED.md) para más detalles.

# P4 — Variance Retention as a Diagnostic Complement to Persistence-Relative Skill in Multi-Horizon PM10 Forecasting
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20185328.svg)](https://doi.org/10.5281/zenodo.20185328)

The canonical P4 manuscript is `paper_a.tex`. Its checked-in hourly Casa de
Campo reproduction is identified by producer commit
`f57f076078760af8a88bd87815fdf94ab0064fa3`, uses
`scripts/run_paper_a_empirical.py`, and is documented in
`docs/p4_canonical_provenance_manifest.json`.

This repository also retains historical daily P33/E1-RR pipelines and adjacent
exploratory materials. They are preserved for traceability but are **legacy and
non-canonical for the hourly P4 manuscript**. They must not be used to regenerate
P4 results.

## Legacy daily P33/E1-RR work package (non-canonical for P4)

The historical scope was restricted to the post-evaluation of E1-RR outputs through variance retention, `alpha`, and `skill_vp` diagnostics.

Do not mix this work package with E2-MET, E3-PROB, meteorological ablations, probabilistic extensions, or new model-family exploration.

See `docs/e1_rr_post_evaluation_contract.md` before running or modifying the pipeline.

## Legacy daily scientific question

P33 evaluates whether a model can outperform a persistence baseline across horizons up to 7 days while still preserving enough dynamic variability to remain operationally interpretable. The central diagnostic is the joint reading of:

- `skill`: baseline-relative forecast improvement
- `alpha`: variance-retention indicator
- `skill_vp`: auxiliary diagnostic adjustment combining skill and variance retention

High skill with `alpha` near 1 supports stronger dynamic credibility. High skill with very low `alpha` is treated as a plausible ghost-skill pattern.

## Legacy daily scope

In scope:

- daily PM10 forecasting
- leakage-free rolling-origin evaluation
- train-only preprocessing when needed inside each split
- persistence as mandatory baseline
- optional seasonal persistence if temporal structure justifies it
- linear or autoregressive models
- one boosting tabular model with lagged inputs
- variance-retention diagnostics and final tabular outputs for the paper

Out of scope:

- notebooks
- figures as a development target
- additional model families
- unrelated exploratory analyses
- generic public API design

## Repository layout and legacy boundary

The following directories primarily contain the legacy daily P33 pipeline and
adjacent explorations; they are not the canonical P4 producer unless a file is
listed in the P4 provenance manifest:

- `configs/`: dataset, experiment, and evaluation configuration
- `scripts/`: executable entry points for the paper workflow
- `src/`: core implementation modules
- `tests/`: conceptual tests for rolling-origin, skill, and variance diagnostics
- `docs/`: protocol, data dictionary, and runbook

## Legacy daily data contracts

Canonical processed dataset:

- `date`
- `y`

Predictions table:

- `dataset`
- `model`
- `fold`
- `origin_date`
- `horizon`
- `date`
- `y_true`
- `y_pred`

Aggregated skill table:

- `dataset`
- `model`
- `horizon`
- `skill`

Final diagnostic table:

- `dataset`
- `model`
- `horizon`
- `skill`
- `alpha`
- `skill_vp`
- `collapse_flag`
- `inflation_flag`
- `near_ideal_flag`

Required project output:

- `outputs/tables/variance_retention_summary.csv`

## Legacy daily pipeline

1. Validate raw daily PM10 data.
2. Build canonical processed datasets with columns `date` and `y`.
3. Generate leakage-free rolling-origin splits with `Hmax = 7`.
4. Run persistence and optional seasonal persistence baselines.
5. Run linear/autoregressive models.
6. Run one lag-based boosting model.
7. Build skill tables relative to persistence.
8. Build the variance-retention summary table.
9. Write a run summary referencing the final P33 output table.

## Legacy daily execution

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the minimal paper pipeline:

```bash
python scripts/run_p33_pipeline.py
```

## Canonical P4 empirical reproduction

The submitted case study uses the 2023 hourly Madrid Open Data archive for
Casa de Campo (station 024, PM10 magnitude 10). The checked-in archive is
verified by SHA-256; invalid measurements remain missing and are never filled
from the future.

```bash
make data          # re-download, verify, and parse the official archive
make reproduce     # rolling-origin + 80/20 holdout + figures
make paper         # compile the manuscript
make test          # causal-protocol and metric tests
```

For the exact empirical software environment used in the committed rerun:

```bash
pip install -r requirements-reproduction.txt
```

The SARIMA evaluation updates state at every hourly origin and can take about
15 minutes on a laptop. Row-level prediction artifacts and their aggregate
tables are in `outputs/reproduction/`. The recovery and discrepancy audit is
in `docs/empirical_reproducibility_audit.md`; the immutable path/hash map is in
`docs/p4_canonical_provenance_manifest.json`.

### Legacy daily diagnostic command

Run the variance-retention table only:

```bash
python scripts/07_build_variance_retention_table.py
```

Run tests:

```bash
pytest
```

Build the two submission figures and compile `paper_a.tex` with BibTeX:

```bash
make paper
```

The canonical P4 figures read
`outputs/reproduction/metrics_rolling_origin.csv` and
`outputs/reproduction/events_p75_rolling_origin.csv`. Their underlying hourly
row-level predictions are distributed in
`outputs/reproduction/predictions_rolling_origin.parquet`; the analogous holdout
artifacts are stored beside them.

## Notes

The canonical P4 implementation is intentionally restrained, explicit, and
auditable. Historical P33, meteorology, KGE, H*, multi-station, and model-family
materials remain available only as non-canonical legacy context unless the P4
provenance manifest names them.
## Citation

If you use this software, please cite:

> García Crespi, F. (2026). varret-pm10-paper: Variance Retention and Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20185328
