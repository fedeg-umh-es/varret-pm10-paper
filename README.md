# P33 — Variance Retention Diagnostic for Persistence-Relative Skill in Daily PM10 Forecasting
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20185328.svg)](https://doi.org/10.5281/zenodo.20185328)

This repository is a paper-first research codebase for the PM10 variance-retention
paper. It supports a reproducible analysis of whether positive multi-horizon
forecasting skill for daily PM10 under rolling-origin evaluation can coexist with
forecast-trajectory variance collapse.

The active manuscript source is Overleaf. Local TeX/PDF files are treated as
exports or historical snapshots unless explicitly synchronized from Overleaf.
The target submission strategy is Q1, with the paper positioned as a restrained
post-evaluation diagnostic study rather than a new model paper.

## Current Work Package

The active scope is restricted to the post-evaluation of E1-RR outputs through variance retention, `alpha`, and `skill_vp` diagnostics.

Do not mix this work package with E2-MET, E3-PROB, meteorological ablations,
probabilistic extensions, or new model-family exploration.

See `docs/e1_rr_post_evaluation_contract.md` before running or modifying the pipeline.

## Scientific Question

P33 evaluates whether a model can outperform a persistence baseline across horizons up to 7 days while still preserving enough dynamic variability to remain operationally interpretable. The central diagnostic is the joint reading of:

- `skill`: baseline-relative forecast improvement
- `alpha`: variance-retention indicator
- `skill_vp`: auxiliary diagnostic adjustment combining skill and variance retention

High skill with `alpha` near 1 supports stronger dynamic credibility. High skill with very low `alpha` is treated as a plausible ghost-skill pattern.

## Scope for the Q1 version

In scope:

- daily PM10 forecasting
- leakage-free rolling-origin evaluation
- train-only preprocessing when needed inside each split
- persistence as mandatory baseline
- linear/direct lag-only model
- one lag-based boosting or histogram-gradient model
- variance-retention diagnostics and final tabular outputs for the paper
- event/exceedance relevance and Murphy-style decomposition only when they
  directly support the variance-collapse claim

Out of scope:

- notebooks
- figures as a development target
- additional model families
- unrelated exploratory analyses
- generic public API design
- H* cross-domain claims from `paper2H`
- treating `Skill_VP` as a universal replacement for RMSE, skill, or formal tests

## Repository Layout

The repository contains legacy materials from earlier work and a new minimal P33-oriented structure. The P33 pipeline lives in the following directories:

- `configs/`: dataset, experiment, and evaluation configuration
- `scripts/`: executable entry points for the paper workflow
- `src/`: core implementation modules
- `tests/`: conceptual tests for rolling-origin, skill, and variance diagnostics
- `docs/`: protocol, data dictionary, and runbook

## Data Contracts

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

## Pipeline

1. Validate raw daily PM10 data.
2. Build canonical processed datasets with columns `date` and `y`.
3. Generate leakage-free rolling-origin splits with `Hmax = 7`.
4. Run persistence and optional seasonal persistence baselines.
5. Run linear/autoregressive models.
6. Run one lag-based boosting model.
7. Build skill tables relative to persistence.
8. Build the variance-retention summary table.
9. Write a run summary referencing the final P33 output table.

For submission cleanup, every manuscript figure/table should map to one artifact
under `outputs/figures/`, `outputs/tables/`, or `outputs/metrics/`. Duplicate
root-level figure exports are treated as convenience copies only.

## Execution

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

Run the variance-retention table only:

```bash
python scripts/07_build_variance_retention_table.py
```

Run tests:

```bash
pytest
```

## Notes

This repository is oriented to the P33 paper workflow rather than a general-purpose forecasting package. The implementation is intentionally restrained, explicit, and auditable.

## Companion KGE Note

A consolidated companion-note package for the horizon-wise KGE diagnostic is indexed in `docs/companion_kge/`. It documents the post-evaluation KGE analysis, final tables and figures, supported claims, limitations, and manuscript drafting scaffold without regenerating the base forecasts.

## Citation

If you use this software, please cite:

> García Crespi, F. (2026). varret-pm10-paper: Variance Retention and Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20185328
