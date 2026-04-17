# P33 — Variance Retention and Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting Under Rolling-Origin Evaluation

This repository is a paper-first research codebase for P33. It is designed to support a reproducible analysis of whether positive multi-horizon forecasting skill for daily PM10 under rolling-origin evaluation reflects credible operational value or instead coincides with variance collapse and plausible ghost skill.

## Scientific Question

P33 evaluates whether a model can outperform a persistence baseline across horizons up to 7 days while still preserving enough dynamic variability to remain operationally interpretable. The central diagnostic is the joint reading of:

- `skill`: baseline-relative forecast improvement
- `alpha`: variance-retention indicator
- `skill_vp`: auxiliary diagnostic adjustment combining skill and variance retention

High skill with `alpha` near 1 supports stronger dynamic credibility. High skill with very low `alpha` is treated as a plausible ghost-skill pattern.

## Scope

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
