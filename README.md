# P33 — Variance Retention and Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20185328.svg)](https://doi.org/10.5281/zenodo.20185328)
[![tests](https://github.com/fedeg-umh-es/varret-pm10-paper/actions/workflows/tests.yml/badge.svg)](https://github.com/fedeg-umh-es/varret-pm10-paper/actions/workflows/tests.yml)

Research codebase for P33. Evaluates whether positive multi-horizon PM10 forecasting skill under rolling-origin evaluation reflects credible dynamic value or coincides with variance collapse — a pattern the paper terms *plausible ghost skill*.

---

## Scientific question

P33 reads `skill` and `vr` (variance retention %) jointly:

| `skill` | `vr` | Interpretation |
|---|---|---|
| > 0 | ≈ 100 % | Credible dynamic improvement over persistence |
| > 0 | ≪ 100 % (`collapse_flag`) | Plausible ghost skill — low dynamic fidelity |
| > 0 | ≫ 100 % (`inflation_flag`) | Variance inflation — audit forecast distribution |

`skill_vp` is an auxiliary diagnostic: `skill × min(1, vr/100)`. It is not a replacement for the joint reading.

---

## Metrics

| Symbol | Definition | Implementation |
|---|---|---|
| Skill_RMSE | `1 − RMSE_model / RMSE_persistence` | `src/evaluation/skill.py` |
| VR(h) | `100 · Var_pred(h) / Var_obs(h)` | `src/diagnostics/variance.py: _compute_vr` |
| Skill_VP | `Skill × min(1, VR/100)` | `src/diagnostics/variance.py: build_variance_retention_summary` |

---

## Pipeline

```
data/raw/pm10_daily.csv            ← place raw data here (see docs/data_download.md)
         │
scripts/01  validate_raw_data
scripts/02  build_processed_datasets  → data/processed/pm10_daily.parquet
         │
scripts/03  run_baselines             → outputs/predictions/persistence.parquet
                                         outputs/predictions/seasonal_persistence.parquet
scripts/04  run_linear_models         → outputs/predictions/sarima.parquet
scripts/05  run_boosting_model        → outputs/predictions/lgbm.parquet
         │
scripts/06  build_skill_tables        → outputs/metrics/predictions.csv
                                         outputs/metrics/skill_summary.csv
scripts/07  build_variance_retention  → outputs/tables/variance_retention_summary.csv
scripts/08  build_run_summary         → outputs/reports/run_summary.txt
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data

See [`docs/data_download.md`](docs/data_download.md) for step-by-step EEA download instructions.

Place the file at `data/raw/pm10_daily.csv` with columns:
- `date`: `YYYY-MM-DD`
- `y`: daily mean PM10 concentration (µg/m³)

### 3. Run the full pipeline

```bash
python scripts/run_p33_pipeline.py
```

Or step by step (see `docs/runbook.md` for details):

```bash
python scripts/02_build_processed_datasets.py
python scripts/03_run_baselines.py
python scripts/04_run_linear_models.py      # SARIMA — ~5 min
python scripts/05_run_boosting_model.py     # LightGBM — ~2 min
python scripts/06_build_skill_tables.py
python scripts/07_build_variance_retention_table.py
python scripts/08_build_run_summary.py
```

### 4. Run tests (no data required)

```bash
pytest tests/ -v
```

---

## Data contracts

### Input: `data/processed/pm10_daily.parquet`

| Column | Type | Description |
|---|---|---|
| `date` | date | Daily timestamp |
| `y` | float | PM10 concentration (µg/m³) |

### Predictions: `outputs/predictions/*.parquet`

| Column | Description |
|---|---|
| `dataset` | Dataset identifier |
| `model` | Model identifier (`persistence`, `seasonal_persistence`, `sarima`, `lgbm`) |
| `fold` | Rolling-origin (origin, horizon) index |
| `origin_date` | Forecast origin date |
| `horizon` | Lead time (days, 1–7) |
| `date` | Target date |
| `y_true` | Observed PM10 value |
| `y_pred` | Predicted PM10 value |

### Variance retention summary: `outputs/tables/variance_retention_summary.csv`

| Column | Description |
|---|---|
| `skill` | Skill_RMSE relative to persistence |
| `vr` | Variance retention % — `100 · Var_pred / Var_obs` |
| `skill_vp` | `skill × min(1, vr/100)` |
| `collapse_flag` | `True` when `vr < 50` |
| `inflation_flag` | `True` when `vr > 150` |
| `near_ideal_flag` | `True` when `skill > 0` and `80 ≤ vr ≤ 120` |

---

## Repository layout

```
configs/              ← experiment and evaluation configuration (canonical)
  datasets/           ← dataset paths and column names
  evaluation/         ← rolling-origin settings, variance diagnostic thresholds
  experiments/        ← model lists per experiment
docs/                 ← protocol, runbook, data download instructions
scripts/              ← pipeline entry points (01–08 + orchestrator)
src/
  data/               ← schema, validation, loading
  diagnostics/        ← variance retention computation
  evaluation/         ← skill, metrics, aggregation
  models/             ← persistence, seasonal_persistence, SARIMA, LightGBM
  splits/             ← rolling-origin fold generation
tests/                ← unit tests (no data required)
```

---

## Rolling-origin protocol

- **Expanding window**: each origin uses all data up to that point as training
- `min_train_size`: 60 observations
- `max_horizon`: 7 days
- SARIMA re-fitted every 30 origins (~monthly); LightGBM every 30 origins
- Persistence and seasonal persistence evaluated at every origin
- No future data enters feature construction, scaling, or threshold estimation

---

## Configuration

Edit `configs/` files to adjust the experiment:

| File | Controls |
|---|---|
| `configs/datasets/pm10.yaml` | Raw and processed paths, column names |
| `configs/evaluation/rolling_origin.yaml` | `min_train_size`, `max_horizon`, `sarima_refit_every`, `lgbm_refit_every` |
| `configs/experiments/exp_p33_main.yaml` | Which models to run |
| `configs/evaluation/variance_diagnostics.yaml` | Thresholds for `collapse_flag`, `inflation_flag`, `near_ideal_flag` |

---

## Citation

```
García Crespi, F. (2026). varret-pm10-paper: Variance Retention and Diagnostic
Skill Adjustment in Multi-Horizon PM10 Forecasting (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.20185328
```
