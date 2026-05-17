# Runbook

## Prerequisites

- Python 3.10+, virtual environment activated
- `pip install -r requirements.txt`
- Raw PM10 dataset at `data/raw/pm10_daily.csv` — see `docs/data_download.md`

---

## Pipeline steps

```bash
# Step 1: Validate raw data file exists and has expected columns
python scripts/01_validate_raw_data.py

# Step 2: Build canonical (date, y) processed dataset
python scripts/02_build_processed_datasets.py
# → data/processed/pm10_daily.parquet  (365 rows for year 2023)

# Step 3: Run baseline models — fast (seconds)
python scripts/03_run_baselines.py
# → outputs/predictions/persistence.parquet
# → outputs/predictions/seasonal_persistence.parquet

# Step 4: Run SARIMA — moderate (~5 min, ~10 re-fits on daily data)
python scripts/04_run_linear_models.py
# → outputs/predictions/sarima.parquet
# Re-fit frequency controlled by configs/evaluation/rolling_origin.yaml: sarima_refit_every

# Step 5: Run LightGBM — moderate (~2 min)
python scripts/05_run_boosting_model.py
# → outputs/predictions/lgbm.parquet
# Re-fit frequency controlled by: lgbm_refit_every

# Step 6: Aggregate predictions and compute skill relative to persistence
python scripts/06_build_skill_tables.py
# → outputs/metrics/predictions.csv    (all models combined)
# → outputs/metrics/skill_summary.csv  (skill per model per horizon)

# Step 7: Build variance retention diagnostic table (key P33 output)
python scripts/07_build_variance_retention_table.py
# → outputs/tables/variance_retention_summary.csv

# Step 8: Write run summary
python scripts/08_build_run_summary.py
# → outputs/reports/run_summary.txt
```

Or run everything at once:

```bash
python scripts/run_p33_pipeline.py
```

---

## Configuration reference

| File | Key settings |
|---|---|
| `configs/datasets/pm10.yaml` | `raw_path`, `processed_path`, `date_column`, `target_column` |
| `configs/evaluation/rolling_origin.yaml` | `max_horizon`, `min_train_size`, `sarima_refit_every`, `lgbm_refit_every` |
| `configs/experiments/exp_p33_main.yaml` | List of models to run |
| `configs/evaluation/variance_diagnostics.yaml` | Flag thresholds |

---

## Running tests (no data required)

```bash
pytest tests/ -v
```

Five tests covering:
- `test_rolling_origin.py`: temporal ordering — test index always after train indices
- `test_skill.py`: skill positive when model beats baseline
- `test_variance.py`: vr formula, collapse/inflation flags, skill_vp cap

---

## Interpreting `variance_retention_summary.csv`

| `collapse_flag` | `inflation_flag` | `near_ideal_flag` | Reading |
|---|---|---|---|
| False | False | True | Credible improvement — high skill, vr ≈ 100 % |
| True | False | False | Plausible ghost skill — model has low dynamic fidelity |
| False | True | False | Variance inflation — predictions overshoot observed variability |
| False | False | False | Mixed — read `skill` and `vr` jointly |

---

## Adding a new model

1. Implement the model class in `src/models/my_model.py` with `fit()` and `predict()`.
2. Create or extend a script in `scripts/` that calls the model and saves predictions
   in the `PREDICTIONS_COLUMNS` schema to `outputs/predictions/my_model.parquet`.
3. Add the model name to `configs/experiments/exp_p33_main.yaml`.
4. Re-run from step 6 (`build_skill_tables`) — steps 3–5 only need re-running for
   the new model.

---

## Adding a new dataset

1. Place the raw CSV at `data/raw/<dataset_name>.csv` with columns `date` and `y`.
2. Copy and edit `configs/datasets/pm10.yaml` → `configs/datasets/<dataset_name>.yaml`.
3. Update `configs/experiments/exp_p33_main.yaml` to include the new dataset name.
4. Re-run the full pipeline.
