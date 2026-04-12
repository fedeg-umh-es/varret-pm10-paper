# Paper A: Ghost Skill & Variance Collapse in Multi-Horizon PM10 Forecasting

**Empirical diagnostic study on predictive skill, variance retention, and event-based usefulness in rolling-origin evaluation.**

---

## Paper A Scope

This repository implements **Paper A only** — a focused diagnostic study of three phenomena:

1. **Ghost Skill**: Positive relative skill (vs. persistence) coupled with severely degraded predictive variance
2. **Variance Collapse**: Predicted variance << observed variance, worsening with horizon
3. **Fit-for-Purpose**: Poor recall of high-pollution events despite positive RMSE skill

### Three Experiments

**E1: Canonical Horizon Table**
- Compare persistence, LightGBM, LSTM across horizons (h=1, 6, 24, 48)
- Metrics: RMSE, skill, variance, Skill_VP
- Protocol: rolling-origin
- Output: `outputs/tables/table_canonical_full.csv`

**E2: Protocol Robustness**
- Evaluate consistency of metrics across protocols
- Protocol comparison: rolling-origin vs. blocked holdout
- Models: LightGBM, LSTM
- Output: `outputs/metrics/metrics_protocol_comparison.csv`

**E3: Fit-for-Purpose Event Evaluation**
- Measure ability to predict high PM10 events
- Metric: recall of exceedances at P75, P90 thresholds
- Models: LightGBM, LSTM
- Output: `outputs/metrics/metrics_events_full.csv`

### Final Figure

**Master Figure** (`outputs/figures/master_figure.png`):
- Panel 1: Skill vs. horizon (ghost skill detection)
- Panel 2: Variance % vs. horizon (collapse detection)
- Panel 3: Skill_VP vs. horizon (diagnostic metric)
- Panel 4: Recall of events vs. horizon (usefulness)

---

## What Paper A Is NOT

- ✗ Not a method paper (no new models or architectures)
- ✗ Not a benchmarking study (no comparison with 10+ methods)
- ✗ Not multi-station or multi-pollutant (PM10 only, single location)
- ✗ Not Paper B (no explainability, recalibration, or deeper analysis)
- ✗ Not SARIMA, XGBoost, Prophet, or other methods (only persistence, LightGBM, LSTM)

---

## Repository Structure

```
ghost-skill-paper-a/
├── config/                          # Configuration files
│   ├── config.yaml                  # Main config (paths, hyperparameters)
│   ├── horizons.yaml                # Prediction horizons
│   └── thresholds.yaml              # Event detection thresholds
├── data/
│   ├── raw/                         # PM10 CSV (user-provided)
│   ├── interim/                     # Loaded & imputed data
│   └── processed/                   # Features, sequences, splits
├── src/
│   ├── data/                        # Data pipeline
│   │   ├── load_data.py
│   │   ├── preprocess_pm10.py
│   │   ├── make_features_lgbm.py
│   │   ├── make_sequences_lstm.py
│   │   └── make_splits.py
│   ├── models/                      # Model wrappers
│   │   ├── persistence.py
│   │   ├── lgbm_model.py
│   │   └── lstm_model.py
│   ├── training/                    # Model training
│   │   ├── train_persistence.py
│   │   ├── train_lgbm.py
│   │   └── train_lstm.py
│   ├── evaluation/                  # Metrics & analysis
│   │   ├── compute_metrics_by_horizon.py
│   │   ├── compute_event_metrics.py
│   │   ├── compare_protocols.py
│   │   └── build_canonical_table.py
│   └── plotting/
│       └── plot_master_figure.py
└── outputs/
    ├── predictions/                 # Model predictions
    ├── metrics/                     # CSV metrics files
    ├── tables/                      # Output tables
    └── figures/                     # Output plots
```

---

## Execution Order

All scripts use argparse and accept `--config` parameter. Run from repo root.

### Phase 1: Data Preparation (leakage-free)

```bash
# 1a. Load raw PM10 CSV
python src/data/load_data.py --config config/config.yaml
# Output: data/interim/raw_loaded.parquet

# 1b. Preprocess: imputation, detrending, z-score normalization
python src/data/preprocess_pm10.py --config config/config.yaml
# Output: data/processed/pm10_preprocessed.parquet
#         data/processed/normalization_params.json

# 1c. Create LightGBM features: lags, rolling stats, temporal
python src/data/make_features_lgbm.py --config config/config.yaml
# Output: data/processed/features_lgbm.parquet

# 1d. Create LSTM sequences: (context_len, 1) format
python src/data/make_sequences_lstm.py --config config/config.yaml
# Output: data/processed/sequences_lstm.npz

# 1e. Create time-ordered splits: rolling-origin + holdout
python src/data/make_splits.py --config config/config.yaml
# Output: data/processed/splits_rolling_origin.json
#         data/processed/splits_holdout.json
```

### Phase 2: Model Training (time-ordered, no leakage)

```bash
# 2a. Train persistence baseline (no parameters)
python src/training/train_persistence.py --config config/config.yaml
# Output: outputs/predictions/persistence_rolling_origin.parquet
#         outputs/predictions/persistence_holdout.parquet

# 2b. Train LightGBM (independent model per horizon)
python src/training/train_lgbm.py --config config/config.yaml
# Output: outputs/predictions/lgbm_rolling_origin.parquet
#         outputs/predictions/lgbm_holdout.parquet
#         outputs/models/lgbm_rolling_origin/fold_*
#         outputs/models/lgbm_holdout/

# 2c. Train LSTM (single model outputs all horizons)
python src/training/train_lstm.py --config config/config.yaml
# Output: outputs/predictions/lstm_rolling_origin.parquet
#         outputs/predictions/lstm_holdout.parquet
#         outputs/models/lstm_rolling_origin/fold_*
#         outputs/models/lstm_holdout/
```

### Phase 3: Compute Metrics (E1, E2, E3)

```bash
# 3a. Compute RMSE, Skill, Variance, Skill_VP by horizon
python src/evaluation/compute_metrics_by_horizon.py --config config/config.yaml
# Output: outputs/metrics/metrics_rolling_origin_by_horizon.csv
#         outputs/metrics/metrics_holdout_by_horizon.csv
# Columns: model | h | rmse | skill | var_obs | var_pred | var_pct | skill_vp

# 3b. Compute event metrics: recall, precision, F1
python src/evaluation/compute_event_metrics.py --config config/config.yaml
# Output: outputs/metrics/metrics_events_rolling_origin.csv
#         outputs/metrics/metrics_events_holdout.csv
#         outputs/metrics/metrics_events_full.csv
# Columns: model | h | threshold | recall_events | skill | var_pct | skill_vp
```

### Phase 4: Analysis & Tables (E2, E1)

```bash
# 4a. Compare rolling-origin vs holdout (E2)
python src/evaluation/compare_protocols.py --config config/config.yaml
# Output: outputs/metrics/metrics_protocol_comparison.csv
# Columns: model | h | metric | ro_value | ho_value | diff_absolute | diff_pct

# 4b. Build canonical table (E1)
python src/evaluation/build_canonical_table.py --config config/config.yaml
# Output: outputs/tables/table_canonical_full.csv
# Columns: h | rmse_persistence | var_obs | lgbm_rmse | lgbm_skill | lgbm_var_pct 
#          | lgbm_skill_vp | lstm_rmse | lstm_skill | lstm_var_pct | lstm_skill_vp
#          | lgbm_recall_p75 | lstm_recall_p75 | ...
```

### Phase 5: Visualization (Master Figure)

```bash
# 5. Generate master figure
python src/plotting/plot_master_figure.py --config config/config.yaml
# Output: outputs/figures/master_figure.png
# Contains 4 panels: Skill, Var%, Skill_VP, Recall vs. horizon
```

---

## Configuration Files

### `config/config.yaml`

Main config with:
- **Data paths**: raw → interim → processed → outputs
- **Preprocessing**: imputation method, normalization
- **Features**: lag windows, rolling windows for LightGBM
- **LSTM**: context length (168 = 7 days), hyperparameters
- **Splits**: rolling-origin (5 folds, expanding window), holdout (80/20)
- **Model hyperparameters**: LightGBM (n_estimators, max_depth), LSTM (units, dropout, epochs)
- **Plotting**: figure size, colors, fonts

### `config/horizons.yaml`

Prediction horizons in hours (frequency-dependent):
```yaml
horizons:
  - 1      # Very short-term
  - 6      # Short-term
  - 24     # Medium-term (1 day)
  - 48     # Long-term (2 days)
```

### `config/thresholds.yaml`

Event detection thresholds:
```yaml
event_detection:
  method: "percentile"
  percentiles: [75, 90]  # P75, P90 of training observations
```

---

## Input Data Requirements

**File**: `data/raw/pm10_measurements.csv`

Minimal required columns (auto-detected):
- **Timestamp**: `timestamp`, `time`, `date`, or similar
- **PM10 value**: `pm10_value`, `pm10`, `value`, or similar
- **Frequency**: 1-hour assumed (configurable via context length)

Example:
```csv
timestamp,pm10_value
2020-01-01 00:00:00,45.2
2020-01-01 01:00:00,48.5
2020-01-01 02:00:00,50.1
...
```

---

## Key Metrics & Definitions

### Horizon-based Metrics

For each (model, horizon h):

- **RMSE**: √(mean((y_pred - y_true)²))
- **Skill**: 1 - RMSE_model / RMSE_persistence
  - Positive: better than persistence
  - Negative: **ghost skill** (worse than persistence despite variance)
- **var_obs**: variance of observed values
- **var_pred**: variance of predicted values
- **var_pct**: 100 × var_pred / var_obs
  - 100% = full variance captured
  - <100% = **variance collapse** (underdispersion)
  - >100% = variance expansion (overdispersion)
- **Skill_VP**: skill × (var_pct / 100)
  - Diagnostic metric penalizing skill by variance collapse
  - "Useful skill" accounting for distribution fidelity

### Event Metrics

For each (model, horizon, threshold):

- **recall**: P(pred > threshold | obs > threshold)
  - Ability to detect high PM10 events
- **precision**: P(obs > threshold | pred > threshold)
  - False alarm rate
- **F1**: Harmonic mean of precision and recall

---

## Rolling-Origin Protocol (E2 Robustness)

**Expanding window** with K folds:

```
Fold 0: |----train----|-test-|
Fold 1: |------train------|--test-|
Fold 2: |--------train---------|---test-|
...
```

- **Train** grows with each fold (expanding window)
- **Test** moves forward chronologically
- **No overlap** between train and test
- **Prevents temporal leakage**

**Holdout** (for comparison):

```
|--------train--------|----test----|
        (80%)           (20%)
```

---

## Leakage-Free Evaluation

Critical design choices:

1. **Per-horizon models** (LightGBM): Prevent information leakage across horizons
2. **Train-only preprocessing**: Normalization parameters fitted on train only
3. **Time-ordered splits**: No future data used to compute past transformations
4. **No lookback beyond context**: LSTM context strictly ≤ 168 hours
5. **Validation split (LSTM)**: 15% of train, no contamination of test

---

## Skill_VP Interpretation

**Skill_VP is a diagnostic auxiliary metric**, not a replacement for RMSE:

- **Skill_VP > 0.1**: Model maintains both accuracy and variance fidelity
- **Skill ≈ 0.3, var_pct ≈ 0.3**: Ghost skill visible — model reduces variance while claiming skill
- **Skill ≈ -0.05, var_pct ≈ 0.4**: Honest diagnosis — worse than persistence, collapsed variance

Use Skill_VP to **diagnose** ghost skill, not to rank models overall. RMSE remains primary metric for accuracy; Skill_VP adds variance context for usefulness.

---

## Expected Outputs

### 1. Canonical Table (`table_canonical_full.csv`)

| h   | rmse_persist | var_obs | lgbm_rmse | lgbm_skill | lgbm_var_pct | lgbm_skill_vp | lstm_rmse | lstm_skill | ... |
|-----|--------------|---------|-----------|------------|--------------|---------------|-----------|------------|-----|
| 1   | 14.2         | 195     | 13.8      | 0.028      | 82.5         | 0.023         | 13.9      | 0.021     | ... |
| 6   | 18.5         | 280     | 17.1      | 0.076      | 65.4         | 0.050         | 17.8      | 0.038     | ... |
| 24  | 22.3         | 420     | 22.9      | -0.027     | 35.2         | -0.009        | 21.5      | 0.036     | ... |
| 48  | 25.1         | 520     | 26.8      | -0.068     | 28.1         | -0.019        | 24.2      | 0.036     | ... |

**Interpretation**: LightGBM shows ghost skill at h=24,48 (negative or very low skill despite positive Skill_VP). LSTM maintains skill across horizons.

### 2. Protocol Comparison (`metrics_protocol_comparison.csv`)

| model | h  | metric   | ro_value | ho_value | diff_pct |
|-------|----|---------:|----------|----------|----------|
| lgbm  | 1  | rmse     | 13.8     | 13.9     | 0.7%     |
| lgbm  | 1  | skill    | 0.028    | 0.025    | 10.7%    |
| ...   | .. | ...      | ...      | ...      | ...      |

**Interpretation**: Low diff_pct → rolling-origin and holdout agree (protocol robust). High diff_pct → may indicate overfitting or protocol instability.

### 3. Event Metrics (`metrics_events_full.csv`)

| protocol      | model | h  | threshold | recall_events | skill  | var_pct | skill_vp |
|---------------|-------|----|-----------|-----------:|------:|---------:|-----:|
| rolling_origin | lgbm | 24 | 75        | 0.65      | -0.027 | 35.2     | -0.009 |
| rolling_origin | lstm | 24 | 75        | 0.72      | 0.036  | 42.1     | 0.015 |
| ...           | ...   | .. | ...       | ...       | ...    | ...      | ...    |

**Interpretation**: LSTM catches more events (recall 0.72) than LightGBM (0.65), with better variance fidelity.

### 4. Master Figure (`master_figure.png`)

4-panel diagnostic plot showing ghost skill presence, variance collapse severity, Skill_VP trajectory, and event recall across horizons.

---

## Reproduction

To reproduce Paper A exactly:

```bash
# Full pipeline (assuming pm10_measurements.csv in data/raw/)
bash scripts/run_paper_a.sh  # (optional convenience script)

# Or step-by-step:
python src/data/load_data.py && \
python src/data/preprocess_pm10.py && \
python src/data/make_features_lgbm.py && \
python src/data/make_sequences_lstm.py && \
python src/data/make_splits.py && \
python src/training/train_persistence.py && \
python src/training/train_lgbm.py && \
python src/training/train_lstm.py && \
python src/evaluation/compute_metrics_by_horizon.py && \
python src/evaluation/compute_event_metrics.py && \
python src/evaluation/compare_protocols.py && \
python src/evaluation/build_canonical_table.py && \
python src/plotting/plot_master_figure.py

# Check final outputs
ls -la outputs/tables/table_canonical_full.csv
ls -la outputs/figures/master_figure.png
```

---

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `numpy`, `pandas`, `scipy` — numerical computing
- `lightgbm` — LightGBM models
- `tensorflow` — LSTM (optional; remove if LightGBM only)
- `matplotlib`, `seaborn` — visualization
- `pyyaml` — configuration
- `scikit-learn` — utilities

---

## Important Notes

1. **Seed reproducibility**: All random seeds fixed (config), but LSTM training may have small variations due to GPU/TF nondeterminism
2. **LSTM requires TensorFlow**: Comment out LSTM scripts if TF not available
3. **Data frequency**: Assumes 1-hour frequency; adjust `context_length` in config for other frequencies
4. **Normalization**: Z-score fitted per rolling-origin fold (train only) — each test set normalized using fold's train statistics
5. **No hyperparameter tuning**: Hyperparameters fixed in config; Paper A is diagnostic, not optimization-focused

---

## Limitations (Acknowledged in Paper A)

- Single location, single pollutant (PM10)
- Fixed horizons (not adaptive)
- Simple models (no ensembles, physics-informed components)
- No exogenous variables (temperature, humidity, pressure)
- No uncertainty quantification (point forecasts only)

---

## Citations & Framing

**Paper A** demonstrates that positive relative skill metric does not guarantee useful forecasts when predictive variance collapses. Skill_VP is presented as a **diagnostic auxiliary metric** to flag this phenomenon, not as a universal replacement for RMSE-based metrics.

---

## Contact

Paper A repository: `ghost-skill-paper-a`  
Scope: Diagnostic empirical study, 3 experiments, rolling-origin evaluation  
Last updated: 2026-04-11
