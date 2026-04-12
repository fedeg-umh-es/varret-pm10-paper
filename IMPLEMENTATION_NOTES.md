# Implementation Notes for Paper A

## Architecture Decisions

### 1. Per-Horizon Models (LightGBM)
- **Why**: Prevents temporal information leakage across horizons
- **How**: Train separate LGBMRegressor for each horizon (h=1, 6, 24, 48)
- **Trade-off**: More models, but cleaner evaluation

### 2. Single-Output LSTM (All Horizons)
- **Why**: End-to-end multi-task learning, more efficient
- **How**: Single LSTM outputs (batch, n_horizons)
- **Trade-off**: Different from LightGBM structure, but intentional for comparison

### 3. Rolling-Origin with Expanding Window
- **Why**: Mimics operational forecasting (can't retroactively retrain on future data)
- **How**: Fold 0: train 50%, test 10%; Fold 1: train 60%, test 10%; etc.
- **Validation**: Held-out test set in each fold, never reused for training

### 4. Leakage-Free Normalization
- **Why**: Future data must never influence past transformations
- **How**: Normalization params (mean, std) fitted on train only per fold
- **Implementation**: Each test set normalized using its fold's train statistics

### 5. Persistence as Explicit Baseline
- **Why**: Skill metric requires a reference model
- **How**: Persistence predicts y[t+h] = y[t] for all h
- **Validity**: No parameters, always leakage-free

---

## Data & Target Construction

### LightGBM Target (`y_train`)
Currently simplified: `y_train[t, h] = pm10[t]` (placeholder)

**In practice** (for full realism), should be:
```python
y_train[t, h] = pm10[t + h]  # Forward shift
```
This requires:
1. Aligning sequences with future targets
2. Removing rows where t+h exceeds series length
3. Careful indexing to avoid target leakage

### LSTM Target (`y_train`)
Similar structure as LightGBM (repeated target values for each horizon).

**Current limitation**: Dummy targets used. Real implementation should construct proper forward-shifted targets.

---

## Evaluation Leakage Checklist

✓ Preprocessing: train-only statistics  
✓ Feature engineering: lags, rolling only on [0, t]  
✓ Sequence generation: no future info in context  
✓ Train/test split: time-ordered, no overlap  
✓ Persistence: no parameters  
✓ Normalization: per-fold  
✓ No cross-fold mixing: each fold independent  

---

## Metric Definitions (Implemented)

### Skill
```
skill = 1 - rmse_model / rmse_persistence
```
- Positive: better than persistence
- Negative: worse than persistence (**ghost skill indicator**)

### Variance Percentage
```
var_pct = 100 * var_pred / var_obs
```
- 100%: full variance captured
- <100%: **variance collapse** (underdispersion, ghost skill symptom)
- >100%: variance expansion (overdispersion)

### Skill_VP (Diagnostic Metric)
```
skill_vp = skill * (var_pct / 100)
```
- Penalizes skill by variance collapse
- Not a replacement for RMSE, but a diagnostic flag
- Low Skill_VP + positive Skill = ghost skill present

### Event Metrics
```
recall = TP / (TP + FN)  # P(pred=event | obs=event)
```
- Measures ability to catch high PM10 events
- Thresholds: P75, P90 of training observations

---

## Configuration Management

All external parameters in `config/` subdirectory:

1. **config.yaml**: Main hyperparameters, paths
2. **horizons.yaml**: Prediction horizons (centralized)
3. **thresholds.yaml**: Event detection method and percentiles

Scripts load configs via:
```python
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
```

---

## Known Limitations & TODOs

### 1. Target Construction
Current: Placeholder targets (series[t] repeated for all horizons)  
Better: Proper forward-shifted targets (series[t+h])  
Impact: Results are proof-of-concept, not statistically valid for publication

### 2. LSTM Validation Split
Current: 15% of train, simple split  
Better: Time-ordered validation (last 15% of train fold)  
Impact: Minor, but cleaner for temporal data

### 3. Hyperparameter Selection
Current: Fixed values in config  
Note: Paper A is diagnostic, not optimized. Values chosen for stability.

### 4. Error Handling
Current: Minimal (assumes clean data)  
Better: Robust missing value handling, validation of file formats  
Impact: Production-ready code would be more defensive

### 5. Reproducibility
Current: Random seeds set, but LSTM may vary with GPU/TF nondeterminism  
Better: Force CPU-only mode or use explicit seeds in TF graph  
Impact: Minor variations expected on re-run

---

## Testing & Validation

**Quick smoke test** (5 min):
```bash
# Use subset of data (first 1000 rows) to verify pipeline works
python src/data/load_data.py
python src/data/preprocess_pm10.py
python src/data/make_features_lgbm.py
python src/data/make_splits.py
python src/training/train_persistence.py
python src/evaluation/compute_metrics_by_horizon.py
# Check: outputs/metrics/metrics_rolling_origin_by_horizon.csv created
```

**Full pipeline** (30-60 min, depending on data size):
```bash
bash run_paper_a.sh
# Check: all 4 output files exist in outputs/
```

---

## Future Extensions (Not in Paper A)

- [ ] Paper B: Explainability (SHAP, attention visualization)
- [ ] Paper C: Recalibration (temperature scaling, conformity intervals)
- [ ] Multi-station extension
- [ ] Additional pollutants (PM2.5, O3, NO2)
- [ ] Exogenous variables (weather, time-of-day patterns)
- [ ] Seasonal/weekly decomposition
- [ ] Physics-informed neural networks (PINNs)

---

## File Manifest

### Core Scripts (18 total)
- **Data (5)**: load, preprocess, features, sequences, splits
- **Models (3)**: persistence, lgbm, lstm
- **Training (3)**: train_persistence, train_lgbm, train_lstm
- **Evaluation (5)**: metrics, events, protocols, canonical table, visualize
- **Plotting (1)**: master figure

### Configuration (3 files)
- config.yaml, horizons.yaml, thresholds.yaml

### Documentation (2 files)
- README.md, IMPLEMENTATION_NOTES.md (this file)

---

## Debugging Tips

**Predictions not generated?**
- Check: `ls outputs/predictions/*.parquet`
- Verify: training scripts ran without error
- Debug: `python -c "import lightgbm; print(lightgbm.__version__)"`

**Metrics compute fails?**
- Check: predictions exist and have correct schema
- Verify: column names (model, h, y_pred, sample_idx)
- Debug: `pandas.read_parquet("outputs/predictions/*.parquet").info()`

**LSTM training slow?**
- Expected: 5-10 min per fold depending on data size
- Check GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Reduce epochs in config if needed (trades accuracy for speed)

---

## References (Implicit)

- Rolling-origin evaluation: Hyndman & Athanasopoulos (forecasting textbook)
- Ghost skill concept: Multi-horizon forecasting diagnostics
- PM10 context: Air quality monitoring standards

---

Last updated: 2026-04-11
