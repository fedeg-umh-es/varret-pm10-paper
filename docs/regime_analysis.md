# Regime-conditioned analysis

Run the additive TAC-oriented diagnostic layer with:

```bash
python3 scripts/09_run_regime_analysis.py \
  --protocol rolling_origin \
  --meteorology-source ../P1_PM10_Meteorology_Hstar/data_processed/madrid_pm10_meteorology_experiment_base.csv
```

If the external meteorology source is unavailable, the script degrades to the origin-time covariates already present in `data/processed/features_lgbm.parquet` and documents that fallback in the report.
