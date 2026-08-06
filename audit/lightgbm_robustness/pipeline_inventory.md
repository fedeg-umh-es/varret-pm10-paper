# Pipeline Inventory — P4 LightGBM Robustness Arm

| Componente | Fichero | Función/clase | Configuración | Evidencia | Estado |
| --- | --- | --- | --- | --- | --- |
| Tabular Features & Baselines | `scripts/01_generate_e1_rr_lags_only_predictions.py` | `_make_supervised`, `_fit_predict_models` | Lags 0..6, train_end < forecast_origin, 5 folds rolling-origin | `outputs/metrics/predictions.csv` | Verified |
| HistGradientBoosting (HGB) | `scripts/01_generate_e1_rr_lags_only_predictions.py` | `HistGradientBoostingRegressor` | `max_iter=100`, `learning_rate=0.05`, `max_leaf_nodes=15`, `random_state=42` | `master_diagnostic_table.csv` | Verified |
| Ridge Linear Direct | `scripts/01_generate_e1_rr_lags_only_predictions.py` | `Ridge` | `alpha=1.0`, `StandardScaler` | `master_diagnostic_table.csv` | Verified |
| STL-Ridge Direct | `src/models/stl_ridge.py` | `STLRidgeForecaster` | `season_length=7`, `ridge_alpha=1.0`, `n_lags=7` | `master_diagnostic_table.csv` | Verified |
| SARIMA Direct | `src/models/sarima_model.py` | `SarimaForecaster` | `order=(1,0,1)`, `seasonal_order=(1,0,1,7)` | `master_diagnostic_table.csv` | Verified |
| Seasonal Naive | `src/models/seasonal_persistence.py` | `SeasonalPersistenceModel` | `season_length=7` | `master_diagnostic_table.csv` | Verified |
| Persistence Baseline | `scripts/01_generate_e1_rr_lags_only_predictions.py` | Lag 0 persistence | `y_pred = y_origin` | `master_diagnostic_table.csv` | Verified |
| LightGBM Direct (New Arm) | `audit/lightgbm_robustness/generate_lightgbm_arm.py` | `lgb.LGBMRegressor` | `n_estimators=100`, `learning_rate=0.05`, `num_leaves=15`, `random_state=42`, `n_jobs=1`, `verbosity=-1`, `deterministic=True`, `force_col_wise=True` | `outputs/analysis/master_diagnostic_table_with_lightgbm.csv` | Pending Run |
| Diebold-Mariano Test | `scripts/05_dm_significance.py` | `compute_dm_test` | Loss=MSE vs persistence, HAC lag h-1, HLN correction, BH p-value adjustment < 0.05 | `outputs/tables/dm_significance_all_stations.csv` | Verified |
| Variance Retention (\alpha) | `scripts/07_build_variance_retention_table.py` | `alpha = Var(y_pred)/Var(y_true)` | Unadjusted variance ratio | `outputs/tables/variance_retention_all_stations.csv` | Verified |
| Episode Recall (p75) | `scripts/03_exceedance_analysis.py` | `recall_p75` | Event = y_true >= p75(train) | `outputs/tables/exceedance_all_stations.csv` | Verified |
| Master Table Construction | `scripts/09_build_comprehensive_unified_table.py` | `main` | Inner/left join on station_id x model x horizon | `master_diagnostic_table.csv` | Verified |
| Decision Analysis | `scripts/15_decision_change_analysis.py` | `Rule A vs Rule B` | Rule A: skill>0 & DM_sig; Rule B: Rule A & \alpha>=0.50 & recall_p75>=0.20 | `audit/decision_change/REPORT.md` | Verified |
