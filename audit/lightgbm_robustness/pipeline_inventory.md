# Pipeline Inventory — P4 LightGBM Robustness Arm
## Status: BLOCKED_BY_PIPELINE_OR_DATA
### Reason: Raw daily PM10 time series unavailable for 14 of 17 stations

The rolling-origin protocol requires the FULL historical series (from 2017) for each station to train models at each expanding window origin. Only 3 stations have raw data. The predictions_all_stations.csv release asset (evaluation period only) cannot substitute for the full historical training data.

| Componente | Fichero | Función/clase | Configuración | Evidencia | Estado |
| --- | --- | --- | --- | --- | --- |
| Registro de modelos | scripts/01_generate_e1_rr_lags_only_predictions.py | _fit_predict_models() | Ridge(alpha=1.0), HGB(max_iter=100, lr=0.05, max_leaf_nodes=15, seed=42) | 54b1ab1363a7... | CONFIRMED |
| hgb_direct | scripts/01_generate_e1_rr_lags_only_predictions.py | _fit_predict_models() → hgb variable | HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=42) | line 87 | CONFIRMED |
| ridge_direct | scripts/01_generate_e1_rr_lags_only_predictions.py | _fit_predict_models() → ridge variable | make_pipeline(StandardScaler(), Ridge(alpha=1.0)) | line 85 | CONFIRMED |
| stl_ridge_direct | scripts/01_generate_e1_rr_lags_only_predictions.py | _fit_predict_models() → stl_ridge | STLRidgeForecaster(season_length=7, ridge_alpha=1.0, n_lags=7) | line 93 | CONFIRMED |
| seasonal_naive | scripts/01_generate_e1_rr_lags_only_predictions.py | _fit_predict_models() → sp | SeasonalPersistenceModel(season_length=7) | line 89 | CONFIRMED |
| sarima | scripts/02_generate_sarima_predictions.py | SarimaForecaster | order=(1,0,1), seasonal_order=(1,0,1,7) | 7b0c7736458d... | CONFIRMED |
| rolling_origin | scripts/01_generate_e1_rr_lags_only_predictions.py | _generate_predictions_for_horizon() | Expanding window: train=all data < origin; origin_step=1 (every observed day) | MIN_TRAIN_ROWS=365 | CONFIRMED |
| feature_construction | scripts/01_generate_e1_rr_lags_only_predictions.py | _make_supervised() | LAGS=(0,1,2,3,4,5,6), target=pm10.shift(-horizon) | line 66-73 | CONFIRMED |
| imputación | scripts/01_generate_e1_rr_lags_only_predictions.py | _load_daily_pm10() | No imputation in _make_supervised; dropna on feature+target cols before train | line 115: complete = supervised.dropna(...) | CONFIRMED |
| escalado | scripts/01_generate_e1_rr_lags_only_predictions.py | _fit_predict_models() | StandardScaler inside pipeline for Ridge only; HGB no scaling | line 85 | CONFIRMED |
| horizons | scripts/01_generate_e1_rr_lags_only_predictions.py | HORIZONS constant | tuple(range(1, 8)) = (1,2,3,4,5,6,7) | line 41 | CONFIRMED |
| skill | scripts/01_generate_e1_rr_lags_only_predictions.py | _build_skill_summary() | skill = 1 - RMSE(model)/RMSE(persistence); inner join on (origin_date, date) | lines 155-175 | CONFIRMED |
| alpha (variance ratio) | scripts/07_build_variance_retention_table.py | compute_variance_retention() | alpha = Var(y_pred)/Var(y_true), ddof=0 or ddof=1 (per station script) | 30d6023b9b12... | CONFIRMED |
| recall_p75 | scripts/03_exceedance_analysis.py | compute_exceedance_metrics() | threshold=train_p75; recall=TP/actual_positives | a3a4ef80f87e... | CONFIRMED |
| DM-HLN | scripts/05_dm_significance.py | dm_test() | HLN-corrected Diebold-Mariano; BH FDR correction | 254aa30c3505... | CONFIRMED |
| master_diagnostic_table | scripts/09_build_comprehensive_unified_table.py | build_comprehensive_table() | merges variance_retention + exceedance + DM + station_metadata | 04e4a4942dff... | CONFIRMED |
| decision_change_analysis | scripts/15_decision_change_analysis.py | main() | Rule A: skill>0 AND dm_significant; Rule B: Rule A AND alpha>=0.50 AND recall_p75>=0.20 | d49ee8cb3b41... | CONFIRMED |
| raw_station_data | data/raw/pm10_*.csv | N/A | 3 of 17 stations available (pm10_daily, pm10_valencia_vivers, pm10_zarra_emep) | ls data/raw/ | BLOCKED: 14/17 stations missing raw data |

## Data Availability

| Station ID | Dataset | Raw File | Available |
| --- | --- | --- | --- |
| 03014002_10_M | e1_rr_daily | data/raw/pm10_daily.csv | ✓ AVAILABLE |
| 46250043_10_M | e1_rr_valencia_vivers | data/raw/pm10_valencia_vivers.csv | ✓ AVAILABLE |
| 46263999_10_M | e1_rr_zarra_emep | data/raw/pm10_zarra_emep.csv | ✓ AVAILABLE |
| 08019004_10_M | e1_rr_08019004_10_M | data/raw/pm10_08019004_10_M.csv | ✗ MISSING |
| 08019028_10_M | e1_rr_08019028_10_M | data/raw/pm10_08019028_10_M.csv | ✗ MISSING |
| 08019043_10_M | e1_rr_08019043_10_M | data/raw/pm10_08019043_10_M.csv | ✗ MISSING |
| 08019045_10_M | e1_rr_08019045_10_M | data/raw/pm10_08019045_10_M.csv | ✗ MISSING |
| 08019052_10_M | e1_rr_08019052_10_M | data/raw/pm10_08019052_10_M.csv | ✗ MISSING |
| 08019054_10_M | e1_rr_08019054_10_M | data/raw/pm10_08019054_10_M.csv | ✗ MISSING |
| 08263007_10_M | e1_rr_08263007_10_M | data/raw/pm10_08263007_10_M.csv | ✗ MISSING |
| 22125001_10_M | e1_rr_22125001_10_M | data/raw/pm10_22125001_10_M.csv | ✗ MISSING |
| 43004005_10_M | e1_rr_43004005_10_M | data/raw/pm10_43004005_10_M.csv | ✗ MISSING |
| 43004006_10_M | e1_rr_43004006_10_M | data/raw/pm10_43004006_10_M.csv | ✗ MISSING |
| 44013007_10_M | e1_rr_44013007_10_M | data/raw/pm10_44013007_10_M.csv | ✗ MISSING |
| 44216001_10_M | e1_rr_44216001_10_M | data/raw/pm10_44216001_10_M.csv | ✗ MISSING |
| 45153999_10_M | e1_rr_45153999_10_M | data/raw/pm10_45153999_10_M.csv | ✗ MISSING |
| 50008001_10_M | e1_rr_50008001_10_M | data/raw/pm10_50008001_10_M.csv | ✗ MISSING |
