#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
START_YEAR="${START_YEAR:-2020}"
END_YEAR="${END_YEAR:-2024}"
N_JOBS="${N_JOBS:-4}"
SARIMA_JOBS="${SARIMA_JOBS:-2}"
SARIMA_ORIGIN_STEP="${SARIMA_ORIGIN_STEP:-14}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for csv_file in "$DATA_DIR"/pm10_*.csv; do
  station="$(basename "$csv_file" .csv)"
  station="${station#pm10_}"
  varret_stem="$station"
  if [[ "$station" == "daily" ]]; then
    varret_stem="summary"
  fi
  echo "=== $station ==="

  "$PYTHON_BIN" scripts/01_generate_e1_rr_lags_only_predictions.py \
    --input "$csv_file" \
    --dataset "e1_rr_${station}" \
    --start-year "$START_YEAR" --end-year "$END_YEAR" \
    --n-jobs "$N_JOBS" \
    --sarima-origin-step 0 \
    --predictions-output "outputs/metrics/predictions_base_${station}.csv" \
    --skill-output "outputs/metrics/skill_base_${station}.csv"

  "$PYTHON_BIN" scripts/02_generate_sarima_predictions.py \
    --input "$csv_file" \
    --dataset "e1_rr_${station}" \
    --origin-step "$SARIMA_ORIGIN_STEP" \
    --n-jobs "$SARIMA_JOBS" \
    --predictions-output "outputs/metrics/predictions_sarima_${station}.csv" \
    --skill-output "outputs/metrics/skill_sarima_${station}.csv"

  "$PYTHON_BIN" scripts/combine_prediction_tables.py \
    --base-predictions "outputs/metrics/predictions_base_${station}.csv" \
    --base-skill "outputs/metrics/skill_base_${station}.csv" \
    --sarima-predictions "outputs/metrics/predictions_sarima_${station}.csv" \
    --sarima-skill "outputs/metrics/skill_sarima_${station}.csv" \
    --predictions-output "outputs/metrics/predictions_${station}.csv" \
    --skill-output "outputs/metrics/skill_${station}.csv"

  "$PYTHON_BIN" scripts/07_build_variance_retention_table.py \
    --predictions "outputs/metrics/predictions_${station}.csv" \
    --skill "outputs/metrics/skill_${station}.csv" \
    --output "outputs/tables/variance_retention_${varret_stem}.csv" \
    --station-name "$station"

  "$PYTHON_BIN" scripts/03_exceedance_analysis.py \
    --predictions "outputs/metrics/predictions_${station}.csv" \
    --output "outputs/metrics/exceedance_${station}.csv" \
    --station "$station"

  "$PYTHON_BIN" scripts/05_dm_significance.py \
    --predictions "outputs/metrics/predictions_${station}.csv" \
    --output "outputs/metrics/dm_${station}.csv" \
    --station "$station"

  echo "Done: $station"
done

"$PYTHON_BIN" scripts/build_unified_variance_table.py
"$PYTHON_BIN" scripts/build_unified_predictions_table.py
"$PYTHON_BIN" scripts/04_build_unified_exceedance_table.py
"$PYTHON_BIN" scripts/build_unified_dm_table.py
"$PYTHON_BIN" scripts/06_threshold_sensitivity.py
"$PYTHON_BIN" scripts/07_murphy_decomposition.py \
  --predictions outputs/metrics/predictions_all_stations.csv \
  --output outputs/tables/murphy_decomposition_all_stations.csv
"$PYTHON_BIN" scripts/08_concentration_scale.py \
  --predictions outputs/metrics/predictions_all_stations.csv \
  --output outputs/tables/concentration_scale_summary.csv
"$PYTHON_BIN" scripts/09_build_comprehensive_unified_table.py
"$PYTHON_BIN" scripts/10_generate_all_figures.py

echo "=== PIPELINE COMPLETE ==="
