#!/bin/bash
set -e
STATIONS_DIR="data/raw"
METRICS_DIR="outputs/metrics"
TABLES_DIR="outputs/tables"
mkdir -p "$METRICS_DIR" "$TABLES_DIR"

# Existing 3 stations (already have results — skip if CSVs exist)
# New stations: all pm10_*.csv files except the original 3
for f in "$STATIONS_DIR"/pm10_*.csv; do
    name=$(basename "$f" .csv)          # e.g. pm10_28079004_10_M
    station="${name#pm10_}"             # e.g. 28079004_10_M
    pred_out="$METRICS_DIR/predictions_${station}.csv"
    skill_out="$METRICS_DIR/skill_${station}.csv"
    vr_out="$TABLES_DIR/variance_retention_${station}.csv"

    if [ -f "$vr_out" ]; then
        echo "SKIP $station (already done)"
        continue
    fi

    echo "=== Running $station ==="
    python3 scripts/01_generate_e1_rr_lags_only_predictions.py \
        --input "$f" \
        --dataset "e1_rr_${station}" \
        --predictions-output "$pred_out" \
        --skill-output "$skill_out" \
        --n-jobs 4

    python3 -c "
import pandas as pd
from src.diagnostics.variance import build_variance_retention_summary
pred  = pd.read_csv('$pred_out')
skill = pd.read_csv('$skill_out')
summ  = build_variance_retention_summary(pred, skill)
summ.to_csv('$vr_out', index=False)
print('Wrote $vr_out')
"
done
echo "=== All stations done ==="
