#!/bin/bash
# Paper A: Full pipeline execution script
# Runs all steps in order: data prep → training → evaluation → plotting

set -e  # Exit on error

echo "=========================================="
echo "Paper A: Ghost Skill & Variance Collapse"
echo "Full Pipeline Execution"
echo "=========================================="
echo ""

CONFIG="${1:-config/config.yaml}"

# Phase 1: Data preparation
echo "[1/12] Loading raw data..."
python src/data/load_data.py --config "$CONFIG"

echo "[2/12] Preprocessing..."
python src/data/preprocess_pm10.py --config "$CONFIG"

echo "[3/12] Creating LightGBM features..."
python src/data/make_features_lgbm.py --config "$CONFIG"

echo "[4/12] Creating LSTM sequences..."
python src/data/make_sequences_lstm.py --config "$CONFIG"

echo "[5/12] Creating splits (rolling-origin + holdout)..."
python src/data/make_splits.py --config "$CONFIG"

# Phase 2: Training
echo "[6/12] Training persistence baseline..."
python src/training/train_persistence.py --config "$CONFIG"

echo "[7/12] Training LightGBM..."
python src/training/train_lgbm.py --config "$CONFIG"

echo "[8/12] Training LSTM..."
python src/training/train_lstm.py --config "$CONFIG"

# Phase 3: Metrics computation
echo "[9/12] Computing metrics by horizon..."
python src/evaluation/compute_metrics_by_horizon.py --config "$CONFIG"

echo "[10/12] Computing event metrics..."
python src/evaluation/compute_event_metrics.py --config "$CONFIG"

# Phase 4: Analysis
echo "[11/12] Comparing protocols..."
python src/evaluation/compare_protocols.py --config "$CONFIG"

echo "[Building canonical table..."
python src/evaluation/build_canonical_table.py --config "$CONFIG"

# Phase 5: Visualization
echo "[12/12] Plotting master figure..."
python src/plotting/plot_master_figure.py --config "$CONFIG"

echo ""
echo "=========================================="
echo "Paper A Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Canonical table:     outputs/tables/table_canonical_full.csv"
echo "  - Protocol comparison: outputs/metrics/metrics_protocol_comparison.csv"
echo "  - Event metrics:       outputs/metrics/metrics_events_full.csv"
echo "  - Master figure:       outputs/figures/master_figure.png"
echo ""
echo "Review results:"
echo "  cat outputs/tables/table_canonical_full.csv"
echo "  open outputs/figures/master_figure.png  # macOS"
echo ""
