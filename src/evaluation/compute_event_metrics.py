"""
Compute event metrics: exceedance recall
========================================
Evaluates model ability to predict high PM10 events.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         threshold: float) -> dict:
    """
    Compute event detection metrics.
    
    Args:
        y_true: observations
        y_pred: predictions
        threshold: exceedance threshold (value)
    
    Returns:
        dict with recall, precision, f1
    """
    event_true = y_true > threshold
    event_pred = y_pred > threshold
    
    tp = np.sum(event_true & event_pred)
    fn = np.sum(event_true & ~event_pred)
    fp = np.sum(~event_true & event_pred)
    
    # Recall: P(pred=event | true=event)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Precision: P(true=event | pred=event)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1
    }


def main():
    """Compute event metrics."""
    parser = argparse.ArgumentParser(description="Compute event metrics")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    with open("config/horizons.yaml") as f:
        horizons_cfg = yaml.safe_load(f)
    horizons = horizons_cfg['horizons']
    
    with open("config/thresholds.yaml") as f:
        thresholds_cfg = yaml.safe_load(f)
    percentiles = thresholds_cfg['event_detection']['percentiles']
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])
    metrics_dir = Path(cfg['paths']['metrics_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load observations
    print("Loading observations...")
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    observations = df_pre['pm10_normalized'].values
    
    # Compute percentile-based thresholds from training data
    with open(processed_dir / "splits_rolling_origin.json") as f:
        splits_ro = json.load(f)
    train_idx = splits_ro['folds'][0]['train_idx']
    obs_train = observations[train_idx]
    
    thresholds = {p: np.percentile(obs_train, p) for p in percentiles}
    print(f"Thresholds (percentiles): {thresholds}")
    
    # Rolling-origin
    print("\nComputing event metrics (rolling-origin)...")
    pred_lgbm_ro = pd.read_parquet(pred_dir / "lgbm_rolling_origin.parquet")
    
    try:
        pred_lstm_ro = pd.read_parquet(pred_dir / "lstm_rolling_origin.parquet")
    except:
        pred_lstm_ro = None
    
    # Also compute skill/var for events
    pred_persist_ro = pd.read_parquet(pred_dir / "persistence_rolling_origin.parquet")
    
    metrics_ro = []
    for h in horizons:
        for p in percentiles:
            threshold = thresholds[p]
            
            # LightGBM
            y_lgbm = pred_lgbm_ro[pred_lgbm_ro['horizon'] == h]['y_pred'].values
            y_persist = pred_persist_ro[pred_persist_ro['horizon'] == h]['y_pred'].values
            
            indices = pred_lgbm_ro[pred_lgbm_ro['horizon'] == h]['sample_idx'].values
            y_true = observations[indices]
            
            event_metrics = compute_event_metrics(y_true, y_lgbm, threshold)
            
            # Also compute skill and var_pct for event context
            rmse_model = np.sqrt(np.mean((y_lgbm - y_true) ** 2))
            rmse_persist = np.sqrt(np.mean((y_persist - y_true) ** 2))
            skill = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0
            var_pct = 100 * np.var(y_lgbm) / np.var(y_true) if np.var(y_true) > 0 else 0
            skill_vp = skill * (var_pct / 100)
            
            metrics_ro.append({
                'model': 'lgbm',
                'h': h,
                'threshold': p,
                'recall_events': event_metrics['recall'],
                'skill': skill,
                'var_pct': var_pct,
                'skill_vp': skill_vp
            })
            
            # LSTM
            if pred_lstm_ro is not None:
                y_lstm = pred_lstm_ro[pred_lstm_ro['horizon'] == h]['y_pred'].values
                if len(y_lstm) == len(y_true):
                    event_metrics = compute_event_metrics(y_true, y_lstm, threshold)
                    
                    rmse_model = np.sqrt(np.mean((y_lstm - y_true) ** 2))
                    skill = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0
                    var_pct = 100 * np.var(y_lstm) / np.var(y_true) if np.var(y_true) > 0 else 0
                    skill_vp = skill * (var_pct / 100)
                    
                    metrics_ro.append({
                        'model': 'lstm',
                        'h': h,
                        'threshold': p,
                        'recall_events': event_metrics['recall'],
                        'skill': skill,
                        'var_pct': var_pct,
                        'skill_vp': skill_vp
                    })
    
    df_events_ro = pd.DataFrame(metrics_ro)
    ro_path = metrics_dir / "metrics_events_rolling_origin.csv"
    df_events_ro.to_csv(ro_path, index=False)
    print(f"Saved to {ro_path}")
    
    # Holdout (same structure)
    print("\nComputing event metrics (holdout)...")
    pred_lgbm_ho = pd.read_parquet(pred_dir / "lgbm_holdout.parquet")
    pred_persist_ho = pd.read_parquet(pred_dir / "persistence_holdout.parquet")
    
    try:
        pred_lstm_ho = pd.read_parquet(pred_dir / "lstm_holdout.parquet")
    except:
        pred_lstm_ho = None
    
    metrics_ho = []
    for h in horizons:
        for p in percentiles:
            threshold = thresholds[p]
            
            y_lgbm = pred_lgbm_ho[pred_lgbm_ho['horizon'] == h]['y_pred'].values
            y_persist = pred_persist_ho[pred_persist_ho['horizon'] == h]['y_pred'].values
            
            indices = pred_lgbm_ho[pred_lgbm_ho['horizon'] == h]['sample_idx'].values
            y_true = observations[indices]
            
            event_metrics = compute_event_metrics(y_true, y_lgbm, threshold)
            
            rmse_model = np.sqrt(np.mean((y_lgbm - y_true) ** 2))
            rmse_persist = np.sqrt(np.mean((y_persist - y_true) ** 2))
            skill = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0
            var_pct = 100 * np.var(y_lgbm) / np.var(y_true) if np.var(y_true) > 0 else 0
            skill_vp = skill * (var_pct / 100)
            
            metrics_ho.append({
                'model': 'lgbm',
                'h': h,
                'threshold': p,
                'recall_events': event_metrics['recall'],
                'skill': skill,
                'var_pct': var_pct,
                'skill_vp': skill_vp
            })
            
            if pred_lstm_ho is not None:
                y_lstm = pred_lstm_ho[pred_lstm_ho['horizon'] == h]['y_pred'].values
                if len(y_lstm) == len(y_true):
                    event_metrics = compute_event_metrics(y_true, y_lstm, threshold)
                    rmse_model = np.sqrt(np.mean((y_lstm - y_true) ** 2))
                    skill = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0
                    var_pct = 100 * np.var(y_lstm) / np.var(y_true) if np.var(y_true) > 0 else 0
                    skill_vp = skill * (var_pct / 100)
                    
                    metrics_ho.append({
                        'model': 'lstm',
                        'h': h,
                        'threshold': p,
                        'recall_events': event_metrics['recall'],
                        'skill': skill,
                        'var_pct': var_pct,
                        'skill_vp': skill_vp
                    })
    
    df_events_ho = pd.DataFrame(metrics_ho)
    ho_path = metrics_dir / "metrics_events_holdout.csv"
    df_events_ho.to_csv(ho_path, index=False)
    print(f"Saved to {ho_path}")
    
    # Combine into metrics_events_full.csv
    print("\nCombining event metrics...")
    df_events_ro['protocol'] = 'rolling_origin'
    df_events_ho['protocol'] = 'holdout'
    df_events_full = pd.concat([df_events_ro, df_events_ho], ignore_index=True)
    
    full_path = metrics_dir / "metrics_events_full.csv"
    df_events_full.to_csv(full_path, index=False)
    print(f"Saved combined events to {full_path}")


if __name__ == "__main__":
    main()
