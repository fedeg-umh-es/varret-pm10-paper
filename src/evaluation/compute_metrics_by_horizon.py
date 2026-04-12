"""
Compute metrics by horizon
==========================
RMSE, skill, variance, Skill_VP per (model, horizon) pair.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_persist: np.ndarray = None) -> dict:
    """
    Compute diagnostic metrics for horizon h.
    
    Args:
        y_true: observed values
        y_pred: model predictions
        y_persist: persistence predictions (for skill baseline)
    
    Returns:
        dict with metrics
    """
    # RMSE
    rmse_model = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Persistence baseline
    if y_persist is not None:
        rmse_persist = np.sqrt(np.mean((y_persist - y_true) ** 2))
    else:
        # If no persistence provided, use last value
        rmse_persist = rmse_model  # skill will be 0
    
    # Skill
    if rmse_persist > 0:
        skill = 1 - (rmse_model / rmse_persist)
    else:
        skill = 0
    
    # Variance
    var_obs = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Variance percentage
    if var_obs > 0:
        var_pct = 100 * var_pred / var_obs
    else:
        var_pct = 0
    
    # Skill_VP: skill penalized by variance collapse
    skill_vp = skill * (var_pct / 100)
    
    return {
        'rmse_model': rmse_model,
        'rmse_persist': rmse_persist,
        'skill': skill,
        'var_obs': var_obs,
        'var_pred': var_pred,
        'var_pct': var_pct,
        'skill_vp': skill_vp
    }


def main():
    """Compute metrics for rolling-origin and holdout."""
    parser = argparse.ArgumentParser(description="Compute metrics by horizon")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    with open("config/horizons.yaml") as f:
        horizons_cfg = yaml.safe_load(f)
    horizons = horizons_cfg['horizons']
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])
    metrics_dir = Path(cfg['paths']['metrics_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load observations
    print("Loading observations...")
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    observations = df_pre['pm10_normalized'].values
    
    # Load features to align samples
    features = pd.read_parquet(processed_dir / "features_lgbm.parquet")
    feature_indices = features.index.get_indexer(df_pre.index[df_pre.index.isin(features.index)])
    
    # Rolling-origin
    print("\nComputing metrics (rolling-origin)...")
    with open(processed_dir / "splits_rolling_origin.json") as f:
        splits_ro = json.load(f)
    
    pred_persist_ro = pd.read_parquet(pred_dir / "persistence_rolling_origin.parquet")
    pred_lgbm_ro = pd.read_parquet(pred_dir / "lgbm_rolling_origin.parquet")

    try:
        pred_lstm_ro = pd.read_parquet(pred_dir / "lstm_rolling_origin.parquet")
    except Exception:
        pred_lstm_ro = None

    try:
        pred_sarima_ro = pd.read_parquet(pred_dir / "sarima_rolling_origin.parquet")
    except Exception:
        pred_sarima_ro = None

    metrics_ro = []
    for h in horizons:
        # Filter predictions for this horizon
        y_persist = pred_persist_ro[pred_persist_ro['horizon'] == h]['y_pred'].values
        y_lgbm = pred_lgbm_ro[pred_lgbm_ro['horizon'] == h]['y_pred'].values

        # Get indices
        indices = pred_persist_ro[pred_persist_ro['horizon'] == h]['sample_idx'].values
        y_true = observations[indices]

        # Compute metrics
        metrics_lgbm = compute_metrics(y_true, y_lgbm, y_persist)

        metrics_ro.append({
            'model': 'lgbm',
            'h': h,
            'rmse': metrics_lgbm['rmse_model'],
            'rmse_persistence': metrics_lgbm['rmse_persist'],
            'skill': metrics_lgbm['skill'],
            'var_obs': metrics_lgbm['var_obs'],
            'var_pred': metrics_lgbm['var_pred'],
            'var_pct': metrics_lgbm['var_pct'],
            'skill_vp': metrics_lgbm['skill_vp']
        })

        # LSTM
        if pred_lstm_ro is not None:
            y_lstm = pred_lstm_ro[pred_lstm_ro['horizon'] == h]['y_pred'].values
            if len(y_lstm) == len(y_true):
                metrics_lstm = compute_metrics(y_true, y_lstm, y_persist)
                metrics_ro.append({
                    'model': 'lstm',
                    'h': h,
                    'rmse': metrics_lstm['rmse_model'],
                    'rmse_persistence': metrics_lstm['rmse_persist'],
                    'skill': metrics_lstm['skill'],
                    'var_obs': metrics_lstm['var_obs'],
                    'var_pred': metrics_lstm['var_pred'],
                    'var_pct': metrics_lstm['var_pct'],
                    'skill_vp': metrics_lstm['skill_vp']
                })

        # SARIMA
        if pred_sarima_ro is not None:
            y_sarima = pred_sarima_ro[pred_sarima_ro['horizon'] == h]['y_pred'].values
            if len(y_sarima) == len(y_true):
                metrics_sarima = compute_metrics(y_true, y_sarima, y_persist)
                metrics_ro.append({
                    'model': 'sarima',
                    'h': h,
                    'rmse': metrics_sarima['rmse_model'],
                    'rmse_persistence': metrics_sarima['rmse_persist'],
                    'skill': metrics_sarima['skill'],
                    'var_obs': metrics_sarima['var_obs'],
                    'var_pred': metrics_sarima['var_pred'],
                    'var_pct': metrics_sarima['var_pct'],
                    'skill_vp': metrics_sarima['skill_vp']
                })
    
    df_metrics_ro = pd.DataFrame(metrics_ro)
    ro_path = metrics_dir / "metrics_rolling_origin_by_horizon.csv"
    df_metrics_ro.to_csv(ro_path, index=False)
    print(f"Saved to {ro_path}")
    print(df_metrics_ro)
    
    # Holdout
    print("\nComputing metrics (holdout)...")
    with open(processed_dir / "splits_holdout.json") as f:
        splits_ho = json.load(f)
    
    pred_persist_ho = pd.read_parquet(pred_dir / "persistence_holdout.parquet")
    pred_lgbm_ho = pd.read_parquet(pred_dir / "lgbm_holdout.parquet")

    try:
        pred_lstm_ho = pd.read_parquet(pred_dir / "lstm_holdout.parquet")
    except Exception:
        pred_lstm_ho = None

    try:
        pred_sarima_ho = pd.read_parquet(pred_dir / "sarima_holdout.parquet")
    except Exception:
        pred_sarima_ho = None

    metrics_ho = []
    for h in horizons:
        y_persist = pred_persist_ho[pred_persist_ho['horizon'] == h]['y_pred'].values
        y_lgbm = pred_lgbm_ho[pred_lgbm_ho['horizon'] == h]['y_pred'].values

        indices = pred_persist_ho[pred_persist_ho['horizon'] == h]['sample_idx'].values
        y_true = observations[indices]

        metrics_lgbm = compute_metrics(y_true, y_lgbm, y_persist)

        metrics_ho.append({
            'model': 'lgbm',
            'h': h,
            'rmse': metrics_lgbm['rmse_model'],
            'rmse_persistence': metrics_lgbm['rmse_persist'],
            'skill': metrics_lgbm['skill'],
            'var_obs': metrics_lgbm['var_obs'],
            'var_pred': metrics_lgbm['var_pred'],
            'var_pct': metrics_lgbm['var_pct'],
            'skill_vp': metrics_lgbm['skill_vp']
        })

        if pred_lstm_ho is not None:
            y_lstm = pred_lstm_ho[pred_lstm_ho['horizon'] == h]['y_pred'].values
            if len(y_lstm) == len(y_true):
                metrics_lstm = compute_metrics(y_true, y_lstm, y_persist)
                metrics_ho.append({
                    'model': 'lstm',
                    'h': h,
                    'rmse': metrics_lstm['rmse_model'],
                    'rmse_persistence': metrics_lstm['rmse_persist'],
                    'skill': metrics_lstm['skill'],
                    'var_obs': metrics_lstm['var_obs'],
                    'var_pred': metrics_lstm['var_pred'],
                    'var_pct': metrics_lstm['var_pct'],
                    'skill_vp': metrics_lstm['skill_vp']
                })

        if pred_sarima_ho is not None:
            y_sarima = pred_sarima_ho[pred_sarima_ho['horizon'] == h]['y_pred'].values
            if len(y_sarima) == len(y_true):
                metrics_sarima = compute_metrics(y_true, y_sarima, y_persist)
                metrics_ho.append({
                    'model': 'sarima',
                    'h': h,
                    'rmse': metrics_sarima['rmse_model'],
                    'rmse_persistence': metrics_sarima['rmse_persist'],
                    'skill': metrics_sarima['skill'],
                    'var_obs': metrics_sarima['var_obs'],
                    'var_pred': metrics_sarima['var_pred'],
                    'var_pct': metrics_sarima['var_pct'],
                    'skill_vp': metrics_sarima['skill_vp']
                })
    
    df_metrics_ho = pd.DataFrame(metrics_ho)
    ho_path = metrics_dir / "metrics_holdout_by_horizon.csv"
    df_metrics_ho.to_csv(ho_path, index=False)
    print(f"Saved to {ho_path}")
    print(df_metrics_ho)


if __name__ == "__main__":
    main()
