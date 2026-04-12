"""
Train LightGBM models
=====================
Trains LightGBM per horizon on rolling-origin and holdout.
CRITICAL: Targets are forward-shifted by horizon to avoid leakage.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lgbm_model import LGBMMultiHorizon


def train_rolling_origin(features: pd.DataFrame, splits: dict, horizons: list, cfg: dict) -> pd.DataFrame:
    """Train LightGBM on rolling-origin, generate predictions."""
    predictions_list = []
    models_dir = Path(cfg['paths']['models_dir']) / "lgbm_rolling_origin"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original preprocessed for targets
    processed_dir = Path(cfg['paths']['processed_dir'])
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    pm10_series = df_pre['pm10_normalized'].values
    
    for fold_info in splits['folds']:
        fold = fold_info['fold']
        train_idx = fold_info['train_idx']
        test_idx = fold_info['test_idx']
        
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        
        # Build targets: y[t, h] = pm10[t+h] (forward shifted)
        # Only keep samples where t+max(horizons) < len(pm10_series)
        y_train_data = []
        valid_train_idx = []
        
        for train_i, global_idx in enumerate(train_idx):
            # Get targets for this sample, forward-shifted by each horizon
            targets = []
            valid = True
            for h in horizons:
                if global_idx + h < len(pm10_series):
                    targets.append(pm10_series[global_idx + h])
                else:
                    valid = False
                    break
            
            if valid:
                y_train_data.append(targets)
                valid_train_idx.append(train_i)
        
        # Filter X_train to match valid samples
        X_train = X_train.iloc[valid_train_idx].reset_index(drop=True)
        y_train = pd.DataFrame(y_train_data, columns=[f'h{h}' for h in horizons])
        
        # Build test targets
        y_test_data = []
        valid_test_idx = []
        
        for test_i, global_idx in enumerate(test_idx):
            targets = []
            valid = True
            for h in horizons:
                if global_idx + h < len(pm10_series):
                    targets.append(pm10_series[global_idx + h])
                else:
                    valid = False
                    break
            
            if valid:
                y_test_data.append(targets)
                valid_test_idx.append(test_i)
        
        # Filter X_test
        X_test = X_test.iloc[valid_test_idx].reset_index(drop=True)
        y_test_true = np.array(y_test_data) if y_test_data else np.array([]).reshape(0, len(horizons))
        valid_test_idx_global = [test_idx[i] for i in valid_test_idx]
        
        # Train
        model = LGBMMultiHorizon(horizons, cfg['models']['lgbm'])
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Save model
        model.save(models_dir / f"fold_{fold}")
        
        # Store predictions with correct indices
        for i, global_idx in enumerate(valid_test_idx_global):
            for h_idx, h in enumerate(horizons):
                predictions_list.append({
                    'fold': fold,
                    'sample_idx': global_idx,
                    'horizon': h,
                    'y_pred': y_pred[i, h_idx]
                })
    
    return pd.DataFrame(predictions_list)


def train_holdout(features: pd.DataFrame, splits: dict, horizons: list, cfg: dict) -> pd.DataFrame:
    """Train LightGBM on holdout, generate predictions."""
    train_idx = splits['train_idx']
    test_idx = splits['test_idx']
    
    X_train = features.iloc[train_idx]
    X_test = features.iloc[test_idx]
    
    # Load targets
    processed_dir = Path(cfg['paths']['processed_dir'])
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    pm10_series = df_pre['pm10_normalized'].values
    
    # Build train targets
    y_train_data = []
    valid_train_idx = []
    
    for train_i, global_idx in enumerate(train_idx):
        targets = []
        valid = True
        for h in horizons:
            if global_idx + h < len(pm10_series):
                targets.append(pm10_series[global_idx + h])
            else:
                valid = False
                break
        
        if valid:
            y_train_data.append(targets)
            valid_train_idx.append(train_i)
    
    X_train = X_train.iloc[valid_train_idx].reset_index(drop=True)
    y_train = pd.DataFrame(y_train_data, columns=[f'h{h}' for h in horizons])
    
    # Build test targets
    y_test_data = []
    valid_test_idx = []
    
    for test_i, global_idx in enumerate(test_idx):
        targets = []
        valid = True
        for h in horizons:
            if global_idx + h < len(pm10_series):
                targets.append(pm10_series[global_idx + h])
            else:
                valid = False
                break
        
        if valid:
            y_test_data.append(targets)
            valid_test_idx.append(test_i)
    
    X_test = X_test.iloc[valid_test_idx].reset_index(drop=True)
    valid_test_idx_global = [test_idx[i] for i in valid_test_idx]
    
    # Train
    model = LGBMMultiHorizon(horizons, cfg['models']['lgbm'])
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Save model
    models_dir = Path(cfg['paths']['models_dir']) / "lgbm_holdout"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir)
    
    # Store predictions
    predictions_list = []
    for i, global_idx in enumerate(valid_test_idx_global):
        for h_idx, h in enumerate(horizons):
            predictions_list.append({
                'sample_idx': global_idx,
                'horizon': h,
                'y_pred': y_pred[i, h_idx]
            })
    
    return pd.DataFrame(predictions_list)


def main():
    """Train LightGBM."""
    parser = argparse.ArgumentParser(description="Train LightGBM models")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    with open("config/horizons.yaml") as f:
        horizons_cfg = yaml.safe_load(f)
    horizons = horizons_cfg['horizons']
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    print("Loading features...")
    features = pd.read_parquet(processed_dir / "features_lgbm.parquet")
    
    print("Loading splits...")
    with open(processed_dir / "splits_rolling_origin.json") as f:
        splits_ro = json.load(f)
    with open(processed_dir / "splits_holdout.json") as f:
        splits_ho = json.load(f)
    
    # Rolling-origin
    print("\nTraining LightGBM (rolling-origin)...")
    pred_ro = train_rolling_origin(features, splits_ro, horizons, cfg)
    pred_ro_path = pred_dir / "lgbm_rolling_origin.parquet"
    pred_ro.to_parquet(pred_ro_path)
    print(f"Saved to {pred_ro_path} ({len(pred_ro)} records)")
    
    # Holdout
    print("Training LightGBM (holdout)...")
    pred_ho = train_holdout(features, splits_ho, horizons, cfg)
    pred_ho_path = pred_dir / "lgbm_holdout.parquet"
    pred_ho.to_parquet(pred_ho_path)
    print(f"Saved to {pred_ho_path} ({len(pred_ho)} records)")


if __name__ == "__main__":
    main()
