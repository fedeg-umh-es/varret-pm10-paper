"""
Train Neural Network models (sklearn MLPRegressor as LSTM proxy)
================================================================
Trains multi-horizon neural network on rolling-origin and holdout.
Uses sklearn MLPRegressor as a lightweight alternative to LSTM.
Fixed version: uses actual feature indices from splits.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def train_rolling_origin(features: pd.DataFrame, splits: dict, 
                         horizons: list, cfg: dict) -> pd.DataFrame:
    """Train NN on rolling-origin with forward-shifted targets."""
    
    predictions_list = []
    obs = features['pm10'].values
    
    # Get feature columns (exclude pm10)
    feature_cols = [c for c in features.columns if c != 'pm10']
    X = features[feature_cols].values
    
    nn_cfg = cfg['models']['lstm'].copy()
    epochs = nn_cfg.pop('epochs', 50)
    batch_size = nn_cfg.pop('batch_size', 32)
    validation_split = nn_cfg.pop('validation_split', 0.15)
    random_state = nn_cfg.pop('random_state', 42)
    
    for fold_info in splits['folds']:
        fold = fold_info['fold']
        train_idx = np.array(fold_info['train_idx'])
        test_idx = np.array(fold_info['test_idx'])
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        # Build targets: forward-shifted obs[t+h] for each horizon
        y_train_dict = {}
        y_test_dict = {}
        
        for h in horizons:
            y_train_h = []
            for idx in train_idx:
                if idx + h < len(obs):
                    y_train_h.append(obs[idx + h])
                else:
                    y_train_h.append(obs[-1])
            y_train_dict[h] = np.array(y_train_h)
            
            y_test_h = []
            for idx in test_idx:
                if idx + h < len(obs):
                    y_test_h.append(obs[idx + h])
                else:
                    y_test_h.append(obs[-1])
            y_test_dict[h] = np.array(y_test_h)
        
        y_train = np.column_stack([y_train_dict[h] for h in horizons])
        y_test = np.column_stack([y_test_dict[h] for h in horizons])
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=epochs,
            batch_size=batch_size,
            learning_rate_init=0.001,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=validation_split,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Store predictions
        for i, idx in enumerate(test_idx):
            for h_idx, h in enumerate(horizons):
                predictions_list.append({
                    'fold': fold,
                    'sample_idx': int(idx),
                    'horizon': h,
                    'y_pred': float(y_pred[i, h_idx])
                })
    
    return pd.DataFrame(predictions_list)


def train_holdout(features: pd.DataFrame, splits: dict, horizons: list, cfg: dict) -> pd.DataFrame:
    """Train NN on holdout with forward-shifted targets."""
    
    train_idx = np.array(splits['train_idx'])
    test_idx = np.array(splits['test_idx'])
    
    obs = features['pm10'].values
    feature_cols = [c for c in features.columns if c != 'pm10']
    X = features[feature_cols].values
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    # Build targets
    y_train_dict = {}
    y_test_dict = {}
    
    for h in horizons:
        y_train_h = []
        for idx in train_idx:
            if idx + h < len(obs):
                y_train_h.append(obs[idx + h])
            else:
                y_train_h.append(obs[-1])
        y_train_dict[h] = np.array(y_train_h)
        
        y_test_h = []
        for idx in test_idx:
            if idx + h < len(obs):
                y_test_h.append(obs[idx + h])
            else:
                y_test_h.append(obs[-1])
        y_test_dict[h] = np.array(y_test_h)
    
    y_train = np.column_stack([y_train_dict[h] for h in horizons])
    y_test = np.column_stack([y_test_dict[h] for h in horizons])
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    nn_cfg = cfg['models']['lstm'].copy()
    epochs = nn_cfg.pop('epochs', 50)
    batch_size = nn_cfg.pop('batch_size', 32)
    validation_split = nn_cfg.pop('validation_split', 0.15)
    random_state = nn_cfg.pop('random_state', 42)
    
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=epochs,
        batch_size=batch_size,
        learning_rate_init=0.001,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=validation_split,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Store
    predictions_list = []
    for i, idx in enumerate(test_idx):
        for h_idx, h in enumerate(horizons):
            predictions_list.append({
                'sample_idx': int(idx),
                'horizon': h,
                'y_pred': float(y_pred[i, h_idx])
            })
    
    return pd.DataFrame(predictions_list)


def main():
    """Train NN."""
    parser = argparse.ArgumentParser(description="Train Neural Network models")
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
    print("\nTraining NN (rolling-origin)...")
    pred_ro = train_rolling_origin(features, splits_ro, horizons, cfg)
    if not pred_ro.empty:
        pred_ro_path = pred_dir / "nn_rolling_origin.parquet"
        pred_ro.to_parquet(pred_ro_path)
        print(f"Saved to {pred_ro_path}")
        print(f"Shape: {pred_ro.shape}")
    
    # Holdout
    print("Training NN (holdout)...")
    pred_ho = train_holdout(features, splits_ho, horizons, cfg)
    if not pred_ho.empty:
        pred_ho_path = pred_dir / "nn_holdout.parquet"
        pred_ho.to_parquet(pred_ho_path)
        print(f"Saved to {pred_ho_path}")
        print(f"Shape: {pred_ho.shape}")


if __name__ == "__main__":
    main()
