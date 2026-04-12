"""
Train persistence baseline
==========================
Generates persistence predictions for rolling-origin and holdout.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.persistence import PersistenceModel


def train_rolling_origin(features: pd.DataFrame, splits: dict, horizons: list) -> pd.DataFrame:
    """Generate persistence predictions for rolling-origin."""
    predictions_list = []
    
    for fold_info in splits['folds']:
        fold = fold_info['fold']
        test_idx = fold_info['test_idx']
        
        X_test = features.iloc[test_idx]
        
        model = PersistenceModel(horizons=horizons)
        model.fit(None, None)
        y_pred = model.predict(X_test.values)
        
        # Store with fold and index info
        for i, idx in enumerate(test_idx):
            for h_idx, h in enumerate(horizons):
                predictions_list.append({
                    'fold': fold,
                    'sample_idx': idx,
                    'horizon': h,
                    'y_pred': y_pred[i, h_idx]
                })
    
    return pd.DataFrame(predictions_list)


def train_holdout(features: pd.DataFrame, splits: dict, horizons: list) -> pd.DataFrame:
    """Generate persistence predictions for holdout."""
    test_idx = splits['test_idx']
    X_test = features.iloc[test_idx]
    
    model = PersistenceModel(horizons=horizons)
    model.fit(None, None)
    y_pred = model.predict(X_test.values)
    
    predictions_list = []
    for i, idx in enumerate(test_idx):
        for h_idx, h in enumerate(horizons):
            predictions_list.append({
                'sample_idx': idx,
                'horizon': h,
                'y_pred': y_pred[i, h_idx]
            })
    
    return pd.DataFrame(predictions_list)


def main():
    """Train persistence."""
    parser = argparse.ArgumentParser(description="Train persistence baseline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Load horizons
    with open("config/horizons.yaml") as f:
        horizons_cfg = yaml.safe_load(f)
    horizons = horizons_cfg['horizons']
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features and splits
    print("Loading features...")
    features = pd.read_parquet(processed_dir / "features_lgbm.parquet")
    
    print("Loading rolling-origin splits...")
    with open(processed_dir / "splits_rolling_origin.json") as f:
        splits_ro = json.load(f)
    
    print("Loading holdout split...")
    with open(processed_dir / "splits_holdout.json") as f:
        splits_ho = json.load(f)
    
    # Rolling-origin
    print("\nGenerating persistence predictions (rolling-origin)...")
    pred_ro = train_rolling_origin(features, splits_ro, horizons)
    pred_ro_path = pred_dir / "persistence_rolling_origin.parquet"
    pred_ro.to_parquet(pred_ro_path)
    print(f"Saved to {pred_ro_path} ({len(pred_ro)} records)")
    
    # Holdout
    print("Generating persistence predictions (holdout)...")
    pred_ho = train_holdout(features, splits_ho, horizons)
    pred_ho_path = pred_dir / "persistence_holdout.parquet"
    pred_ho.to_parquet(pred_ho_path)
    print(f"Saved to {pred_ho_path} ({len(pred_ho)} records)")


if __name__ == "__main__":
    main()
