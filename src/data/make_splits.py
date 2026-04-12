"""
Create rolling-origin and holdout splits
========================================
Time-ordered, leakage-free splits.
"""

import json
from pathlib import Path
import argparse
import yaml


def create_rolling_origin_splits(n_samples: int, cfg: dict) -> list:
    """
    Create expanding window splits.
    
    Args:
        n_samples: total samples
        cfg: config dict with rolling_origin settings
    
    Returns:
        list of dicts with train_idx, test_idx
    """
    ro_cfg = cfg['splits']['rolling_origin']
    n_folds = ro_cfg['n_folds']
    initial_train_pct = ro_cfg['initial_train_size_pct']
    test_pct = ro_cfg['test_size_pct']
    
    initial_train = int(n_samples * initial_train_pct)
    test_size = int(n_samples * test_pct)
    
    folds = []
    for fold in range(n_folds):
        # Expand training set
        train_end = initial_train + fold * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end <= n_samples:
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, test_end))
            folds.append({
                'fold': fold,
                'train_idx': train_idx,
                'test_idx': test_idx
            })
    
    return folds


def create_holdout_split(n_samples: int, cfg: dict) -> dict:
    """
    Create chronological holdout split.
    
    Args:
        n_samples: total samples
        cfg: config dict
    
    Returns:
        dict with train_idx, test_idx
    """
    train_pct = cfg['splits']['holdout']['train_size_pct']
    split_idx = int(n_samples * train_pct)
    
    return {
        'train_idx': list(range(0, split_idx)),
        'test_idx': list(range(split_idx, n_samples))
    }


def main():
    """Create and save splits."""
    parser = argparse.ArgumentParser(description="Create train/test splits")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Determine n_samples from features
    from pathlib import Path
    import pandas as pd
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    df_features = pd.read_parquet(processed_dir / "features_lgbm.parquet")
    n_samples = len(df_features)
    
    print(f"Creating splits for {n_samples} samples...")
    
    # Rolling-origin
    print("\nRolling-origin splits:")
    ro_folds = create_rolling_origin_splits(n_samples, cfg)
    for fold_info in ro_folds:
        print(f"  Fold {fold_info['fold']}: train={len(fold_info['train_idx'])}, "
              f"test={len(fold_info['test_idx'])}")
    
    ro_path = processed_dir / "splits_rolling_origin.json"
    with open(ro_path, 'w') as f:
        json.dump({'folds': ro_folds}, f, indent=2)
    print(f"Saved to {ro_path}")
    
    # Holdout
    print("\nHoldout split:")
    holdout = create_holdout_split(n_samples, cfg)
    print(f"  train={len(holdout['train_idx'])}, test={len(holdout['test_idx'])}")
    
    ho_path = processed_dir / "splits_holdout.json"
    with open(ho_path, 'w') as f:
        json.dump(holdout, f, indent=2)
    print(f"Saved to {ho_path}")


if __name__ == "__main__":
    main()
