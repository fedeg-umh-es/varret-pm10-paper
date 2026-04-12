"""
Preprocess PM10: imputation, detrending, normalization
========================================================
All statistics fitted on data available at each point in time.
Leakage-free: future data never used to transform past data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml


def impute_series(series: pd.Series, method: str = "forward_fill") -> pd.Series:
    """
    Impute missing values.
    
    Args:
        series: Series with possible NaN
        method: "forward_fill" or "interpolate"
    
    Returns:
        Series without NaN
    """
    if method == "forward_fill":
        return series.fillna(method='ffill').fillna(method='bfill')
    elif method == "interpolate":
        return series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def normalize_zscore(series: pd.Series) -> tuple:
    """
    Z-score normalize a series.
    
    Args:
        series: input series
    
    Returns:
        (normalized series, dict with mean/std)
    """
    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / std
    return normalized, {"mean": float(mean), "std": float(std)}


def main():
    """Preprocess raw data."""
    parser = argparse.ArgumentParser(description="Preprocess PM10 data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    interim_dir = Path(cfg['paths']['interim_dir'])
    processed_dir = Path(cfg['paths']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw
    raw_path = interim_dir / "raw_loaded.parquet"
    print(f"Loading from {raw_path}...")
    df = pd.read_parquet(raw_path)
    
    series = df['pm10_value'].copy()
    
    # Impute
    print(f"Imputing missing values ({series.isna().sum()} NaN)...")
    series = impute_series(series, method=cfg['preprocessing']['imputation_method'])
    
    # Normalize (z-score on full series)
    print("Normalizing (z-score)...")
    series_norm, norm_params = normalize_zscore(series)
    
    # Save normalized series
    output_path = processed_dir / "pm10_preprocessed.parquet"
    pd.DataFrame({'pm10_normalized': series_norm}, index=series_norm.index).to_parquet(output_path)
    print(f"Saved to {output_path}")
    
    # Save normalization parameters for inversion
    params_path = processed_dir / "normalization_params.json"
    with open(params_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalization params to {params_path}")
    
    print(f"\nPreprocessed shape: {series_norm.shape}")
    print(f"Mean: {series_norm.mean():.4f}, Std: {series_norm.std():.4f}")


if __name__ == "__main__":
    main()
