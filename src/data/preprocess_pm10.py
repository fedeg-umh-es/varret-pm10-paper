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
        method: "forward_fill". Interpolation is rejected because ordinary
            linear interpolation uses a future endpoint.
    
    Returns:
        Series without NaN
    """
    if method == "forward_fill":
        return series.ffill()
    elif method == "interpolate":
        raise ValueError("Linear interpolation is not causal; use forward_fill")
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def normalize_zscore(series: pd.Series, fit_series: pd.Series | None = None) -> tuple:
    """
    Z-score normalize a series.
    
    Args:
        series: input series
    
    Returns:
        (normalized series, dict with mean/std)
    """
    fit_series = series if fit_series is None else fit_series
    mean = fit_series.mean()
    std = fit_series.std()
    normalized = (series - mean) / std
    return normalized, {"mean": float(mean), "std": float(std)}


def main():
    """Preprocess raw data."""
    parser = argparse.ArgumentParser(description="Preprocess PM10 data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--train-end",
        required=True,
        help="Last timestamp included when fitting normalization parameters",
    )
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
    
    train = series.loc[:pd.Timestamp(args.train_end)].dropna()
    if train.empty:
        raise ValueError("--train-end selects an empty training window")
    print("Normalizing (z-score fitted on training window only)...")
    series_norm, norm_params = normalize_zscore(series, fit_series=train)
    
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
