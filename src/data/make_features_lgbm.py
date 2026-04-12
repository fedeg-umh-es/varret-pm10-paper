"""
Feature engineering for LightGBM
=================================
Create lag features, rolling statistics, temporal features.
All computed from normalized series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml


def create_lag_features(series: pd.Series, lags: list) -> pd.DataFrame:
    """Create lag features."""
    df = pd.DataFrame({'pm10': series})
    for lag in lags:
        df[f'lag_{lag}'] = series.shift(lag)
    return df


def create_rolling_features(series: pd.Series, windows: list) -> pd.DataFrame:
    """Create rolling window statistics."""
    df = pd.DataFrame()
    for window in windows:
        df[f'rolling_mean_{window}'] = series.rolling(window).mean()
        df[f'rolling_std_{window}'] = series.rolling(window).std()
        df[f'rolling_min_{window}'] = series.rolling(window).min()
        df[f'rolling_max_{window}'] = series.rolling(window).max()
    return df


def create_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create temporal features."""
    df = pd.DataFrame(index=index)
    df['hour'] = index.hour
    df['dayofweek'] = index.dayofweek
    df['month'] = index.month
    # One-hot encode month for seasonality (optional, can add if needed)
    return df


def main():
    """Build LightGBM features."""
    parser = argparse.ArgumentParser(description="Create LightGBM features")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    
    # Load preprocessed
    print("Loading preprocessed data...")
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    series = df_pre['pm10_normalized'].squeeze()
    
    # Create features
    print("Creating lag features...")
    lags = cfg['features_lgbm']['lags']
    df_lags = create_lag_features(series, lags)
    
    print("Creating rolling features...")
    windows = cfg['features_lgbm']['rolling_windows']
    df_rolling = create_rolling_features(series, windows)
    
    print("Creating temporal features...")
    df_temporal = create_temporal_features(series.index)
    
    # Combine
    df_features = pd.concat([df_lags, df_rolling, df_temporal], axis=1)
    
    # Remove rows with NaN (due to lags/rolling)
    print(f"Rows before dropping NaN: {len(df_features)}")
    df_features = df_features.dropna()
    print(f"Rows after dropping NaN: {len(df_features)}")
    
    # Save
    output_path = processed_dir / "features_lgbm.parquet"
    df_features.to_parquet(output_path)
    print(f"Saved to {output_path}")
    print(f"Features shape: {df_features.shape}")
    print(f"Feature columns: {list(df_features.columns)}")


if __name__ == "__main__":
    main()
