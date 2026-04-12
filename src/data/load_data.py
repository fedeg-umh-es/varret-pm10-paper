"""
Load and validate raw PM10 data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml


def load_raw_pm10(csv_path: str) -> pd.DataFrame:
    """
    Load PM10 CSV, validate, convert timestamp.
    
    Expected columns: timestamp (or similar), pm10_value (or similar)
    Auto-detects column names with some flexibility.
    
    Args:
        csv_path: path to CSV file
    
    Returns:
        DataFrame with columns: ['timestamp', 'pm10_value']
    """
    df = pd.read_csv(csv_path)
    
    # Auto-detect timestamp column
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if not time_cols:
        time_col = df.columns[0]  # First column
    else:
        time_col = time_cols[0]
    
    # Auto-detect PM10 value column
    pm10_cols = [c for c in df.columns if 'pm10' in c.lower() or 'pm_10' in c.lower()]
    if not pm10_cols:
        pm10_cols = [c for c in df.columns if 'value' in c.lower()]
    if not pm10_cols:
        pm10_col = df.columns[1]  # Second column
    else:
        pm10_col = pm10_cols[0]
    
    # Rename for consistency
    df = df[[time_col, pm10_col]].copy()
    df.columns = ['timestamp', 'pm10_value']
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pm10_value'] = pd.to_numeric(df['pm10_value'], errors='coerce')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df


def main():
    """Load raw data from config and save to interim."""
    parser = argparse.ArgumentParser(description="Load raw PM10 data")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to config.yaml")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    raw_path = cfg['paths']['raw']
    interim_dir = Path(cfg['paths']['interim_dir'])
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PM10 from {raw_path}...")
    df = load_raw_pm10(raw_path)
    
    # Save
    output_path = interim_dir / "raw_loaded.parquet"
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df['pm10_value'].isna().sum()}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    main()
