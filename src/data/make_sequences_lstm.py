"""
Create sequences for LSTM
=========================
Transform time series into (context_len, horizon) samples.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import yaml


def create_sequences(
    series: np.ndarray,
    context_len: int,
    stride: int = 1
) -> tuple:
    """
    Create sequences for LSTM.
    
    Args:
        series: 1D array
        context_len: context length
        stride: step between samples
    
    Returns:
        (X, indices)
        X: (n_samples, context_len, 1)
        indices: start index of each sample
    """
    X_list = []
    indices = []
    
    for start_idx in range(0, len(series) - context_len + 1, stride):
        X_list.append(series[start_idx:start_idx + context_len])
        indices.append(start_idx + context_len - 1)  # End index of context
    
    X = np.array(X_list).reshape(-1, context_len, 1).astype(np.float32)
    
    return X, np.array(indices)


def main():
    """Create LSTM sequences."""
    parser = argparse.ArgumentParser(description="Create LSTM sequences")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    processed_dir = Path(cfg['paths']['processed_dir'])
    
    # Load preprocessed
    print("Loading preprocessed data...")
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    series = df_pre['pm10_normalized'].values.squeeze()
    
    context_len = cfg['sequences_lstm']['context_length']
    stride = cfg['sequences_lstm']['stride']
    
    print(f"Creating sequences (context_len={context_len}, stride={stride})...")
    X, indices = create_sequences(series, context_len, stride)
    
    # Save
    output_path = processed_dir / "sequences_lstm.npz"
    np.savez(output_path, X=X, indices=indices)
    print(f"Saved to {output_path}")
    print(f"X shape: {X.shape}")
    print(f"Indices shape: {indices.shape}")


if __name__ == "__main__":
    main()
