"""Input and output helpers for P33 datasets and tabular artifacts."""

from pathlib import Path

import pandas as pd


def read_canonical_dataset(path: str | Path) -> pd.DataFrame:
    """Read a canonical processed dataset with columns date and y."""
    dataset = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return dataset


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a tabular artifact to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
