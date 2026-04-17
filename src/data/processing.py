"""Processing utilities for canonical daily PM10 datasets."""

import pandas as pd

from src.data.schema import CANONICAL_DATASET_COLUMNS


def build_canonical_daily_dataset(df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
    """Return a canonical dataset with columns date and y, sorted by date."""
    processed = df[[date_column, target_column]].copy()
    processed.columns = CANONICAL_DATASET_COLUMNS
    processed["date"] = pd.to_datetime(processed["date"])
    processed = processed.sort_values("date").reset_index(drop=True)
    return processed
