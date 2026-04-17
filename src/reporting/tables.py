"""Tabular reporting utilities for P33."""

import pandas as pd


def ensure_sorted_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Return a sorted copy of a reporting table."""
    return df.sort_values(by).reset_index(drop=True)
