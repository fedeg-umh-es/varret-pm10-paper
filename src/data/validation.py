"""Validation helpers for P33 datasets and tabular outputs."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, expected: Iterable[str], name: str) -> None:
    """Raise an error if a dataframe does not contain the expected columns."""
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
