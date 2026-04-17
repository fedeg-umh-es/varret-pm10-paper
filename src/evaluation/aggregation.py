"""Aggregation helpers for skill and diagnostic outputs."""

import pandas as pd


def aggregate_skill_rows(rows: list[dict]) -> pd.DataFrame:
    """Convert aggregated metric rows into a dataframe."""
    return pd.DataFrame.from_records(rows)
