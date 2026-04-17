"""Helpers for assembling forecast outputs under the P33 contracts."""

import pandas as pd


def build_predictions_table(records: list[dict]) -> pd.DataFrame:
    """Build the canonical predictions table from row records."""
    return pd.DataFrame.from_records(records)
