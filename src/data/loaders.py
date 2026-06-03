"""Dataset loading utilities for reproducible forecasting experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def load_csv_series(
    path: str | Path,
    timestamp_col: str,
    value_cols: Iterable[str],
    sort: bool = True,
) -> pd.DataFrame:
    """Load a time series CSV and enforce minimal schema checks."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    required = {timestamp_col, *value_cols}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    out = df.loc[:, [timestamp_col, *value_cols]].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="raise")
    if sort:
        out = out.sort_values(timestamp_col).reset_index(drop=True)
    return out


def _resolve_path(path_str: str, config_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    repo_root = config_path.resolve().parents[2]
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return (repo_root / path).resolve()


def load_dataset(config_path: str | Path) -> dict[str, Any]:
    """
    Returns:
        {
            "data": pd.DataFrame,
            "X": pd.DataFrame,
            "y": pd.Series,
            "time_index": pd.Series,
            "target_column": str,
            "datetime_column": str,
            "horizons": list[int],
            "frequency": str,
            "name": str,
            "metadata": dict[str, Any],
        }
    """
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_keys = {
        "name",
        "target_column",
        "datetime_column",
        "frequency",
        "horizons",
        "features",
        "paths",
    }
    missing_keys = required_keys.difference(config)
    if missing_keys:
        raise ValueError(
            f"Missing required dataset config keys in {config_file}: {sorted(missing_keys)}"
        )

    target_column = config["target_column"]
    datetime_column = config["datetime_column"]
    endogenous = list(config["features"].get("endogenous", []))
    exogenous = list(config["features"].get("exogenous", []))
    value_cols = list(dict.fromkeys([target_column, *endogenous, *exogenous]))

    data_path = _resolve_path(config["paths"]["raw"], config_file)
    df = load_csv_series(
        path=data_path,
        timestamp_col=datetime_column,
        value_cols=value_cols,
        sort=True,
    )

    if df[value_cols].isna().any().any():
        valid_mask = df[value_cols].notna().all(axis=1)
        segment_ids = (~valid_mask).cumsum()[valid_mask]
        if segment_ids.empty:
            raise ValueError(f"No valid data rows found in {data_path} for target/features.")
        longest_segment_id = segment_ids.value_counts().idxmax()
        df = df.loc[segment_ids[segment_ids == longest_segment_id].index].copy()
        df = df.reset_index(drop=True)

    feature_columns = list(dict.fromkeys([*endogenous, *exogenous]))
    X = df.loc[:, feature_columns].copy()
    y = df.loc[:, target_column].copy()
    time_index = df.loc[:, datetime_column].copy()

    metadata = {
        "config_path": str(config_file.resolve()),
        "raw_data_path": str(data_path),
        "split": config.get("split", {}),
        "preprocessing": config.get("preprocessing", {}),
        "features": config["features"],
    }

    return {
        "data": df,
        "X": X,
        "y": y,
        "time_index": time_index,
        "target_column": target_column,
        "datetime_column": datetime_column,
        "horizons": list(config["horizons"]),
        "frequency": config["frequency"],
        "name": config["name"],
        "metadata": metadata,
    }
