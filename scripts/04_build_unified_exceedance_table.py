#!/usr/bin/env python3
"""Concatenate station-level exceedance tables and attach station metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_GLOB = "outputs/metrics/exceedance_*.csv"
DEFAULT_METADATA = Path("outputs/tables/variance_retention_all_stations.csv")
DEFAULT_OUTPUT = Path("outputs/tables/exceedance_all_stations.csv")


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cols = ["dataset", "station_id", "station_name", "station_type", "province"]
    df = pd.read_csv(path)
    present = [col for col in cols if col in df.columns]
    if "dataset" not in present:
        return pd.DataFrame()
    return df[present].drop_duplicates("dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified exceedance table.")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = sorted(path for path in Path(".").glob(args.input_glob) if path.name != "exceedance_test.csv")
    if not paths:
        raise FileNotFoundError(f"No exceedance files matched {args.input_glob}")
    frames = [pd.read_csv(path) for path in paths]
    table = pd.concat(frames, ignore_index=True)
    metadata = _load_metadata(args.metadata)
    if not metadata.empty:
        table = table.merge(metadata, on="dataset", how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows from {len(paths)} files")


if __name__ == "__main__":
    main()
