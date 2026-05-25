#!/usr/bin/env python3
"""Concatenate final per-station prediction tables into one analysis table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXCLUDE_PREFIXES = ("predictions_base_", "predictions_sarima_")
EXCLUDE_NAMES = {"predictions.csv", "predictions_all_stations.csv", "predictions_test.csv"}


def _is_final_station_predictions(path: Path) -> bool:
    return (
        path.name.startswith("predictions_")
        and path.name not in EXCLUDE_NAMES
        and not any(path.name.startswith(prefix) for prefix in EXCLUDE_PREFIXES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/predictions_all_stations.csv"))
    args = parser.parse_args()

    paths = sorted(p for p in args.input_dir.glob("predictions_*.csv") if _is_final_station_predictions(p))
    if not paths:
        raise FileNotFoundError(f"No final station prediction tables found in {args.input_dir}")

    frames = [pd.read_csv(path) for path in paths]
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["dataset", "model", "horizon", "origin_date", "date"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} from {len(paths)} station tables with {len(out)} rows")


if __name__ == "__main__":
    main()
