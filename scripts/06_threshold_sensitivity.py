#!/usr/bin/env python3
"""Collapse-rate sensitivity over alpha thresholds 0.2 to 0.8."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/variance_retention_all_stations.csv")
DEFAULT_OUTPUT = Path("outputs/tables/threshold_sensitivity.csv")
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def build_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Variance-retention table missing columns: {sorted(missing)}")
    rows = []
    for threshold in THRESHOLDS:
        for model, group in df.groupby("model", sort=True):
            n_cells = int(len(group))
            n_collapsed = int((group["alpha"] < threshold).sum())
            rows.append(
                {
                    "threshold": threshold,
                    "model": model,
                    "n_cells": n_cells,
                    "n_collapsed": n_collapsed,
                    "collapse_rate": n_collapsed / n_cells if n_cells else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build alpha threshold sensitivity table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    table = build_threshold_sensitivity(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()
