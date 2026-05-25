#!/usr/bin/env python3
"""Concatenate per-station Diebold-Mariano test outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/dm_significance_all_stations.csv"))
    args = parser.parse_args()

    paths = sorted(path for path in args.input_dir.glob("dm_*.csv") if path.name != "dm_test.csv")
    if not paths:
        raise FileNotFoundError(f"No dm_*.csv files found in {args.input_dir}")

    frames = [pd.read_csv(path) for path in paths]
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["dataset", "station", "model", "horizon"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} from {len(paths)} station DM tables with {len(out)} rows")


if __name__ == "__main__":
    main()
