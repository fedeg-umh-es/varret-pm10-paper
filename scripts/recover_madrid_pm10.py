#!/usr/bin/env python3
"""Recover the exact public input used by the leakage-free Paper A rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.madrid_hourly import (
    MADRID_2023_SHA256,
    MADRID_2023_URL,
    download_2023_archive,
    parse_casa_de_campo_pm10,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/madrid_air_hourly_2023.zip"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/casa_de_campo_pm10_2023.csv")
    )
    args = parser.parse_args()

    download_2023_archive(args.raw)
    frame = parse_casa_de_campo_pm10(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    manifest = {
        "source_url": MADRID_2023_URL,
        "archive_sha256": MADRID_2023_SHA256,
        "station": "Casa de Campo",
        "station_code": 24,
        "magnitude": "PM10",
        "magnitude_code": 10,
        "hourly_grid_rows": int(len(frame)),
        "validated_observations": int(frame["is_valid"].sum()),
        "missing_or_invalid": int((~frame["is_valid"]).sum()),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
