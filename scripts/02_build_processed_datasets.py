"""Build canonical (date, y) processed dataset from raw CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

_EEA_HINT = (
    "\nDownload PM10 daily data from EEA Discomap:\n"
    "  https://eeadmz1-downloads-webapp.azurewebsites.net/\n"
    "  Station: ES0118A (Casa de Campo, Madrid) | Year: 2023 | Pollutant: PM10\n"
    "Place the CSV at data/raw/pm10_daily.csv with columns: date, y\n"
    "  date: ISO-8601 date (YYYY-MM-DD)\n"
    "  y:    daily mean PM10 concentration (µg/m³)\n"
    "See docs/data_download.md for full instructions.\n"
)


def main() -> None:
    cfg = yaml.safe_load(Path("configs/datasets/pm10.yaml").read_text())["dataset"]
    raw_path = Path(cfg["raw_path"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}{_EEA_HINT}")

    date_col = cfg["date_column"]
    target_col = cfg["target_column"]

    df = pd.read_csv(raw_path)
    missing = [c for c in (date_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing} in {raw_path}. Found: {list(df.columns)}"
        )

    out = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col]),
                "y": pd.to_numeric(df[target_col], errors="coerce"),
            }
        )
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    out_path = Path(cfg["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        f"Wrote {len(out)} records "
        f"({out['date'].min().date()} → {out['date'].max().date()}) → {out_path}"
    )


if __name__ == "__main__":
    main()
