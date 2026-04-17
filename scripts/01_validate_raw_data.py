"""Validate raw daily PM10 data files before processing."""

from pathlib import Path


def main() -> None:
    """Check that the expected raw data path exists."""
    raw_path = Path("data/raw/pm10_daily.csv")
    if raw_path.exists():
        print(f"Raw dataset found: {raw_path}")
    else:
        print(f"Raw dataset missing: {raw_path}")


if __name__ == "__main__":
    main()
