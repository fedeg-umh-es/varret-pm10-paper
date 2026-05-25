"""Build a variance-retention diagnostic table from prediction and skill CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics.variance import build_variance_retention_summary


def main() -> None:
    """Read predictions and skill tables, then write the final diagnostic summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/metrics/predictions.csv"),
        help="Prediction table with y_true/y_pred rows.",
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=Path("outputs/metrics/skill_summary.csv"),
        help="Skill summary table with one row per dataset/model/horizon.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/variance_retention_summary.csv"),
        help="Output variance-retention CSV.",
    )
    parser.add_argument(
        "--station-name",
        default=None,
        help="Accepted for pipeline compatibility; station metadata is added by build_unified_variance_table.py.",
    )
    args = parser.parse_args()

    predictions_path = args.predictions
    skill_path = args.skill
    output_path = args.output

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions table: {predictions_path}")
    if not skill_path.exists():
        raise FileNotFoundError(f"Missing skill table: {skill_path}")

    predictions_df = pd.read_csv(predictions_path)
    skill_df = pd.read_csv(skill_path)
    summary_df = build_variance_retention_summary(predictions_df, skill_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Wrote variance-retention summary to {output_path}")


if __name__ == "__main__":
    main()
