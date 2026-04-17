"""Build the mandatory P33 variance-retention diagnostic table."""

from pathlib import Path

import pandas as pd

from src.diagnostics.variance import build_variance_retention_summary


def main() -> None:
    """Read predictions and skill tables, then write the final diagnostic summary."""
    predictions_path = Path("outputs/metrics/predictions.csv")
    skill_path = Path("outputs/metrics/skill_summary.csv")
    output_path = Path("outputs/tables/variance_retention_summary.csv")

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
