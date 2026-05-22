"""Build a short textual summary of the latest P33 run."""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Write a concise run summary that references the main P33 table."""
    report_path = Path("outputs/reports/run_summary.txt")
    raw_path = Path("data/raw/pm10_daily.csv")
    predictions_path = Path("outputs/metrics/predictions.csv")
    table_path = Path("outputs/tables/variance_retention_summary.csv")

    raw = pd.read_csv(raw_path)
    predictions = pd.read_csv(predictions_path)
    summary = pd.read_csv(table_path)
    model_summary = (
        summary.groupby("model", sort=True)
        .agg(
            mean_skill=("skill", "mean"),
            mean_alpha=("alpha", "mean"),
            mean_skill_vp=("skill_vp", "mean"),
            collapse=("collapse_flag", "sum"),
            total=("horizon", "count"),
            min_n=("n", "min"),
        )
        .reset_index()
    )

    model_lines = []
    for row in model_summary.itertuples(index=False):
        model_lines.append(
            f"- {row.model}: mean_skill={row.mean_skill:.3f}, "
            f"mean_alpha={row.mean_alpha:.3f}, mean_skill_vp={row.mean_skill_vp:.3f}, "
            f"collapse={int(row.collapse)}/{int(row.total)}, min_n={int(row.min_n)}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "P33 run summary\n"
        "==============\n"
        f"Raw dataset: {raw_path}\n"
        f"Raw rows: {len(raw)}\n"
        f"Raw date range: {raw['date'].min()} to {raw['date'].max()}\n"
        f"Predictions: {predictions_path} ({len(predictions)} rows)\n"
        f"Prediction date range: {predictions['date'].min()} to {predictions['date'].max()}\n"
        f"Primary diagnostic table: {table_path}\n\n"
        "Model summary\n"
        "-------------\n"
        + "\n".join(model_lines)
        + "\n\nReview skill, alpha, and skill_vp jointly to assess variance retention.\n",
        encoding="utf-8",
    )
    print(f"Wrote run summary to {report_path}")


if __name__ == "__main__":
    main()
