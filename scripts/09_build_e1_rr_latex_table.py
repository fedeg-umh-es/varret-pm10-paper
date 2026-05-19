"""Build a paper-ready LaTeX table for E1-RR variance-retention diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUMMARY_PATH = Path("outputs/tables/variance_retention_summary.csv")
OUTPUT_PATH = Path("outputs/tables/e1_rr_variance_retention_summary.tex")

CAPTION = (
    "Summary of persistence-relative skill and variance-retention diagnostics "
    "for the E1-RR daily lags-only post-evaluation."
)
LABEL = "tab:e1_rr_variance_retention_summary"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _build_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model, group in summary_df.groupby("model", sort=True):
        total = len(group)
        rows.append(
            {
                "Model": str(model),
                "Mean skill": _fmt(float(group["skill"].mean())),
                "Mean alpha": _fmt(float(group["alpha"].mean())),
                "Mean Skill_VP": _fmt(float(group["skill_vp"].mean())),
                "Collapse horizons": f"{int(group['collapse_flag'].sum())}/{total}",
                "Inflation horizons": f"{int(group['inflation_flag'].sum())}/{total}",
                "Near-ideal horizons": f"{int(group['near_ideal_flag'].sum())}/{total}",
                "Low-sample horizons": f"{int(group['low_sample_flag'].sum())}/{total}",
            }
        )
    return pd.DataFrame(rows)


def _to_latex(table_df: pd.DataFrame) -> str:
    latex = table_df.to_latex(index=False, escape=True, column_format="lrrrrrrr")
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{CAPTION}}}\n"
        f"\\label{{{LABEL}}}\n"
        f"{latex}"
        "\\end{table}\n"
    )


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing variance retention table: {SUMMARY_PATH}")

    summary_df = pd.read_csv(SUMMARY_PATH)
    required = {
        "model",
        "skill",
        "alpha",
        "skill_vp",
        "collapse_flag",
        "inflation_flag",
        "near_ideal_flag",
        "low_sample_flag",
    }
    missing = sorted(required - set(summary_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {SUMMARY_PATH}: {missing}")

    table_df = _build_summary_table(summary_df)
    latex = _to_latex(table_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
