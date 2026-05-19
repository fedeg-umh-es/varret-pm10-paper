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
LATEX_ROW_END = r"\\"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _latex_escape(value: str) -> str:
    """Escape a small subset of LaTeX-sensitive characters used in table text."""
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _build_summary_table(summary_df: pd.DataFrame) -> list[dict[str, str]]:
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
    return rows


def _to_latex(rows: list[dict[str, str]]) -> str:
    headers = [
        "Model",
        "Mean skill",
        "Mean alpha",
        r"Mean Skill\_VP",
        "Collapse horizons",
        "Inflation horizons",
        "Near-ideal horizons",
        "Low-sample horizons",
    ]
    keys = [
        "Model",
        "Mean skill",
        "Mean alpha",
        "Mean Skill_VP",
        "Collapse horizons",
        "Inflation horizons",
        "Near-ideal horizons",
        "Low-sample horizons",
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{CAPTION}}}",
        f"\\label{{{LABEL}}}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        " & ".join(headers) + " " + LATEX_ROW_END,
        r"\midrule",
    ]

    for row in rows:
        values = [_latex_escape(row[key]) if key == "Model" else row[key] for key in keys]
        lines.append(" & ".join(values) + " " + LATEX_ROW_END)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


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

    rows = _build_summary_table(summary_df)
    latex = _to_latex(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
