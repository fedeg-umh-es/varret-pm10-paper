"""Build PRISMA reporting-audit table and figure artifacts.

The audit values are fixed manuscript inputs:
- PRISMA eligible studies: 503
- Abstract-coded studies: 486
- Coding basis: abstract-level evidence
- Interpretation rule: low reporting percentages indicate abstract-level
  reporting opacity, not confirmed absence of the corresponding practice
  in the full text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
CSV_PATH = TABLES_DIR / "prisma_reporting_audit_summary.csv"
TEX_PATH = TABLES_DIR / "prisma_reporting_audit_summary.tex"
FIGURE_STEM = "figure1_reporting_gap_audit"
DENOMINATOR = 486

CAPTION = (
    "Abstract-level reporting visibility of evaluation-relevant practices in "
    "PM forecasting studies. Percentages are computed over 486 abstract-coded "
    "studies; low reporting rates indicate reporting opacity, not confirmed "
    "absence of the corresponding practice in full text."
)
LABEL = "tab:prisma-reporting-audit"
LATEX_ROW_END = r"\\"


def _audit_rows() -> list[dict[str, object]]:
    return [
        {
            "dimension": "Persistence or naive baseline",
            "count": 18,
            "denominator": DENOMINATOR,
            "percent": 3.7,
            "interpretation": "Minimum operational comparator",
        },
        {
            "dimension": "Horizon-wise reporting",
            "count": 11,
            "denominator": DENOMINATOR,
            "percent": 2.3,
            "interpretation": "Multi-step interpretability",
        },
        {
            "dimension": "Temporally appropriate validation",
            "count": 24,
            "denominator": DENOMINATOR,
            "percent": 4.9,
            "interpretation": "Deployment-like temporal credibility",
        },
        {
            "dimension": "Explicit leakage-safe safeguards",
            "count": 3,
            "denominator": DENOMINATOR,
            "percent": 0.6,
            "interpretation": "Protection against information bleed",
        },
        {
            "dimension": "Operational-usefulness framing",
            "count": 51,
            "denominator": DENOMINATOR,
            "percent": 10.5,
            "interpretation": "Decision relevance",
        },
    ]


def _validate_audit_df(df: pd.DataFrame) -> None:
    if not (df["denominator"] == DENOMINATOR).all():
        raise ValueError("All PRISMA reporting-audit denominators must be 486.")

    computed = 100.0 * df["count"] / df["denominator"]
    delta = (computed - df["percent"]).abs()
    if (delta > 0.1).any():
        bad = df.loc[delta > 0.1, ["dimension", "count", "denominator", "percent"]]
        raise ValueError(
            "Hardcoded percentages do not match count/denominator within 0.1 points: "
            f"{bad.to_dict(orient='records')}"
        )


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _write_csv(df: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH}")


def _write_latex_table(df: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{CAPTION}}}",
        f"\\label{{{LABEL}}}",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        "Evaluation dimension & Explicitly reported & \\% & Interpretive function "
        + LATEX_ROW_END,
        r"\midrule",
    ]

    for _, row in df.iterrows():
        dimension = _latex_escape(str(row["dimension"]))
        reported = f"{int(row['count'])}/{int(row['denominator'])}"
        percent = f"{float(row['percent']):.1f}"
        interpretation = _latex_escape(str(row["interpretation"]))
        lines.append(
            f"{dimension} & {reported} & {percent} & {interpretation} {LATEX_ROW_END}"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


def _write_reporting_gap_figure(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Order from highest to lowest visibility for readability.
    plot_df = df.sort_values("percent", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["percent"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["dimension"])
    ax.set_xlabel("Explicitly reported in abstracts (%)")
    ax.set_xlim(0, max(12.0, float(plot_df["percent"].max()) + 2.0))
    ax.invert_yaxis()

    for idx, row in plot_df.iterrows():
        ax.text(
            float(row["percent"]) + 0.25,
            idx,
            f"{int(row['count'])}/{int(row['denominator'])} ({row['percent']:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    pdf_path = FIGURES_DIR / f"{FIGURE_STEM}.pdf"
    png_path = FIGURES_DIR / f"{FIGURE_STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    df = pd.DataFrame(_audit_rows())
    _validate_audit_df(df)
    _write_csv(df)
    _write_latex_table(df)
    _write_reporting_gap_figure(df)


if __name__ == "__main__":
    main()
