#!/usr/bin/env python3
"""Build threshold-sensitivity tables for alpha-based collapse diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "tables" / "variance_retention_all_stations.csv"
CSV_PATH = ROOT / "outputs" / "tables" / "alpha_threshold_sensitivity.csv"
TEX_PATH = ROOT / "outputs" / "tables" / "alpha_threshold_sensitivity.tex"
SUMMARY_PATH = ROOT / "outputs" / "audit" / "alpha_threshold_sensitivity_summary.md"

THRESHOLDS = (0.4, 0.5, 0.6)
ML_MODELS = ("hgb_direct", "ridge_direct")
SEASONAL_MODEL = "seasonal_naive"
GROUPS = (
    ("hgb_direct", ("hgb_direct",)),
    ("ridge_direct", ("ridge_direct",)),
    ("ML combined", ML_MODELS),
    ("seasonal_naive", (SEASONAL_MODEL,)),
    ("all models", (*ML_MODELS, SEASONAL_MODEL)),
)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing required columns: {sorted(missing)}")

    rows = []
    for threshold in THRESHOLDS:
        for group, models in GROUPS:
            subset = df[df["model"].isin(models)]
            collapsed = int((subset["alpha"] < threshold).sum())
            total = int(len(subset))
            rate = round(100.0 * collapsed / total, 1) if total else float("nan")
            rows.append(
                {
                    "threshold": threshold,
                    "group": group,
                    "collapsed_cells": collapsed,
                    "total_cells": total,
                    "collapse_rate_pct": rate,
                }
            )
    return pd.DataFrame(rows)


def write_latex(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Sensitivity of variance-retention collapse rates to the diagnostic threshold.}",
        r"\label{tab:alpha_threshold_sensitivity}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Threshold & Group & Collapsed cells & Total cells & Collapse rate (\%) \\",
        r"\midrule",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"{row['threshold']:.1f} & "
            f"{row['group'].replace('_', r'\_')} & "
            f"{int(row['collapsed_cells'])} & "
            f"{int(row['total_cells'])} & "
            f"{float(row['collapse_rate_pct']):.1f} \\\\"
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


def write_text_summary(summary: pd.DataFrame) -> None:
    ml = summary[summary["group"].eq("ML combined")].sort_values("threshold")
    seasonal = summary[summary["group"].eq("seasonal_naive")].sort_values("threshold")
    min_ml_rate = float(ml["collapse_rate_pct"].min())
    conclusion = (
        "Diagnostic threshold sensitivity supports the same qualitative conclusion: "
        "ML variance-retention collapse is near-universal across the tested thresholds."
        if min_ml_rate >= 90.0
        else "Diagnostic threshold sensitivity weakens the near-universal ML collapse conclusion at one or more tested thresholds."
    )
    seasonal_max = int(seasonal["collapsed_cells"].max())
    seasonal_text = (
        "Seasonal naive remains non-collapsed at all tested thresholds."
        if seasonal_max == 0
        else "Seasonal naive is not fully non-collapsed at all tested thresholds; inspect threshold-specific counts."
    )

    ml_counts = "; ".join(
        f"alpha < {row.threshold:.1f}: {int(row.collapsed_cells)}/{int(row.total_cells)} ({row.collapse_rate_pct:.1f}%)"
        for row in ml.itertuples(index=False)
    )
    seasonal_counts = "; ".join(
        f"alpha < {row.threshold:.1f}: {int(row.collapsed_cells)}/{int(row.total_cells)} ({row.collapse_rate_pct:.1f}%)"
        for row in seasonal.itertuples(index=False)
    )

    lines = [
        "# Alpha Threshold Sensitivity Summary",
        "",
        f"- {conclusion}",
        f"- ML combined collapsed cells by threshold: {ml_counts}.",
        f"- {seasonal_text}",
        f"- Seasonal naive collapsed cells by threshold: {seasonal_counts}.",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    summary = build_summary(df)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_PATH, index=False)
    write_latex(summary)
    write_text_summary(summary)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {TEX_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
