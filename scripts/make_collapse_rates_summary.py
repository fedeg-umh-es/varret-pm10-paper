#!/usr/bin/env python3
"""Build the canonical collapse-rate summary table.

The all-model collapse rate is lower than the ML-only rate because seasonal
naive is the variance-preserving reference model and contributes 0 collapsed
cells under the alpha < 0.5 definition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "tables" / "variance_retention_all_stations.csv"
CSV_PATH = ROOT / "outputs" / "tables" / "collapse_rates_summary.csv"
TEX_PATH = ROOT / "outputs" / "tables" / "collapse_rates_summary.tex"

ML_MODELS = ("hgb_direct", "ridge_direct")
SEASONAL_MODEL = "seasonal_naive"

GROUPS = (
    ("HGB direct", ("hgb_direct",)),
    ("Ridge direct", ("ridge_direct",)),
    ("ML combined", ML_MODELS),
    ("Seasonal naive", (SEASONAL_MODEL,)),
    ("All models", (*ML_MODELS, SEASONAL_MODEL)),
)

EXPECTED = {
    "HGB direct": (118, 119, 99.2),
    "Ridge direct": (118, 119, 99.2),
    "ML combined": (236, 238, 99.2),
    "Seasonal naive": (0, 119, 0.0),
    "All models": (236, 357, 66.1),
}


def collapse_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing required columns: {sorted(missing)}")

    rows = []
    for group, models in GROUPS:
        subset = df[df["model"].isin(models)]
        collapsed = int((subset["alpha"] < 0.5).sum())
        total = int(len(subset))
        rate = round(100.0 * collapsed / total, 1) if total else float("nan")
        rows.append(
            {
                "Group": group,
                "Collapsed cells": collapsed,
                "Total cells": total,
                "Collapse rate (%)": rate,
            }
        )
    return pd.DataFrame(rows)


def verify_expected(summary: pd.DataFrame) -> None:
    for row in summary.to_dict(orient="records"):
        group = row["Group"]
        expected = EXPECTED[group]
        actual = (
            int(row["Collapsed cells"]),
            int(row["Total cells"]),
            float(row["Collapse rate (%)"]),
        )
        if actual != expected:
            raise ValueError(f"{group} mismatch: expected {expected}, got {actual}")


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_latex(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Collapse-rate summary for variance-retention diagnostic cells.}",
        r"\label{tab:collapse_rates_summary}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Group & Collapsed cells & Total cells & Collapse rate (\%) \\",
        r"\midrule",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"{latex_escape(str(row['Group']))} & "
            f"{int(row['Collapsed cells'])} & "
            f"{int(row['Total cells'])} & "
            f"{float(row['Collapse rate (%)']):.1f} \\\\"
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


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    summary = collapse_summary(df)
    verify_expected(summary)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CSV_PATH, index=False)
    write_latex(summary)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
