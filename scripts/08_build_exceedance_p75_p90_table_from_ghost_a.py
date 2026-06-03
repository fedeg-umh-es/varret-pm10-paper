#!/usr/bin/env python3
"""Build manuscript-ready exceedance table for P75/P90 under rolling-origin."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "outputs" / "metrics" / "metrics_events_rolling_origin.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "tables"
OUTPUT_CSV = OUTPUT_DIR / "table_c1_exceedance_p75_p90.csv"
OUTPUT_TEX = OUTPUT_DIR / "table_c1_exceedance_p75_p90.tex"

MODEL_MAP = {"lgbm": "LightGBM", "sarima": "SARIMA"}
THRESHOLD_MAP = {75: "P75", 90: "P90"}
HORIZONS = [1, 6, 24, 48]


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[
        df["model"].isin(MODEL_MAP)
        & df["h"].isin(HORIZONS)
        & df["threshold"].isin(THRESHOLD_MAP)
    ].copy()

    table["Threshold"] = table["threshold"].map(THRESHOLD_MAP)
    table["Model"] = table["model"].map(MODEL_MAP)
    table["Horizon"] = table["h"].astype(int)
    table["Recall"] = table["recall_events"].astype(float).round(3)
    table["Precision"] = table["precision_events"].astype(float).round(3)
    table["Flag rate"] = table["flag_rate"].astype(float).round(3)
    table["Base rate"] = table["base_rate_test"].astype(float).round(3)

    threshold_order = {"P75": 0, "P90": 1}
    model_order = {"LightGBM": 0, "SARIMA": 1}
    table = (
        table.assign(
            _threshold_order=table["Threshold"].map(threshold_order),
            _model_order=table["Model"].map(model_order),
        )
        .sort_values(["_threshold_order", "_model_order", "Horizon"])
        .loc[:, ["Threshold", "Model", "Horizon", "Recall", "Precision", "Flag rate", "Base rate"]]
        .reset_index(drop=True)
    )
    return table


def build_latex(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    col_spec = "llrcccc"
    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]

    for row in table.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    table = build_table(df)

    table.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_TEX.write_text(build_latex(table), encoding="utf-8")

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
