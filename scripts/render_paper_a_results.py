#!/usr/bin/env python3
"""Render the canonical LaTeX result rows from empirical output tables."""

from pathlib import Path

import pandas as pd


def display(value: float) -> str:
    formatted = f"{value:.3f}"
    return f"${formatted}$" if value < 0 else formatted


def main() -> None:
    metrics = pd.read_csv("outputs/reproduction/metrics_rolling_origin.csv")
    rows = []
    for horizon in (1, 6, 24, 48):
        values = {}
        for model in ("lightgbm", "sarima"):
            row = metrics[(metrics.model == model) & (metrics.horizon == horizon)].iloc[0]
            values[model] = row
        left, right = values["lightgbm"], values["sarima"]
        rows.append(
            f"{horizon} & {left.rmse:.3f} & {display(left.skill_rmse)} & "
            f"{left.variance_retention_pct:.1f} & {display(left.skill_vp)} & "
            f"{right.rmse:.3f} & {display(right.skill_rmse)} & "
            f"{right.variance_retention_pct:.1f} & {display(right.skill_vp)}" + r" \\"
        )
    output = Path("outputs/tables/paper_a_rolling_results.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")


if __name__ == "__main__":
    main()
