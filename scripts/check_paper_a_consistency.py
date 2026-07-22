#!/usr/bin/env python3
"""Fail when Paper A diverges from its canonical reproduction artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 6, 24, 48)
MODELS = ("lightgbm", "sarima")


def _display(value: float) -> str:
    formatted = f"{value:.3f}"
    return f"${formatted}$" if value < 0 else formatted


def expected_table(metrics: pd.DataFrame) -> str:
    rows: list[str] = []
    for horizon in HORIZONS:
        values = {
            model: metrics.loc[
                metrics["model"].eq(model) & metrics["horizon"].eq(horizon)
            ].iloc[0]
            for model in MODELS
        }
        left, right = values["lightgbm"], values["sarima"]
        rows.append(
            f"{horizon} & {left.rmse:.3f} & {_display(left.skill_rmse)} & "
            f"{left.variance_retention_pct:.1f} & {_display(left.skill_vp)} & "
            f"{right.rmse:.3f} & {_display(right.skill_rmse)} & "
            f"{right.variance_retention_pct:.1f} & {_display(right.skill_vp)}" + r" \\"
        )
    return "\n".join(rows) + "\n\\bottomrule\n"


def run_checks() -> None:
    metrics = pd.read_csv(ROOT / "outputs/reproduction/metrics_rolling_origin.csv")
    events = pd.read_csv(ROOT / "outputs/reproduction/events_p75_rolling_origin.csv")
    predictions = pd.read_parquet(
        ROOT / "outputs/reproduction/predictions_rolling_origin.parquet"
    )
    manuscript = (ROOT / "paper_a.tex").read_text(encoding="utf-8")
    rendered_table = (ROOT / "outputs/tables/paper_a_rolling_results.tex").read_text(
        encoding="utf-8"
    )

    expected_cells = {(model, horizon) for model in MODELS for horizon in HORIZONS}
    assert set(zip(metrics.model, metrics.horizon)) == expected_cells
    assert set(zip(events.model, events.horizon)) == expected_cells
    assert rendered_table == expected_table(metrics)

    comparison_columns = [
        "fold", "origin_time", "target_time", "horizon", "y_true",
        "y_persistence", "p75_train",
    ]
    left = predictions.loc[predictions.model.eq("lightgbm"), comparison_columns]
    right = predictions.loc[predictions.model.eq("sarima"), comparison_columns]
    pd.testing.assert_frame_equal(
        left.sort_values(comparison_columns[:4]).reset_index(drop=True),
        right.sort_values(comparison_columns[:4]).reset_index(drop=True),
        check_dtype=False,
    )

    normalized_manuscript = " ".join(manuscript.split())
    required_text = (
        "LightGBM skill ranges from $-0.079$ to $0.073$",
        "SARIMA skill ranges from $0.031$ to $0.132$",
        "2.9\\% and 0.4\\% at $h=24$ and $48$",
        "precision at $h=48$ is undefined, rather than zero",
        "exactly the same valid",
        "not a regulatory",
        "Protocol sensitivity",
    )
    for phrase in required_text:
        assert phrase in normalized_manuscript, (
            f"Required canonical statement missing: {phrase}"
        )

    forbidden_text = (
        "ghost-skill-paper-a",
        "European Environment Agency",
        "LightGBM reach",  # catches legacy 0.92--0.96 claims
        "highly skilled models",
        "Protocol robustness",
    )
    for phrase in forbidden_text:
        assert phrase not in manuscript, f"Forbidden legacy statement found: {phrase}"

    sarima_48 = events.loc[
        events.model.eq("sarima") & events.horizon.eq(48)
    ].iloc[0]
    assert sarima_48.flag_rate == 0
    assert pd.isna(sarima_48.precision)

    for figure in ("figure1_skill_variance", "figure2_skillvp_events"):
        for suffix in ("png", "pdf"):
            path = ROOT / "outputs/figures" / f"{figure}.{suffix}"
            assert path.is_file() and path.stat().st_size > 0


if __name__ == "__main__":
    run_checks()
    print("Paper A consistency checks passed")
