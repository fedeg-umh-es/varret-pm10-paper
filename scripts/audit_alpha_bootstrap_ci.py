#!/usr/bin/env python3
"""Audit bootstrap confidence intervals for alpha diagnostics.

This script is intentionally descriptive: it checks interval positions relative
to diagnostic thresholds so manuscript wording does not overclaim universal
below/above-threshold behavior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "tables" / "variance_retention_all_stations.csv"
CSV_PATH = ROOT / "outputs" / "audit" / "bootstrap_alpha_audit.csv"
MD_PATH = ROOT / "outputs" / "audit" / "bootstrap_alpha_audit.md"

ML_MODELS = ("hgb_direct", "ridge_direct")
SEASONAL_MODEL = "seasonal_naive"
COLLAPSE_THRESHOLD = 0.5
SEASONAL_THRESHOLDS = (0.95, 0.8)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "station" in df.columns and "station_name" not in df.columns:
        rename["station"] = "station_name"
    if "h" in df.columns and "horizon" not in df.columns:
        rename["h"] = "horizon"
    df = df.rename(columns=rename)

    required = {
        "alpha",
        "alpha_ci_low",
        "alpha_ci_high",
        "model",
        "station_name",
        "horizon",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing required columns: {sorted(missing)}")
    return df


def count_interval_positions(df: pd.DataFrame, threshold: float, group: str) -> list[dict[str, object]]:
    below = int((df["alpha_ci_high"] < threshold).sum())
    crossing = int(((df["alpha_ci_low"] <= threshold) & (threshold <= df["alpha_ci_high"])).sum())
    above = int((df["alpha_ci_low"] > threshold).sum())
    total = int(len(df))
    return [
        {
            "section": group,
            "threshold": threshold,
            "category": "ci_high_below_threshold",
            "count": below,
            "total": total,
            "percentage": round(100.0 * below / total, 1) if total else float("nan"),
        },
        {
            "section": group,
            "threshold": threshold,
            "category": "ci_crosses_threshold",
            "count": crossing,
            "total": total,
            "percentage": round(100.0 * crossing / total, 1) if total else float("nan"),
        },
        {
            "section": group,
            "threshold": threshold,
            "category": "ci_low_above_threshold",
            "count": above,
            "total": total,
            "percentage": round(100.0 * above / total, 1) if total else float("nan"),
        },
    ]


def audit_counts(df: pd.DataFrame) -> pd.DataFrame:
    ml = df[df["model"].isin(ML_MODELS)]
    seasonal = df[df["model"].eq(SEASONAL_MODEL)]
    rows = count_interval_positions(ml, COLLAPSE_THRESHOLD, "ML models")

    for threshold in SEASONAL_THRESHOLDS:
        rows.extend(
            [
                {
                    "section": "seasonal_naive",
                    "threshold": threshold,
                    "category": "ci_low_above_threshold",
                    "count": int((seasonal["alpha_ci_low"] > threshold).sum()),
                    "total": int(len(seasonal)),
                    "percentage": round(100.0 * (seasonal["alpha_ci_low"] > threshold).sum() / len(seasonal), 1)
                    if len(seasonal)
                    else float("nan"),
                },
                {
                    "section": "seasonal_naive",
                    "threshold": threshold,
                    "category": "ci_crosses_threshold",
                    "count": int(
                        ((seasonal["alpha_ci_low"] <= threshold) & (threshold <= seasonal["alpha_ci_high"])).sum()
                    ),
                    "total": int(len(seasonal)),
                    "percentage": round(
                        100.0
                        * ((seasonal["alpha_ci_low"] <= threshold) & (threshold <= seasonal["alpha_ci_high"])).sum()
                        / len(seasonal),
                        1,
                    )
                    if len(seasonal)
                    else float("nan"),
                },
            ]
        )
    return pd.DataFrame(rows)


def crossing_cells(df: pd.DataFrame, threshold: float, models: tuple[str, ...] | None = None) -> pd.DataFrame:
    source = df[df["model"].isin(models)] if models else df
    mask = (source["alpha_ci_low"] <= threshold) & (threshold <= source["alpha_ci_high"])
    columns = ["station_name", "station_id", "model", "horizon", "alpha", "alpha_ci_low", "alpha_ci_high"]
    return source.loc[mask, columns].sort_values(["model", "station_name", "horizon"])


def noncollapsed_point_exceptions(df: pd.DataFrame) -> pd.DataFrame:
    ml = df[df["model"].isin(ML_MODELS)]
    columns = ["station_name", "station_id", "model", "horizon", "alpha", "alpha_ci_low", "alpha_ci_high"]
    return ml.loc[ml["alpha"] >= COLLAPSE_THRESHOLD, columns].sort_values(["model", "station_name", "horizon"])


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    out = df.copy()
    for col in ["alpha", "alpha_ci_low", "alpha_ci_high"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{float(x):.6f}")
    rows = out.astype(str).values.tolist()
    columns = list(out.columns)
    widths = [
        max(len(str(col)), *(len(row[idx]) for row in rows)) if rows else len(str(col))
        for idx, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_markdown(df: pd.DataFrame, counts: pd.DataFrame) -> None:
    ml_cross = crossing_cells(df, COLLAPSE_THRESHOLD, ML_MODELS)
    ml_exceptions = noncollapsed_point_exceptions(df)

    lines = [
        "# Bootstrap Alpha CI Audit",
        "",
        f"Input: `{INPUT_PATH.relative_to(ROOT)}`.",
        "",
        "## Summary Counts",
        "",
        md_table(counts),
        "",
        "## ML Intervals Crossing Alpha = 0.5",
        "",
        md_table(ml_cross),
        "",
        "## ML Non-Collapsed Point-Estimate Exceptions",
        "",
        md_table(ml_exceptions),
        "",
        "## Seasonal Naive CI Notes",
        "",
        "- Seasonal naive threshold positions are reported as counts in the summary table above.",
        "- The audit intentionally does not claim all seasonal naive intervals are above alpha = 0.95; the counts show they are not.",
        "- The audit also checks alpha = 0.8 as a less stringent reference for near variance preservation.",
        "",
        "## Safe Manuscript Wording",
        "",
        (
            "Bootstrap intervals support a robust aggregate separation between smoothed ML forecasts and "
            "variance-preserving seasonal naive, while selected near-threshold h=1 ML cells have intervals "
            "that overlap the diagnostic collapse boundary."
        ),
        "",
        "Do not state that all ML bootstrap intervals are below alpha = 0.5 unless this audit returns zero crossing/above-threshold ML intervals.",
        "Do not state that all seasonal naive bootstrap intervals are above alpha = 0.95 unless this audit returns zero crossing/below-threshold seasonal intervals.",
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def validate_known_exceptions(df: pd.DataFrame) -> None:
    exceptions = noncollapsed_point_exceptions(df)
    expected = {
        ("Barcelona Vall d'Hebron", "ridge_direct", 1),
        ("Huesca", "hgb_direct", 1),
    }
    found = set(zip(exceptions["station_name"], exceptions["model"], exceptions["horizon"]))
    missing = expected - found
    if missing:
        raise ValueError(f"Expected non-collapsed point-estimate exceptions not found: {sorted(missing)}")


def main() -> None:
    df = normalize_columns(pd.read_csv(INPUT_PATH))
    counts = audit_counts(df)
    validate_known_exceptions(df)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(CSV_PATH, index=False)
    write_markdown(df, counts)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
