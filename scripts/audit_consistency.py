#!/usr/bin/env python3
"""Audit manuscript/output consistency for variance-retention results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "outputs" / "tables" / "variance_retention_all_stations.csv"
REPORT_PATH = ROOT / "outputs" / "audit" / "consistency_report.md"
FIGURE_SCRIPT = ROOT / "scripts" / "generate_figures_3_to_6.py"
ML_MODELS = {"hgb_direct", "ridge_direct"}
SEASONAL_MODEL = "seasonal_naive"

RISKY_PHRASES = [
    "100%",
    "complete collapse",
    "all stations are background",
    "background stations",
    "no confidence interval",
    "all confidence intervals",
    "above 0.95",
    "collapse rate encoded",
    "3 models",
    "2 models",
    "14/14",
    "21",
]

REPLACEMENTS = {
    "100%": "Use exact audited percentages for collapse claims; for variance-retention definitions, prefer `full variance retention` when a literal 100% is not needed.",
    "complete collapse": "Use `near-complete ML variance-retention collapse, with two h=1 exceptions`.",
    "all stations are background": "Use `the station set is heterogeneous, including traffic, industrial, urban/background, rural/remote and related classes`.",
    "background stations": "Use `monitoring stations` unless specifically referring only to stations labelled background.",
    "no confidence interval": "Use `the two non-collapsed ML exceptions have bootstrap intervals crossing alpha=0.5` where applicable.",
    "all confidence intervals": "Avoid universal CI claims; use `most ML intervals are below alpha=0.5, except audited h=1 exceptions` if supported by the table.",
    "above 0.95": "Use `seasonal naive has mean alpha approximately 0.986; interval-level claims should be checked cell by cell`.",
    "collapse rate encoded": "Specify whether the rate is `all-model: 3 models x 7 horizons = 21 cells per station` or `ML-only: 2 models x 7 horizons = 14 cells per station`.",
    "3 models": "Use only for all-model calculations that include hgb_direct, ridge_direct and seasonal_naive.",
    "2 models": "Use only for ML-only calculations that include hgb_direct and ridge_direct.",
    "14/14": "Use `14/14` only for ML-only station cells; the two exception stations are `13/14`.",
    "21": "Use `21` only for all-model station cells; because seasonal naive has no collapses, most all-model station rates are 14/21 and exception stations are 13/21.",
}


@dataclass(frozen=True)
class RiskyHit:
    path: Path
    line_no: int
    phrase: str
    line: str


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def find_main_tex(tex_files: list[Path]) -> Path | None:
    candidates = []
    for path in tex_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\\documentclass" in text and "\\begin{document}" in text:
            score = 0
            if "\\title" in text:
                score += 1
            if "\\bibliography" in text or "\\begin{thebibliography}" in text:
                score += 1
            candidates.append((score, len(text), path))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][2]


def manuscript_tex_files() -> list[Path]:
    excluded_dirs = {"outputs", "data", ".git", "__pycache__"}
    files = []
    for path in ROOT.rglob("*.tex"):
        if any(part in excluded_dirs for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def load_variance_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH)
    required = {"model", "horizon", "alpha", "station_id", "station_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{rel(TABLE_PATH)} is missing required columns: {sorted(missing)}")

    if "collapse_flag" in df.columns:
        # CSV booleans can arrive as bools or strings depending on writer/version.
        if df["collapse_flag"].dtype == bool:
            df["_collapse"] = df["collapse_flag"]
        else:
            df["_collapse"] = df["collapse_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        df["_collapse"] = df["alpha"] < 0.5

    df["_collapse_alpha"] = df["alpha"] < 0.5
    return df


def summarise_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[int, int, float]]]:
    by_model = (
        df.groupby("model", sort=True)["_collapse"]
        .agg(collapsed="sum", total="count")
        .reset_index()
    )
    by_model["percentage"] = by_model["collapsed"] / by_model["total"] * 100

    ml = df[df["model"].isin(ML_MODELS)]
    seasonal = df[df["model"].eq(SEASONAL_MODEL)]
    rollups = {
        "combined_ml": (int(ml["_collapse"].sum()), int(len(ml)), percent(int(ml["_collapse"].sum()), int(len(ml)))),
        "seasonal_naive": (
            int(seasonal["_collapse"].sum()),
            int(len(seasonal)),
            percent(int(seasonal["_collapse"].sum()), int(len(seasonal))),
        ),
        "total_all_models": (int(df["_collapse"].sum()), int(len(df)), percent(int(df["_collapse"].sum()), int(len(df)))),
    }
    return by_model, rollups


def station_rates(df: pd.DataFrame, ml_only: bool) -> pd.DataFrame:
    source = df[df["model"].isin(ML_MODELS)] if ml_only else df
    rates = (
        source.groupby(["station_id", "station_name"], sort=True)["_collapse"]
        .agg(collapsed="sum", total="count")
        .reset_index()
    )
    rates["percentage"] = rates["collapsed"] / rates["total"] * 100
    return rates.sort_values(["percentage", "station_name"], ascending=[True, True])


def risky_hits(tex_files: list[Path]) -> list[RiskyHit]:
    hits: list[RiskyHit] = []
    escaped_percent = re.compile(r"100\\?%")
    phrase_patterns = {
        phrase: re.compile(re.escape(phrase), re.IGNORECASE)
        for phrase in RISKY_PHRASES
        if phrase != "100%"
    }
    for path in tex_files:
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if escaped_percent.search(raw_line):
                hits.append(RiskyHit(path, line_no, "100%", raw_line.strip()))
            for phrase, pattern in phrase_patterns.items():
                if pattern.search(raw_line):
                    hits.append(RiskyHit(path, line_no, phrase, raw_line.strip()))
    return hits


def figure6_assessment() -> str:
    if not FIGURE_SCRIPT.exists():
        return "Figure 6 generation script not found."
    text = FIGURE_SCRIPT.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    relevant = []
    for idx, line in enumerate(lines, start=1):
        if "def figure6_collapse_rates" in line or "groupby(" in line and "collapse_flag" in "".join(lines[max(0, idx - 4) : idx + 4]):
            relevant.append(f"- `{rel(FIGURE_SCRIPT)}:{idx}` `{line.strip()}`")
        if "station x model x horizon" in line:
            relevant.append(f"- `{rel(FIGURE_SCRIPT)}:{idx}` `{line.strip()}`")
    figure6_match = re.search(
        r"def figure6_collapse_rates.*?df\.groupby\(\s*\[\"station_id\",\s*\"station_name\",\s*\"station_class\"\]\s*\)"
        r"\s*\[\"collapse_flag\"\]\s*\.mean\(\)",
        text,
        flags=re.DOTALL,
    )
    if figure6_match:
        verdict = "all-model"
    else:
        verdict = "undetermined from static script scan"
    return (
        f"Figure 6 appears to be **{verdict}**. The generation code averages `collapse_flag` over the full input table "
        "without filtering to ML models, and the axis label states `station x model x horizon cells`. "
        "With the current table this means 3 models x 7 horizons = 21 cells per station.\n\n"
        + "\n".join(dict.fromkeys(relevant))
    )


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = df[columns].astype(str).values.tolist()
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


def write_report() -> None:
    tex_files = manuscript_tex_files()
    output_tables = sorted((ROOT / "outputs" / "tables").glob("*")) if (ROOT / "outputs" / "tables").exists() else []
    main_tex = find_main_tex(tex_files)
    df = load_variance_table()
    by_model, rollups = summarise_counts(df)
    all_station = station_rates(df, ml_only=False)
    ml_station = station_rates(df, ml_only=True)
    hits = risky_hits(tex_files)
    mismatches = df[df["_collapse"] != df["_collapse_alpha"]]
    ml_exceptions = df[df["model"].isin(ML_MODELS) & ~df["_collapse"]].copy()
    seasonal = df[df["model"].eq(SEASONAL_MODEL)]

    report: list[str] = []
    report.append("# Consistency Audit Report")
    report.append("")
    report.append(f"Generated by `scripts/audit_consistency.py` from `{rel(TABLE_PATH)}`.")
    report.append("")
    report.append("## Located Inputs")
    report.append("")
    report.append(f"- Main manuscript: `{rel(main_tex) if main_tex else 'not found'}`")
    report.append("- Generated tables under `outputs/tables/`:")
    for path in output_tables:
        if path.is_file() and path.suffix in {".csv", ".tex"}:
            report.append(f"  - `{rel(path)}`")
    report.append("")
    report.append("## Verified Counts")
    report.append("")
    model_out = by_model.copy()
    model_out["collapsed"] = model_out["collapsed"].astype(int)
    model_out["total"] = model_out["total"].astype(int)
    model_out["percentage"] = model_out["percentage"].map(fmt_pct)
    report.append(md_table(model_out, ["model", "collapsed", "total", "percentage"]))
    report.append("")
    rollup_rows = pd.DataFrame(
        [
            {"scope": key, "collapsed": c, "total": t, "percentage": fmt_pct(p)}
            for key, (c, t, p) in rollups.items()
        ]
    )
    report.append(md_table(rollup_rows, ["scope", "collapsed", "total", "percentage"]))
    report.append("")
    report.append(f"- Seasonal naive mean alpha: `{seasonal['alpha'].mean():.3f}`.")
    report.append(f"- Collapse flag agrees with `alpha < 0.5` in `{len(df) - len(mismatches)}/{len(df)}` rows.")
    report.append("")
    report.append("## Non-Collapsed ML Exceptions")
    report.append("")
    exception_cols = ["station_name", "station_id", "model", "horizon", "alpha", "alpha_ci_low", "alpha_ci_high"]
    exception_out = ml_exceptions[exception_cols].copy()
    for col in ["alpha", "alpha_ci_low", "alpha_ci_high"]:
        exception_out[col] = exception_out[col].map(lambda x: f"{x:.6f}")
    report.append(md_table(exception_out, exception_cols))
    report.append("")
    report.append("## Per-Station Collapse Rates: All 3 Models")
    report.append("")
    all_out = all_station.copy()
    all_out["percentage"] = all_out["percentage"].map(fmt_pct)
    report.append(md_table(all_out, ["station_id", "station_name", "collapsed", "total", "percentage"]))
    report.append("")
    report.append("## Per-Station Collapse Rates: ML-Only")
    report.append("")
    ml_out = ml_station.copy()
    ml_out["percentage"] = ml_out["percentage"].map(fmt_pct)
    report.append(md_table(ml_out, ["station_id", "station_name", "collapsed", "total", "percentage"]))
    report.append("")
    report.append("## Risky Manuscript Locations")
    report.append("")
    if hits:
        hit_rows = pd.DataFrame(
            [
                {
                    "file": rel(hit.path),
                    "line": hit.line_no,
                    "phrase": hit.phrase,
                    "text": hit.line.replace("|", "\\|"),
                    "recommended replacement": REPLACEMENTS[hit.phrase].replace("|", "\\|"),
                }
                for hit in hits
            ]
        )
        report.append(md_table(hit_rows, ["file", "line", "phrase", "text", "recommended replacement"]))
    else:
        report.append("No risky phrase hits were found in `.tex` manuscript files for the configured phrase list.")
    report.append("")
    report.append("## Replacement Catalogue")
    report.append("")
    for phrase in RISKY_PHRASES:
        report.append(f"- `{phrase}`: {REPLACEMENTS[phrase]}")
    report.append("")
    report.append("## Figure 6 Assessment")
    report.append("")
    report.append(figure6_assessment())
    report.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_report()
    print(f"Wrote {REPORT_PATH}")
