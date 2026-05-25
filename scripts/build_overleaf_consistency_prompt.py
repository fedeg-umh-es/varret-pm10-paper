#!/usr/bin/env python3
"""Build a self-contained Overleaf editing prompt from verified repo outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "outputs" / "tables" / "variance_retention_all_stations.csv"
PROMPT_PATH = ROOT / "outputs" / "prompts" / "overleaf_consistency_patch_prompt.md"
JSON_PATH = ROOT / "outputs" / "prompts" / "verified_consistency_facts.json"

COLLAPSE_THRESHOLD = 0.5
SEASONAL_REFERENCE_THRESHOLDS = (0.95, 0.8)

EXPECTED_ARTEFACTS = [
    ROOT / "outputs" / "tables" / "collapse_rates_summary.tex",
    ROOT / "outputs" / "tables" / "collapse_rates_summary.csv",
    ROOT / "outputs" / "tables" / "alpha_threshold_sensitivity.tex",
    ROOT / "outputs" / "tables" / "alpha_threshold_sensitivity.csv",
    ROOT / "outputs" / "figures" / "station_map_ml_only_collapse_rate.pdf",
    ROOT / "outputs" / "figures" / "station_map_ml_only_collapse_rate.png",
    ROOT / "outputs" / "tables" / "station_ml_only_collapse_rates.csv",
    ROOT / "outputs" / "audit" / "bootstrap_alpha_audit.md",
    ROOT / "outputs" / "audit" / "bootstrap_alpha_audit.csv",
    ROOT / "outputs" / "audit" / "alpha_threshold_sensitivity_summary.md",
    ROOT / "outputs" / "audit" / "station_map_ml_only_collapse_rate_caption.md",
]

CORE_FIGURES = [
    ROOT / "outputs" / "figures" / "figure2_skill_variance_retention.pdf",
    ROOT / "outputs" / "figures" / "figure3_skill_profiles.pdf",
    ROOT / "outputs" / "figures" / "figure4_alpha_profiles.pdf",
    ROOT / "outputs" / "figures" / "figure5_scatter_skill_alpha.pdf",
    ROOT / "outputs" / "figures" / "figure6_station_collapse_rates.pdf",
    ROOT / "outputs" / "figures" / "figure7_station_map_spain.pdf",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def locate_unified_table() -> Path:
    if DEFAULT_TABLE.exists():
        return DEFAULT_TABLE
    candidates = []
    for path in (ROOT / "outputs" / "tables").glob("*.csv"):
        try:
            cols = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        score = 0
        for group in (
            {"station", "station_name", "name"},
            {"model", "model_name"},
            {"h", "horizon"},
            {"alpha", "variance_retention", "variance_retention_ratio"},
        ):
            if cols & group:
                score += 1
        if score >= 4:
            candidates.append((path.stat().st_size, path))
    if not candidates:
        raise FileNotFoundError("Could not locate a unified variance-retention CSV under outputs/tables.")
    return sorted(candidates, reverse=True)[0][1]


def detect_column(columns: Iterable[str], candidates: tuple[str, ...], required_name: str) -> str:
    col_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    raise ValueError(f"Could not identify required {required_name} column. Tried: {candidates}")


def optional_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    col_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def normalize_table(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = pd.read_csv(path)
    columns = df.columns
    col = {
        "station": detect_column(columns, ("station_name", "station", "name"), "station"),
        "model": detect_column(columns, ("model", "model_name"), "model"),
        "horizon": detect_column(columns, ("horizon", "h"), "horizon"),
        "alpha": detect_column(
            columns,
            ("alpha", "variance_retention", "variance_retention_ratio", "variance_retention_alpha"),
            "alpha",
        ),
        "skill": optional_column(columns, ("skill", "persistence_relative_skill", "skill_score")),
        "skill_vp": optional_column(columns, ("skill_vp", "SkillVP", "skillvp", "skill_vp_adjusted")),
        "alpha_ci_low": optional_column(columns, ("alpha_ci_low", "ci_low", "alpha_low", "alpha_lower")),
        "alpha_ci_high": optional_column(columns, ("alpha_ci_high", "ci_high", "alpha_high", "alpha_upper")),
        "station_type": optional_column(columns, ("station_type", "type", "area_type")),
        "station_class": optional_column(columns, ("station_class", "class")),
        "station_id": optional_column(columns, ("station_id", "station_code", "id")),
    }
    return df, col


def detect_models(df: pd.DataFrame, model_col: str) -> tuple[list[str], str]:
    models = sorted(str(model) for model in df[model_col].dropna().unique())
    seasonal = [m for m in models if "seasonal" in m.lower() and "naive" in m.lower()]
    if not seasonal:
        raise ValueError(f"Could not identify seasonal naive model in models: {models}")
    seasonal_model = seasonal[0]

    preferred = [m for m in ("hgb_direct", "ridge_direct") if m in models]
    if len(preferred) >= 2:
        return preferred, seasonal_model

    direct_ml = [
        m
        for m in models
        if "direct" in m.lower()
        and "seasonal" not in m.lower()
        and "naive" not in m.lower()
        and "persistence" not in m.lower()
    ]
    if not direct_ml:
        raise ValueError(f"Could not identify direct ML models in models: {models}")
    return sorted(direct_ml), seasonal_model


def pct(collapsed: int, total: int) -> float:
    return round(100.0 * collapsed / total, 1) if total else float("nan")


def count_for(df: pd.DataFrame, alpha_col: str) -> dict[str, int | float]:
    collapsed = int((df[alpha_col] < COLLAPSE_THRESHOLD).sum())
    total = int(len(df))
    return {"collapsed": collapsed, "total": total, "percentage": pct(collapsed, total)}


def table_counts(df: pd.DataFrame, col: dict[str, str | None], ml_models: list[str], seasonal_model: str) -> dict:
    model_col = str(col["model"])
    alpha_col = str(col["alpha"])
    per_model = {}
    for model, group in df.groupby(model_col, sort=True):
        per_model[str(model)] = count_for(group, alpha_col)
    ml_df = df[df[model_col].isin(ml_models)]
    seasonal_df = df[df[model_col].eq(seasonal_model)]
    return {
        "per_model_counts": per_model,
        "ml_combined_counts": count_for(ml_df, alpha_col),
        "seasonal_naive_counts": count_for(seasonal_df, alpha_col),
        "all_model_counts": count_for(df, alpha_col),
    }


def noncollapsed_ml(df: pd.DataFrame, col: dict[str, str | None], ml_models: list[str]) -> list[dict]:
    model_col = str(col["model"])
    alpha_col = str(col["alpha"])
    station_col = str(col["station"])
    horizon_col = str(col["horizon"])
    station_id_col = col["station_id"]
    mask = df[model_col].isin(ml_models) & (df[alpha_col] >= COLLAPSE_THRESHOLD)
    fields = [station_col, model_col, horizon_col, alpha_col]
    if station_id_col:
        fields.insert(1, station_id_col)
    if col["alpha_ci_low"] and col["alpha_ci_high"]:
        fields.extend([str(col["alpha_ci_low"]), str(col["alpha_ci_high"])])
    out = df.loc[mask, fields].sort_values([station_col, model_col, horizon_col])
    records = []
    for row in out.to_dict(orient="records"):
        records.append({str(k): normalize_json_value(v) for k, v in row.items()})
    return records


def station_rates(df: pd.DataFrame, col: dict[str, str | None], models: list[str] | None) -> list[dict]:
    model_col = str(col["model"])
    alpha_col = str(col["alpha"])
    station_col = str(col["station"])
    source = df[df[model_col].isin(models)] if models else df
    group_cols = [station_col]
    for maybe_col in (col["station_id"], col["station_type"], col["station_class"]):
        if maybe_col and maybe_col not in group_cols:
            group_cols.append(maybe_col)
    tmp = source.copy()
    tmp["_collapsed"] = tmp[alpha_col] < COLLAPSE_THRESHOLD
    rates = (
        tmp.groupby(group_cols, sort=True)["_collapsed"]
        .agg(collapsed="sum", total="count")
        .reset_index()
        .sort_values([station_col])
    )
    rates["percentage"] = (rates["collapsed"] / rates["total"] * 100).round(1)
    return [{str(k): normalize_json_value(v) for k, v in row.items()} for row in rates.to_dict(orient="records")]


def bootstrap_summary(df: pd.DataFrame, col: dict[str, str | None], ml_models: list[str], seasonal_model: str) -> dict | None:
    low_col = col["alpha_ci_low"]
    high_col = col["alpha_ci_high"]
    if not low_col or not high_col:
        return None
    model_col = str(col["model"])
    station_col = str(col["station"])
    horizon_col = str(col["horizon"])
    alpha_col = str(col["alpha"])
    station_id_col = col["station_id"]

    ml = df[df[model_col].isin(ml_models)]
    seasonal = df[df[model_col].eq(seasonal_model)]

    ml_below = int((ml[high_col] < COLLAPSE_THRESHOLD).sum())
    ml_cross = ml[(ml[low_col] <= COLLAPSE_THRESHOLD) & (COLLAPSE_THRESHOLD <= ml[high_col])]
    ml_above = int((ml[low_col] > COLLAPSE_THRESHOLD).sum())

    fields = [station_col, model_col, horizon_col, alpha_col, low_col, high_col]
    if station_id_col:
        fields.insert(1, station_id_col)
    crossing_records = []
    for row in ml_cross[fields].sort_values([model_col, station_col, horizon_col]).to_dict(orient="records"):
        crossing_records.append({str(k): normalize_json_value(v) for k, v in row.items()})

    seasonal_counts = {}
    for threshold in SEASONAL_REFERENCE_THRESHOLDS:
        key = str(threshold)
        seasonal_counts[key] = {
            "ci_low_above_threshold": int((seasonal[low_col] > threshold).sum()),
            "ci_crosses_threshold": int(((seasonal[low_col] <= threshold) & (threshold <= seasonal[high_col])).sum()),
            "total": int(len(seasonal)),
        }

    return {
        "available": True,
        "ml_alpha_0_5": {
            "ci_high_below_0_5": ml_below,
            "ci_crosses_0_5": int(len(ml_cross)),
            "ci_low_above_0_5": ml_above,
            "total": int(len(ml)),
            "crossing_cells": crossing_records,
        },
        "seasonal_naive": seasonal_counts,
    }


def station_type_summary(df: pd.DataFrame, col: dict[str, str | None]) -> dict:
    station_col = str(col["station"])
    type_col = col["station_type"]
    class_col = col["station_class"]
    if not type_col and not class_col:
        return {"available": False, "heterogeneous": None, "station_types": [], "station_classes": []}
    station_meta_cols = [station_col]
    if type_col:
        station_meta_cols.append(type_col)
    if class_col:
        station_meta_cols.append(class_col)
    meta = df[station_meta_cols].drop_duplicates()
    types = sorted(str(v) for v in meta[type_col].dropna().unique()) if type_col else []
    classes = sorted(str(v) for v in meta[class_col].dropna().unique()) if class_col else []
    return {
        "available": True,
        "heterogeneous": len(types or classes) > 1,
        "station_types": types,
        "station_classes": classes,
    }


def detect_artefacts() -> dict:
    paths = EXPECTED_ARTEFACTS + CORE_FIGURES
    detected = {}
    for path in paths:
        detected[rel(path)] = path.exists()
    return dict(sorted(detected.items()))


def normalize_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    str_rows = []
    for row in rows:
        str_rows.append([format_cell(row.get(col, "")) for col in columns])
    widths = [
        max(len(col), *(len(row[idx]) for row in str_rows)) if str_rows else len(col)
        for idx, col in enumerate(columns)
    ]
    header = "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in str_rows
    ]
    return "\n".join([header, divider, *body])


def format_cell(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def model_label(model: str) -> str:
    labels = {
        "hgb_direct": "HGB direct",
        "ridge_direct": "Ridge direct",
        "seasonal_naive": "Seasonal naive",
    }
    return labels.get(model, model)


def counts_sentence(name: str, counts: dict) -> str:
    return f"{name}: {counts['collapsed']}/{counts['total']} = {counts['percentage']:.1f}%"


def build_prompt(facts: dict, table_path: Path) -> str:
    counts = facts["counts"]
    bootstrap = facts["bootstrap_ci_summary"]
    artefacts = facts["artefacts_detected"]
    missing = [path for path, exists in artefacts.items() if not exists and path in {rel(p) for p in EXPECTED_ARTEFACTS}]
    existing_uploads = [path for path, exists in artefacts.items() if exists and path.startswith("outputs/")]

    noncollapsed = facts["non_collapsed_ml_cells"]
    exception_lines = []
    for cell in noncollapsed:
        station = cell[facts["column_map"]["station"]]
        model = cell[facts["column_map"]["model"]]
        horizon = cell[facts["column_map"]["horizon"]]
        alpha = float(cell[facts["column_map"]["alpha"]])
        exception_lines.append(f"- {station}, `{model}`, h={horizon}, alpha={alpha:.3f} (exact {alpha:.6f}).")

    per_model_rows = []
    for model, item in counts["per_model_counts"].items():
        per_model_rows.append(
            {
                "Model/group": model_label(model),
                "Collapsed": item["collapsed"],
                "Total": item["total"],
                "Rate (%)": f"{item['percentage']:.1f}",
            }
        )
    aggregate_rows = [
        {
            "Model/group": "ML combined",
            "Collapsed": counts["ml_combined_counts"]["collapsed"],
            "Total": counts["ml_combined_counts"]["total"],
            "Rate (%)": f"{counts['ml_combined_counts']['percentage']:.1f}",
        },
        {
            "Model/group": "All models",
            "Collapsed": counts["all_model_counts"]["collapsed"],
            "Total": counts["all_model_counts"]["total"],
            "Rate (%)": f"{counts['all_model_counts']['percentage']:.1f}",
        },
    ]

    ml_station_rows = [
        {
            "Station": row[facts["column_map"]["station"]],
            "Collapsed": row["collapsed"],
            "Total": row["total"],
            "Rate (%)": f"{row['percentage']:.1f}",
        }
        for row in facts["per_station_ml_only_rates"]
    ]
    all_station_rows = [
        {
            "Station": row[facts["column_map"]["station"]],
            "Collapsed": row["collapsed"],
            "Total": row["total"],
            "Rate (%)": f"{row['percentage']:.1f}",
        }
        for row in facts["per_station_all_model_rates"]
    ]

    type_summary = facts["station_type_summary"]
    station_type_text = (
        "The station set is heterogeneous. Station type labels in the table are: "
        + "; ".join(type_summary["station_types"])
        + "."
        if type_summary["available"] and type_summary["heterogeneous"]
        else "Station type labels were not available in the verified CSV; avoid any station-type generalization."
    )

    if bootstrap:
        ml_ci = bootstrap["ml_alpha_0_5"]
        seasonal_ci = bootstrap["seasonal_naive"]
        bootstrap_text = (
            f"For ML models, {ml_ci['ci_high_below_0_5']}/{ml_ci['total']} intervals have alpha_ci_high < 0.5; "
            f"{ml_ci['ci_crosses_0_5']}/{ml_ci['total']} cross 0.5; "
            f"{ml_ci['ci_low_above_0_5']}/{ml_ci['total']} have alpha_ci_low > 0.5. "
            f"For seasonal naive, alpha_ci_low > 0.95 occurs in "
            f"{seasonal_ci['0.95']['ci_low_above_threshold']}/{seasonal_ci['0.95']['total']} cells and "
            f"{seasonal_ci['0.95']['ci_crosses_threshold']}/{seasonal_ci['0.95']['total']} intervals cross 0.95; "
            f"alpha_ci_low > 0.8 occurs in {seasonal_ci['0.8']['ci_low_above_threshold']}/{seasonal_ci['0.8']['total']} cells and "
            f"{seasonal_ci['0.8']['ci_crosses_threshold']}/{seasonal_ci['0.8']['total']} intervals cross 0.8."
        )
        crossing_rows = [
            {
                "Station": row[facts["column_map"]["station"]],
                "Model": row[facts["column_map"]["model"]],
                "h": row[facts["column_map"]["horizon"]],
                "alpha": f"{float(row[facts['column_map']['alpha']]):.3f}",
                "CI low": f"{float(row[facts['column_map']['alpha_ci_low']]):.3f}",
                "CI high": f"{float(row[facts['column_map']['alpha_ci_high']]):.3f}",
            }
            for row in ml_ci["crossing_cells"]
        ]
        bootstrap_crossing_table = md_table(crossing_rows, ["Station", "Model", "h", "alpha", "CI low", "CI high"])
    else:
        bootstrap_text = (
            "Bootstrap interval columns were not available in the verified CSV. Bootstrap wording must be checked "
            "against an available bootstrap output file before inclusion; do not invent CI claims."
        )
        bootstrap_crossing_table = "_Bootstrap columns absent in verified CSV._"

    files_lines = []
    for path in existing_uploads:
        files_lines.append(f"- `{Path(path).name}` from `{path}`")
    for path in missing:
        files_lines.append(f"- `{Path(path).name}`: Do not include this artefact unless it has been generated and uploaded.")

    prompt = f"""# Overleaf Editing Prompt: PM10 Variance-Retention Consistency Patch

You are editing the Overleaf LaTeX manuscript for the PM10 variance-retention paper. Do not ask Codex to patch repository `.tex` files. Use only the verified facts and artefact filenames in this prompt.

## Objective

Revise the Overleaf manuscript so that all numerical claims, bootstrap confidence-interval wording, station-type descriptions, and the station-map caption are consistent with the verified repository outputs generated from `{rel(table_path)}`.

## Verified Facts From Repository

- Verified input table: `{rel(table_path)}`.
- Station count: {facts['station_count']}.
- Model count: {facts['model_count']}.
- Horizon count: {facts['horizon_count']}.
- Models exactly as in the table: {", ".join(f'`{m}`' for m in facts['models'])}.
- ML models: {", ".join(f'`{m}`' for m in facts['ml_models'])}.
- Seasonal naive model: `{facts['seasonal_model']}`.
- Collapsed cell definition: alpha < 0.5.
- ML-only station denominator: {len(facts['ml_models'])} ML models x {facts['horizon_count']} horizons = {facts['ml_station_denominator']} cells per station.
- All-model station denominator: {facts['model_count']} models x {facts['horizon_count']} horizons = {facts['all_model_station_denominator']} cells per station.

### Collapse Counts

{md_table(per_model_rows + aggregate_rows, ["Model/group", "Collapsed", "Total", "Rate (%)"])}

Use these exact prose forms:

- {counts_sentence('HGB direct', counts['per_model_counts'].get('hgb_direct', {}))}
- {counts_sentence('Ridge direct', counts['per_model_counts'].get('ridge_direct', {}))}
- {counts_sentence('ML combined', counts['ml_combined_counts'])}
- {counts_sentence('Seasonal naive', counts['seasonal_naive_counts'])}
- {counts_sentence('All models', counts['all_model_counts'])}

### Non-Collapsed ML Point-Estimate Exceptions

There are {len(noncollapsed)} ML cells with alpha >= 0.5:

{chr(10).join(exception_lines)}

### Per-Station ML-Only Collapse Rates

{md_table(ml_station_rows, ["Station", "Collapsed", "Total", "Rate (%)"])}

### Per-Station All-Model Collapse Rates

{md_table(all_station_rows, ["Station", "Collapsed", "Total", "Rate (%)"])}

### Bootstrap CI Audit

{bootstrap_text}

ML cells whose bootstrap intervals cross alpha = 0.5:

{bootstrap_crossing_table}

### Station-Type Wording

{station_type_text}

Do not describe all stations as background stations. Use "monitoring stations" or "heterogeneous MITECO monitoring stations" unless a sentence refers to a specific station type.

## Files To Upload/Use In Overleaf

Upload or use these generated artefacts exactly as named:

{chr(10).join(files_lines)}

For the ML-only map, use `station_map_ml_only_collapse_rate.pdf` as the preferred manuscript figure file. `station_map_ml_only_collapse_rate.png` is available for quick visual checks but should not be the primary LaTeX include if PDF is accepted.

## Required Manuscript Edits

1. Replace any claim that ML collapse is 100% or complete across all ML cells with the exact counts: HGB direct 118/119 = 99.2%, Ridge direct 118/119 = 99.2%, and ML combined 236/238 = 99.2%.
2. State that seasonal naive is variance-preserving in point-estimate collapse terms: 0/119 collapsed cells = 0.0%.
3. State that the all-model collapse rate is 236/357 = 66.1%, and explain that it is lower because seasonal naive contributes 0 collapsed cells.
4. Replace any station-map text or caption that implies 3-model station collapse rates if using the ML-only map. The ML-only map denominator is 14 cells per station.
5. Add or update a short threshold-sensitivity statement only if `alpha_threshold_sensitivity.tex` has been uploaded.
6. Add or update bootstrap CI wording using the safe wording below. Do not claim all ML intervals are below 0.5. Do not claim all seasonal naive intervals are above 0.95.
7. Replace any statement that all stations are background with a heterogeneous station-set description.

## Safe Replacement Text

### Results

Across 17 stations, 7 horizons, and 3 evaluated models, the variance-retention collapse diagnostic identified 236 collapsed cells out of 357 total cells (66.1%). The model-specific pattern was strongly asymmetric: `hgb_direct` collapsed in 118/119 cells (99.2%), `ridge_direct` collapsed in 118/119 cells (99.2%), and `seasonal_naive` collapsed in 0/119 cells (0.0%). Combining the two direct ML models gives 236/238 collapsed cells (99.2%). The two non-collapsed ML point-estimate exceptions occurred at h=1: Huesca for `hgb_direct` with alpha = {next(float(c[facts['column_map']['alpha']]) for c in noncollapsed if c[facts['column_map']['station']] == 'Huesca' and c[facts['column_map']['model']] == 'hgb_direct'):.3f}, and Barcelona Vall d'Hebron for `ridge_direct` with alpha = {next(float(c[facts['column_map']['alpha']]) for c in noncollapsed if c[facts['column_map']['station']] == "Barcelona Vall d'Hebron" and c[facts['column_map']['model']] == 'ridge_direct'):.3f}.

### Bootstrap

Bootstrap intervals support a robust aggregate separation between smoothed ML forecasts and variance-preserving seasonal naive, while selected near-threshold h=1 ML cells have intervals that overlap the diagnostic collapse boundary.

### Discussion

The central conclusion is not that every ML confidence interval lies below the collapse threshold, but that the aggregate point-estimate and interval evidence consistently separates the two direct ML models from the seasonal naive reference. The direct ML models show near-universal variance-retention collapse by point estimate, whereas seasonal naive preserves variance by this diagnostic and contributes no collapsed point-estimate cells.

### Station Description

The evaluated station set consists of 17 heterogeneous MITECO monitoring stations spanning multiple station-type labels, including traffic, industrial, urban/background, rural/remote, residential, and EMEP-related settings as represented in the verified table. Do not characterize the full station set as background-only.

### Figure 6 Caption For ML-Only Map

Geographic distribution of the 17 MITECO stations. Marker colour denotes station type. Marker size denotes the ML-only collapse rate, computed over 14 model-horizon cells per station (2 direct ML models x 7 horizons). Fifteen stations show 14/14 collapsed ML cells; Huesca and Barcelona Vall d'Hebron show 13/14 due to one non-collapsed h=1 cell each.

### Table Text For Collapse Summary

When referring to `collapse_rates_summary.tex`, say: "The all-model rate is lower than the ML-only rate because seasonal naive is included in the all-model denominator and contributes 0 collapsed cells."

### Threshold Sensitivity

If `alpha_threshold_sensitivity.tex` is included, use this wording: "Diagnostic threshold sensitivity supports the same qualitative conclusion: the combined ML collapse rate remains high around the alpha < 0.5 diagnostic boundary, while seasonal naive remains non-collapsed at the tested thresholds."

## Do Not Say

- Do not say "100% ML collapse" for the combined ML result; the verified value is 236/238 = 99.2%.
- Do not say "complete collapse" without qualifying that there are two h=1 ML point-estimate exceptions.
- Do not say all stations are background stations.
- Do not say all ML bootstrap confidence intervals are below alpha = 0.5.
- Do not say all seasonal naive bootstrap intervals lie above alpha = 0.95.
- Do not use a station-map caption that says or implies marker size is based on 3 models if the figure file is `station_map_ml_only_collapse_rate.pdf`.
- Do not mix all-model station rates (21 cells per station) with ML-only station rates (14 cells per station).

## Final Overleaf Verification Checklist

- The manuscript compiles after uploading all selected artefacts.
- Every included filename matches the exact uploaded filename.
- Collapse summary values are HGB 118/119 = 99.2%, Ridge 118/119 = 99.2%, ML combined 236/238 = 99.2%, seasonal naive 0/119 = 0.0%, and all models 236/357 = 66.1%.
- The two non-collapsed ML point-estimate exceptions are named exactly: Huesca (`hgb_direct`, h=1) and Barcelona Vall d'Hebron (`ridge_direct`, h=1).
- Any bootstrap paragraph uses the safe wording above and does not make universal CI claims contradicted by the audit.
- If the ML-only map is used, its caption states the 14-cell denominator and mentions 15 stations at 14/14 and two stations at 13/14.
- The station description says the station set is heterogeneous, not background-only.
"""
    return prompt


def build_facts() -> tuple[dict, Path]:
    table_path = locate_unified_table()
    df, col = normalize_table(table_path)
    model_col = str(col["model"])
    horizon_col = str(col["horizon"])
    station_col = str(col["station"])
    ml_models, seasonal_model = detect_models(df, model_col)
    counts = table_counts(df, col, ml_models, seasonal_model)
    bootstrap = bootstrap_summary(df, col, ml_models, seasonal_model)
    artefacts = detect_artefacts()
    station_count = int(df[station_col].nunique())
    model_count = int(df[model_col].nunique())
    horizon_count = int(df[horizon_col].nunique())
    facts = {
        "input_table": rel(table_path),
        "column_map": col,
        "station_count": station_count,
        "model_count": model_count,
        "horizon_count": horizon_count,
        "models": sorted(str(v) for v in df[model_col].dropna().unique()),
        "ml_models": ml_models,
        "seasonal_model": seasonal_model,
        "ml_station_denominator": len(ml_models) * horizon_count,
        "all_model_station_denominator": model_count * horizon_count,
        "counts": counts,
        "per_model_counts": counts["per_model_counts"],
        "ml_combined_counts": counts["ml_combined_counts"],
        "seasonal_naive_counts": counts["seasonal_naive_counts"],
        "all_model_counts": counts["all_model_counts"],
        "non_collapsed_ml_cells": noncollapsed_ml(df, col, ml_models),
        "per_station_all_model_rates": station_rates(df, col, None),
        "per_station_ml_only_rates": station_rates(df, col, ml_models),
        "bootstrap_ci_summary": bootstrap,
        "station_type_summary": station_type_summary(df, col),
        "artefacts_detected": artefacts,
    }
    return facts, table_path


def write_outputs() -> tuple[dict, list[str]]:
    facts, table_path = build_facts()
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(facts, table_path)
    PROMPT_PATH.write_text(prompt, encoding="utf-8")
    JSON_PATH.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = [path for path, exists in facts["artefacts_detected"].items() if not exists and path in {rel(p) for p in EXPECTED_ARTEFACTS}]
    return facts, missing


def main() -> None:
    facts, missing = write_outputs()
    counts = facts["counts"]
    print(f"Overleaf prompt: {PROMPT_PATH}")
    print(f"JSON summary: {JSON_PATH}")
    print(
        "Key verified counts: "
        f"ML combined {counts['ml_combined_counts']['collapsed']}/{counts['ml_combined_counts']['total']} "
        f"= {counts['ml_combined_counts']['percentage']:.1f}%; "
        f"seasonal naive {counts['seasonal_naive_counts']['collapsed']}/{counts['seasonal_naive_counts']['total']} "
        f"= {counts['seasonal_naive_counts']['percentage']:.1f}%; "
        f"all models {counts['all_model_counts']['collapsed']}/{counts['all_model_counts']['total']} "
        f"= {counts['all_model_counts']['percentage']:.1f}%."
    )
    if missing:
        print("Missing expected artefacts:")
        for path in missing:
            print(f"- {path}")
    else:
        print("Missing expected artefacts: none")


if __name__ == "__main__":
    main()
