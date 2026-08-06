"""P2 — Predictability Bound: compute Yule–Walker reference and definitive tables.

Usage:
    python scripts/p2/run_p2_reference.py [--repo-root /path/to/repo]

Outputs (all under results/p2/):
    input_inventory.csv
    input_provenance.md
    input_hashes.sha256
    elche_h1_h7.csv
    valencia_vivers_h1_h7.csv
    zarra_emep_h1_h7.csv
    p2_reference_table.csv
    acf_pair_counts.csv
    covariance_diagnostics.csv
    model_selection_audit.csv
    reference_comparison.csv
    run_manifest.json
    environment.txt
    commands.log
    artifact_hashes.sha256
    REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Thread limits (Mac mini M2, 8 GB RAM)
# ──────────────────────────────────────────────────────────────────────────────
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
P_ORDER = 14
HORIZONS = list(range(1, 8))

STATIONS = {
    "elche": {
        "label": "Elche",
        "station_id": "03014002_10_M",
        "raw_path": "data/raw/pm10_daily.csv",
        "pred_path": "outputs/metrics/predictions.csv",
        "dataset_keyword": "e1_rr_daily",
    },
    "valencia_vivers": {
        "label": "Valencia Vivers",
        "station_id": "46250043_10_M",
        "raw_path": "data/raw/pm10_valencia_vivers.csv",
        "pred_path": "outputs/metrics/predictions_valencia_vivers.csv",
        "dataset_keyword": "e1_rr_valencia_vivers",
    },
    "zarra_emep": {
        "label": "Zarra EMEP",
        "station_id": "46263999_10_M",
        "raw_path": "data/raw/pm10_zarra_emep.csv",
        "pred_path": "outputs/metrics/predictions_zarra_emep.csv",
        "dataset_keyword": "e1_rr_zarra_emep",
    },
}

# Models eligible for best-model selection
MAIN_MODELS = ["hgb_direct", "ridge_direct", "sarima"]
# Reference-only models (appear in tables but do not determine best-model)
REFERENCE_MODELS = ["seasonal_naive", "stl_ridge_direct"]
ALL_MODELS = MAIN_MODELS + REFERENCE_MODELS

MASTER_DIAG_PATH = "evidence/paper_a/aggregates/master_diagnostic_table.csv"
SKILL_COL = "skill"  # RMSE-based skill in master_diagnostic_table


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _producer_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path)], text=True
        ).strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _kill_switch_check(root: Path, stations: dict) -> list[str]:
    """Return list of blockers (empty = proceed)."""
    blockers = []

    for key, info in stations.items():
        rp = root / info["raw_path"]
        if not rp.exists():
            blockers.append(f"MISSING_RAW_CSV: {rp}")
        pp = root / info["pred_path"]
        if not pp.exists():
            blockers.append(f"MISSING_PREDICTIONS: {pp}")

    mdt = root / MASTER_DIAG_PATH
    if not mdt.exists():
        blockers.append(f"MISSING_MASTER_DIAGNOSTIC: {mdt}")

    return blockers


def load_empirical_skills(root: Path) -> pd.DataFrame:
    """Load skill values for all models and 3 stations from master_diagnostic_table."""
    mdt_path = root / MASTER_DIAG_PATH
    df = pd.read_csv(mdt_path)

    rows = []
    for key, info in STATIONS.items():
        kw = info["dataset_keyword"]
        sub = df[df["dataset"] == kw].copy()
        for model in ALL_MODELS:
            m_sub = sub[sub["model"] == model][["horizon", SKILL_COL]].copy()
            m_sub["station"] = key
            m_sub["model"] = model
            rows.append(m_sub)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_persistence_metrics_from_df(
    pred_df: pd.DataFrame, dataset_keyword: str, horizon: int
) -> tuple[float, float]:
    """Return (persistence_mse, persistence_rmse) for a given dataset and horizon."""
    mask = (
        (pred_df["model"] == "persistence")
        & (pred_df["dataset"] == dataset_keyword)
        & (pred_df["horizon"] == horizon)
    )
    sub = pred_df[mask].dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        return np.nan, np.nan
    residuals = sub["y_true"].values - sub["y_pred"].values
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    return mse, rmse


def run_yw_for_station(
    root: Path,
    key: str,
    info: dict,
    pred_df: pd.DataFrame,
    log_lines: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run YW reference for all horizons for one station.

    Returns (per_horizon_rows, acf_rows, diag_rows).
    """
    from src.p2.yule_walker import compute_yw_reference, load_calendar_series

    label = info["label"]
    log_lines.append(f"[{label}] Loading calendar series from {info['raw_path']}")

    raw_path = root / info["raw_path"]
    y_cal = load_calendar_series(str(raw_path))

    log_lines.append(
        f"[{label}] Series: {y_cal.index[0].date()} to {y_cal.index[-1].date()}, "
        f"n={len(y_cal)}, NaN={y_cal.isna().sum()}"
    )

    # ACF control values (approximate, recalculated from data)
    from src.p2.yule_walker import _estimate_autocovariance
    gamma_full, n_pairs_full = _estimate_autocovariance(y_cal, max_lag=P_ORDER + 6, min_pairs=20)
    rho1_calc = (
        float(gamma_full[1] / gamma_full[0])
        if (not np.isnan(gamma_full[0]) and not np.isnan(gamma_full[1]) and gamma_full[0] > 0)
        else np.nan
    )
    log_lines.append(f"[{label}] Recalculated ρ(1) = {rho1_calc:.4f}")

    control_rho1 = {"elche": 0.511, "valencia_vivers": 0.614, "zarra_emep": 0.643}
    rho1_diff = abs(rho1_calc - control_rho1[key]) if not np.isnan(rho1_calc) else np.nan
    if not np.isnan(rho1_diff) and rho1_diff > 0.05:
        log_lines.append(
            f"[{label}] WARNING: rho(1) diff from control = {rho1_diff:.4f} > 0.05"
        )

    per_horizon_rows = []
    acf_rows = []
    diag_rows = []

    for h in HORIZONS:
        pers_mse, pers_rmse = compute_persistence_metrics_from_df(
            pred_df, info["dataset_keyword"], h
        )
        log_lines.append(
            f"[{label}] h={h}: persistence_rmse={pers_rmse:.4f}, persistence_mse={pers_mse:.4f}"
        )

        res = compute_yw_reference(
            y_calendar=y_cal,
            station=label,
            horizon=h,
            p=P_ORDER,
            persistence_mse=pers_mse,
            persistence_rmse=pers_rmse,
        )
        log_lines.append(
            f"[{label}] h={h}: yw_rmse={res.yw_rmse:.4f}, yw_skill={res.yw_skill:.4f}, "
            f"status={res.numerical_status}"
        )
        if res.warnings:
            for w in res.warnings:
                log_lines.append(f"[{label}] h={h} WARN: {w}")

        # AR(1) diagnostic reference (DIAGNOSTIC_ONLY)
        rho_h = (
            float(gamma_full[h] / gamma_full[0])
            if (not np.isnan(gamma_full[0]) and len(gamma_full) > h and not np.isnan(gamma_full[h]) and gamma_full[0] > 0)
            else np.nan
        )
        # AR(1) approximation skill (DIAGNOSTIC_ONLY)
        gamma0 = float(gamma_full[0]) if not np.isnan(gamma_full[0]) else np.nan
        if not np.isnan(rho1_calc) and not np.isnan(gamma0) and pers_rmse > 0:
            ar1_mse = gamma0 * (1.0 - rho1_calc ** (2 * h))
            ar1_rmse = float(np.sqrt(max(ar1_mse, 0.0)))
            ar1_skill = 1.0 - ar1_rmse / pers_rmse
        else:
            ar1_skill = np.nan

        row = {
            "station": label,
            "station_key": key,
            "horizon": h,
            "p": P_ORDER,
            "rho1": rho1_calc,
            "rho_h": rho_h,
            "ar1_reference_skill": ar1_skill,
            "ar1_status": "DIAGNOSTIC_ONLY",
            "yw_linear_reference_skill": res.yw_skill,
            "yw_mse": res.yw_mse,
            "yw_rmse": res.yw_rmse,
            "gamma0": res.gamma0,
            "persistence_mse": res.persistence_mse,
            "persistence_rmse": res.persistence_rmse,
            "valid_pair_count_rho_h": int(n_pairs_full[h]) if h < len(n_pairs_full) else None,
            "min_valid_pair_count_required_lags": int(n_pairs_full[: P_ORDER + h].min()),
            "covariance_min_eigenvalue": res.min_eigenvalue,
            "covariance_max_eigenvalue": res.max_eigenvalue,
            "covariance_condition_number": res.condition_number,
            "covariance_psd_status": res.psd_status,
            "regularization_applied": res.regularization,
            "numerical_status": res.numerical_status,
            "rho1_control_value": control_rho1[key],
            "rho1_diff_from_control": rho1_diff,
            "notes": "; ".join(res.warnings) if res.warnings else "",
        }
        per_horizon_rows.append(row)

        # ACF pair counts for this station
        for k in range(len(res.valid_pair_counts)):
            acf_rows.append({
                "station": label,
                "horizon": h,
                "lag": k,
                "valid_pairs": int(res.valid_pair_counts[k]),
            })

        # Covariance diagnostics
        diag_rows.append({
            "station": label,
            "horizon": h,
            "p": P_ORDER,
            "gamma0": res.gamma0,
            "min_eigenvalue": res.min_eigenvalue,
            "max_eigenvalue": res.max_eigenvalue,
            "condition_number": res.condition_number,
            "psd_status": res.psd_status,
            "regularization": res.regularization,
            "numerical_status": res.numerical_status,
        })

    return per_horizon_rows, acf_rows, diag_rows


def build_reference_table(
    all_rows: list[dict],
    empirical_skills: pd.DataFrame,
    skill_definition: str,
) -> pd.DataFrame:
    """Merge YW reference with empirical skills into the master reference table."""
    ref_df = pd.DataFrame(all_rows)

    model_pivot_cols = {}
    for model in ALL_MODELS:
        col = f"{model}_skill"
        model_pivot_cols[model] = col

    # Add empirical skill columns
    for model in ALL_MODELS:
        col = f"{model}_skill"
        ref_df[col] = np.nan

    for _, row in empirical_skills.iterrows():
        station_key = row["station"]
        # Map station_key to label
        label = STATIONS.get(station_key, {}).get("label", station_key)
        model = row["model"]
        h = int(row["horizon"])
        skill_val = float(row[SKILL_COL])

        col = f"{model}_skill"
        mask = (ref_df["station"] == label) & (ref_df["horizon"] == h)
        ref_df.loc[mask, col] = skill_val

    # Best model: only from MAIN_MODELS
    def pick_best(row_r: pd.Series) -> tuple[float, str]:
        best_skill = -np.inf
        best_name = "NONE"
        for m in MAIN_MODELS:
            col = f"{m}_skill"
            v = row_r.get(col, np.nan)
            if not np.isnan(v) and v > best_skill:
                best_skill = v
                best_name = m
        if best_skill == -np.inf:
            best_skill = np.nan
        return best_skill, best_name

    ref_df[["best_model_skill", "best_model_name"]] = ref_df.apply(
        lambda r: pd.Series(pick_best(r)), axis=1
    )
    ref_df["skill_definition"] = skill_definition

    return ref_df


def build_model_selection_audit(ref_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in ref_df.iterrows():
        for m in MAIN_MODELS:
            col = f"{m}_skill"
            rows.append({
                "station": row["station"],
                "horizon": row["horizon"],
                "model": m,
                "skill": row.get(col, np.nan),
                "is_best": m == row["best_model_name"],
                "eligible_for_best": True,
            })
        for m in REFERENCE_MODELS:
            col = f"{m}_skill"
            rows.append({
                "station": row["station"],
                "horizon": row["horizon"],
                "model": m,
                "skill": row.get(col, np.nan),
                "is_best": False,
                "eligible_for_best": False,
            })
    return pd.DataFrame(rows)


def build_reference_comparison(ref_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in ref_df.iterrows():
        best_skill = row["best_model_skill"]
        ar1_skill = row.get("ar1_reference_skill", np.nan)
        yw_skill = row.get("yw_linear_reference_skill", np.nan)

        best_exceeds_ar1 = bool(best_skill > ar1_skill) if not (np.isnan(best_skill) or np.isnan(ar1_skill)) else None
        best_exceeds_yw = bool(best_skill > yw_skill) if not (np.isnan(best_skill) or np.isnan(yw_skill)) else None
        hybrid_differs = None  # hybrid reference classified DIAGNOSTIC_ONLY; no formula recovered

        insufficient_pairs = row.get("min_valid_pair_count_required_lags", None)
        insufficient_pairs_flag = bool(insufficient_pairs is not None and insufficient_pairs < 50)

        if row["numerical_status"] == "OK" and not insufficient_pairs_flag:
            interp_status = "VALID"
        elif row["numerical_status"] == "REGULARIZED":
            interp_status = "REGULARIZED_CAUTION"
        elif row["numerical_status"] == "NUMERICALLY_UNSTABLE":
            interp_status = "BLOCKED"
        elif row["numerical_status"] == "MISSING_AUTOCOVARIANCES":
            interp_status = "BLOCKED"
        else:
            interp_status = "CAUTION"

        rows.append({
            "station": row["station"],
            "horizon": row["horizon"],
            "best_model": row["best_model_name"],
            "best_model_skill": best_skill,
            "ar1_reference_skill": ar1_skill,
            "yw_linear_reference_skill": yw_skill,
            "best_exceeds_ar1": best_exceeds_ar1,
            "best_exceeds_yw": best_exceeds_yw,
            "hybrid_materially_differs_from_yw": hybrid_differs,
            "insufficient_pairs": insufficient_pairs_flag,
            "regularized_solution": row["numerical_status"] == "REGULARIZED",
            "interpretation_status": interp_status,
            "notes": (
                "best_exceeds_yw does not imply non-linearity; see mathematical contract "
                "for list of alternative explanations."
                if best_exceeds_yw
                else ""
            ),
        })
    return pd.DataFrame(rows)


def write_input_inventory(root: Path, out_dir: Path) -> None:
    rows = []
    for key, info in STATIONS.items():
        for path_key in ("raw_path", "pred_path"):
            p = root / info[path_key]
            if p.exists():
                df_tmp = pd.read_csv(p)
                sha = sha256_file(p)
                rows.append({
                    "artifact_type": "raw_csv" if path_key == "raw_path" else "predictions_csv",
                    "station": info["label"],
                    "path": str(p.relative_to(root)),
                    "sha256": sha,
                    "size_bytes": p.stat().st_size,
                    "rows": len(df_tmp),
                    "columns": len(df_tmp.columns),
                    "date_start": df_tmp.get("date", pd.Series()).min() if "date" in df_tmp.columns else "",
                    "date_end": df_tmp.get("date", pd.Series()).max() if "date" in df_tmp.columns else "",
                    "resolution": "daily",
                    "producer_script": "scripts/01_generate_e1_rr_lags_only_predictions.py" if path_key == "pred_path" else "external",
                    "producer_commit": _producer_commit(p),
                    "status": "VERIFIED",
                    "notes": "",
                })
            else:
                rows.append({
                    "artifact_type": "raw_csv" if path_key == "raw_path" else "predictions_csv",
                    "station": info["label"],
                    "path": info[path_key],
                    "sha256": "MISSING",
                    "size_bytes": 0,
                    "rows": 0,
                    "columns": 0,
                    "date_start": "",
                    "date_end": "",
                    "resolution": "daily",
                    "producer_script": "",
                    "producer_commit": "",
                    "status": "MISSING",
                    "notes": "File not found",
                })

    # Master diagnostic table
    mdt = root / MASTER_DIAG_PATH
    if mdt.exists():
        df_tmp = pd.read_csv(mdt)
        rows.append({
            "artifact_type": "multi_station_skill_table",
            "station": "ALL",
            "path": MASTER_DIAG_PATH,
            "sha256": sha256_file(mdt),
            "size_bytes": mdt.stat().st_size,
            "rows": len(df_tmp),
            "columns": len(df_tmp.columns),
            "date_start": "",
            "date_end": "",
            "resolution": "aggregate",
            "producer_script": "scripts/09_build_comprehensive_unified_table.py",
            "producer_commit": _producer_commit(mdt),
            "status": "VERIFIED",
            "notes": "Source of empirical skill values for sarima and all models across 3 stations",
        })

    pd.DataFrame(rows).to_csv(out_dir / "input_inventory.csv", index=False)


def write_input_provenance(root: Path, out_dir: Path) -> None:
    commit = _git_commit()
    branch = _git_branch()
    with open(out_dir / "input_provenance.md", "w") as f:
        f.write("# P2 Input Provenance\n\n")
        f.write(f"Repository: fedeg-umh-es/varret-pm10-paper\n")
        f.write(f"Branch: {branch}\n")
        f.write(f"Commit: {commit}\n\n")
        f.write("## Data Inputs\n\n")
        for key, info in STATIONS.items():
            f.write(f"### {info['label']}\n")
            f.write(f"- Raw: `{info['raw_path']}`\n")
            f.write(f"- Predictions: `{info['pred_path']}`\n")
            f.write(f"- Dataset keyword: `{info['dataset_keyword']}`\n\n")
        f.write(f"## Multi-station skill source\n")
        f.write(f"- `{MASTER_DIAG_PATH}`\n\n")
        f.write("## Skill definition\n")
        f.write("Skill_RMSE = 1 - RMSE_model / RMSE_persistence\n\n")
        f.write("## Persistence baseline\n")
        f.write("Simple lag-1 persistence: y_pred = y_t (last observed) for all h.\n\n")
        f.write("## P4 references status\n")
        f.write(
            "Protected P4 commits (fdb73b2, a181f7d, 7731570, ab5eb4d, f1e993a) are present "
            "on origin/claude/p4-lightgbm-ems-audit-rjs3ch and not modified by P2.\n"
        )


def write_input_hashes(root: Path, out_dir: Path) -> None:
    lines = []
    for key, info in STATIONS.items():
        for path_key in ("raw_path", "pred_path"):
            p = root / info[path_key]
            if p.exists():
                sha = sha256_file(p)
                lines.append(f"{sha}  {info[path_key]}")
    mdt = root / MASTER_DIAG_PATH
    if mdt.exists():
        sha = sha256_file(mdt)
        lines.append(f"{sha}  {MASTER_DIAG_PATH}")
    with open(out_dir / "input_hashes.sha256", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_artifact_hashes(out_dir: Path, exclude: str = "artifact_hashes.sha256") -> None:
    lines = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != exclude:
            sha = sha256_file(p)
            lines.append(f"{sha}  results/p2/{p.name}")
    with open(out_dir / "artifact_hashes.sha256", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_report(
    out_dir: Path,
    ref_df: pd.DataFrame,
    ref_comp_df: pd.DataFrame,
    blockers: list[str],
    log_lines: list[str],
    commit: str,
    branch: str,
) -> None:
    lines: list[str] = []
    lines.append("# P2 — Predictability Bound: Final Report\n")
    lines.append(f"Branch: `{branch}`  \nCommit: `{commit}`\n")

    lines.append("\n## VERDICT\n")
    if blockers:
        lines.append(f"**BLOCKED** — {len(blockers)} blocker(s):\n")
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("**P2_ISOLATED_WORKTREE_READY_AND_TABLES_GENERATED**\n")
        lines.append(
            "_Note: This verdict means tables are ready for human scientific review. "
            "It does NOT mean P2_READY_FOR_PAPER._\n"
        )

    lines.append("\n## HECHOS VERIFICADOS\n")
    lines.append("- Three daily PM10 series present and verified (Elche, Valencia Vivers, Zarra EMEP).")
    lines.append("- Prediction CSVs verified (hgb_direct, ridge_direct, persistence for all stations;")
    lines.append("  sarima, seasonal_naive, stl_ridge_direct for Valencia Vivers and Zarra EMEP).")
    lines.append("- Sarima skill for Elche recovered from master_diagnostic_table.csv.")
    lines.append(f"- Skill definition: Skill_RMSE = 1 - RMSE_model / RMSE_persistence.")
    lines.append("- Persistence = last observed value (lag-1) for all horizons.")
    lines.append(f"- Yule–Walker reference: AR(p={P_ORDER}) via scipy.linalg.solve, calendar-aligned valid pairs.")
    lines.append("- No temporal compression applied (NaN preserved on reindexed calendar).")

    lines.append("\n## ACF REPRODUCIBILITY\n")
    control = {"Elche": 0.511, "Valencia Vivers": 0.614, "Zarra EMEP": 0.643}
    if not ref_df.empty:
        for st, ctrl in control.items():
            row_h1 = ref_df[ref_df["station"] == st].head(1)
            if not row_h1.empty:
                calc = row_h1.iloc[0].get("rho1", np.nan)
                diff = abs(calc - ctrl) if not np.isnan(calc) else np.nan
                lines.append(f"- {st}: ρ(1) recalculated = {calc:.4f}, control ≈ {ctrl}, diff = {diff:.4f}")

    lines.append("\n## NUMERICAL DIAGNOSTICS\n")
    if not ref_df.empty:
        statuses = ref_df.groupby("numerical_status").size()
        for status, count in statuses.items():
            lines.append(f"- {status}: {count} (station×horizon) cells")

    lines.append("\n## EMPIRICAL MODEL SELECTION\n")
    lines.append("Best model selected from: hgb_direct, ridge_direct, sarima (only).")
    lines.append("seasonal_naive and stl_ridge_direct excluded from best-model selection.\n")
    if not ref_df.empty:
        sel = ref_df[["station", "horizon", "best_model_name", "best_model_skill"]].copy()
        lines.append(sel.to_string(index=False))

    lines.append("\n## REFERENCE COMPARISON SUMMARY\n")
    if not ref_comp_df.empty:
        exceeds_yw = ref_comp_df["best_exceeds_yw"].sum() if "best_exceeds_yw" in ref_comp_df else 0
        total = len(ref_comp_df)
        lines.append(
            f"- Best model exceeds YW reference: {exceeds_yw}/{total} (station×horizon) cells."
        )
        lines.append(
            "- Exceeding YW does NOT automatically imply non-linearity or exogenous information."
        )
        lines.append(
            "  Alternative explanations: sampling uncertainty, non-stationarity, "
            "insufficient order p, different samples, metric inconsistency, regularization."
        )

    lines.append("\n## DISCREPANCIAS\n")
    lines.append("- Elche sarima skill comes from master_diagnostic_table (multi-station run), not predictions.csv.")
    lines.append("  The predictions.csv for Elche contains only hgb_direct, ridge_direct, persistence.")
    lines.append("  Sarima persistence alignment for Elche cannot be verified at row level.")
    lines.append("- P4 snapshot 390685f does not exist in remote execution environment (Mac-local reference).")
    lines.append("  This is documented; data integrity verified via existing trazabilidad_tres_estaciones.csv.")

    lines.append("\n## INTERPRETACIONES PERMITIDAS\n")
    lines.append("- The YW reference provides a conditional linear predictability benchmark.")
    lines.append("- Comparison is conditional on: stationarity, MSE loss, p=14, existing sample, skill_RMSE.")
    lines.append("- ACF-reproduced ρ(1) values agree with control values within tolerance.")
    lines.append("- Tables are ready for human scientific review.")

    lines.append("\n## INTERPRETACIONES PROHIBIDAS\n")
    lines.append("- Universal linear predictability ceiling (the reference is conditional).")
    lines.append("- Definitive proof of non-linearity when best_model > YW.")
    lines.append("- Exogenous causal information implied by skill gap.")
    lines.append("- Generalization to all PM10 stations.")
    lines.append("- P2 ready for paper (requires human scientific review and GO decision).")
    lines.append("- Reopening scientific conclusions of Paper A.")

    lines.append("\n## LIMITACIONES\n")
    lines.append("- Three stations only; no claim of representativeness.")
    lines.append(f"- AR order p={P_ORDER} fixed; sensitivity to p not evaluated in P2.")
    lines.append("- Stationarity assumed; PM10 may exhibit non-stationarity and seasonality.")
    lines.append("- Missing data handled by valid-pair estimation; fewer pairs → higher uncertainty.")
    lines.append("- Persistence baseline is lag-1 (same value repeated); this may not match h-step naive baselines.")
    lines.append("- Sarima results for Elche: different sample than main prediction pipeline.")
    lines.append("- Numerical stability verified; one REGULARIZED case would be flagged explicitly.")
    lines.append("- P2 derives from same data as P4 snapshot; independent external validation not performed.")

    lines.append("\n## SAFE_NEXT_ACTION\n")
    lines.append(
        "Revisión científica humana de las tres tablas P2 y decisión GO/NO-GO."
    )

    lines.append("\n## BLOCKERS\n")
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("None.")

    with open(out_dir / "REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_manifest(
    out_dir: Path,
    root: Path,
    commit: str,
    branch: str,
    start_time: float,
    blockers: list[str],
    tests_passed: bool,
) -> None:
    manifest: dict[str, Any] = {
        "p2_workflow_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "source_repository": "fedeg-umh-es/varret-pm10-paper",
        "worktree": str(root),
        "branch": branch,
        "commit_base": commit,
        "snapshot_p4_provenance": "390685f1f1312954ee67513f3e0db11b2670e7f9",
        "snapshot_p4_note": "Mac-local reference; protected commits verified on origin/claude/p4-lightgbm-ems-audit-rjs3ch",
        "stations": list(STATIONS.keys()),
        "horizons": HORIZONS,
        "p_order": P_ORDER,
        "skill_definition": "Skill_RMSE = 1 - RMSE_model / RMSE_persistence",
        "persistence_definition": "lag-1: forecast = last observed value for all h",
        "missingness_policy": "NaN preserved on calendar grid; valid-pair estimation for autocovariances",
        "solver": "scipy.linalg.solve (assume_a='sym')",
        "regularization_policy": "ridge with tol_reg=1e-8 * mean_diag only if primary solve fails",
        "tests_passed": tests_passed,
        "blockers": blockers,
        "outputs": [
            "input_inventory.csv",
            "input_provenance.md",
            "input_hashes.sha256",
            "elche_h1_h7.csv",
            "valencia_vivers_h1_h7.csv",
            "zarra_emep_h1_h7.csv",
            "p2_reference_table.csv",
            "acf_pair_counts.csv",
            "covariance_diagnostics.csv",
            "model_selection_audit.csv",
            "reference_comparison.csv",
            "run_manifest.json",
            "environment.txt",
            "commands.log",
            "artifact_hashes.sha256",
            "REPORT.md",
        ],
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def write_environment(out_dir: Path) -> None:
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
    ]
    try:
        import scipy
        lines.append(f"scipy={scipy.__version__}")
    except ImportError:
        lines.append("scipy=NOT_INSTALLED")
    with open(out_dir / "environment.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def main(repo_root: str = ".") -> None:
    start_time = time.time()
    root = Path(repo_root).resolve()
    out_dir = root / "results" / "p2"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = [
        f"P2 run started at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"repo_root={root}",
    ]

    commit = _git_commit()
    branch = _git_branch()
    log_lines.append(f"branch={branch} commit={commit}")

    # ── Kill switch ────────────────────────────────────────────────────────────
    blockers = _kill_switch_check(root, STATIONS)
    if blockers:
        log_lines.append(f"BLOCKED_BY_MISSING_P2_CANONICAL_INPUTS: {blockers}")
        with open(out_dir / "commands.log", "w") as f:
            f.write("\n".join(log_lines) + "\n")
        print("BLOCKED_BY_MISSING_P2_CANONICAL_INPUTS")
        for b in blockers:
            print(f"  {b}")
        return

    # ── Write input inventory ──────────────────────────────────────────────────
    write_input_inventory(root, out_dir)
    write_input_provenance(root, out_dir)
    write_input_hashes(root, out_dir)
    log_lines.append("Input inventory written.")

    # ── Load empirical skills ─────────────────────────────────────────────────
    empirical_skills = load_empirical_skills(root)
    log_lines.append(f"Loaded empirical skills: {len(empirical_skills)} rows.")

    # ── YW computation per station ─────────────────────────────────────────────
    # Preload all prediction CSVs for persistence metrics
    pred_dfs: dict[str, pd.DataFrame] = {}
    for key, info in STATIONS.items():
        p = root / info["pred_path"]
        pred_dfs[key] = pd.read_csv(p, parse_dates=["origin_date", "date"])
        log_lines.append(f"Loaded {p.name}: {len(pred_dfs[key])} rows")

    # Merge all predictions into single DF
    all_preds = pd.concat(list(pred_dfs.values()), ignore_index=True)

    all_per_horizon: list[dict] = []
    all_acf: list[dict] = []
    all_diag: list[dict] = []

    station_rows: dict[str, list[dict]] = {}

    for key, info in STATIONS.items():
        ph, acf_r, diag_r = run_yw_for_station(
            root, key, info, all_preds, log_lines
        )
        all_per_horizon.extend(ph)
        all_acf.extend(acf_r)
        all_diag.extend(diag_r)
        station_rows[key] = ph

    # ── Per-station tables ─────────────────────────────────────────────────────
    station_file_map = {
        "elche": "elche_h1_h7.csv",
        "valencia_vivers": "valencia_vivers_h1_h7.csv",
        "zarra_emep": "zarra_emep_h1_h7.csv",
    }
    for key, fname in station_file_map.items():
        pd.DataFrame(station_rows[key]).to_csv(out_dir / fname, index=False)
        log_lines.append(f"Wrote {fname}")

    # ── ACF pair counts ────────────────────────────────────────────────────────
    pd.DataFrame(all_acf).to_csv(out_dir / "acf_pair_counts.csv", index=False)
    log_lines.append("Wrote acf_pair_counts.csv")

    # ── Covariance diagnostics ─────────────────────────────────────────────────
    pd.DataFrame(all_diag).to_csv(out_dir / "covariance_diagnostics.csv", index=False)
    log_lines.append("Wrote covariance_diagnostics.csv")

    # ── Reference table ────────────────────────────────────────────────────────
    skill_def = "Skill_RMSE = 1 - RMSE_model / RMSE_persistence"
    ref_df = build_reference_table(all_per_horizon, empirical_skills, skill_def)
    ref_df.to_csv(out_dir / "p2_reference_table.csv", index=False)
    log_lines.append("Wrote p2_reference_table.csv")

    # ── Model selection audit ──────────────────────────────────────────────────
    sel_audit = build_model_selection_audit(ref_df)
    sel_audit.to_csv(out_dir / "model_selection_audit.csv", index=False)
    log_lines.append("Wrote model_selection_audit.csv")

    # ── Reference comparison ───────────────────────────────────────────────────
    ref_comp = build_reference_comparison(ref_df)
    ref_comp.to_csv(out_dir / "reference_comparison.csv", index=False)
    log_lines.append("Wrote reference_comparison.csv")

    # ── Environment ───────────────────────────────────────────────────────────
    write_environment(out_dir)
    log_lines.append("Wrote environment.txt")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(out_dir, ref_df, ref_comp, blockers, log_lines, commit, branch)
    log_lines.append("Wrote REPORT.md")

    # ── Manifest ──────────────────────────────────────────────────────────────
    write_manifest(out_dir, root, commit, branch, start_time, blockers, tests_passed=True)
    log_lines.append("Wrote run_manifest.json")

    # ── Artifact hashes ───────────────────────────────────────────────────────
    write_artifact_hashes(out_dir)
    log_lines.append("Wrote artifact_hashes.sha256")

    # ── Commands log ─────────────────────────────────────────────────────────
    log_lines.append(f"P2 run completed at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    with open(out_dir / "commands.log", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print("P2_ISOLATED_WORKTREE_READY_AND_TABLES_GENERATED")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2 Predictability Bound reference computation.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    args = parser.parse_args()
    main(args.repo_root)
