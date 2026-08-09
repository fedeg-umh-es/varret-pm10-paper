#!/usr/bin/env python3
"""Audit the largest exact row-level subsets already present locally.

This wrapper only connects preserved prediction rows, preserved raw series,
and the existing DM/HLN/BH definitions. It never fits a model and never
reads the 595-row aggregate to choose or validate a result.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path("/private/tmp/p4_predictions_all_stations_gradea.csv")
MATRIX_OUT = ROOT / "audit/final_submission/GRADE_A_UNIT_MATRIX.csv"
DIAGNOSTICS_OUT = ROOT / "audit/final_submission/GRADE_A_SUBSET_DIAGNOSTICS.csv"

MODELS = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]
LOCAL = {
    "03014002_10_M": {
        "name": "Elx-Agroalimentari",
        "predictions": ROOT / "outputs/metrics/predictions.csv",
        "raw": ROOT / "data/raw/pm10_daily.csv",
        "dataset": "e1_rr_daily",
    },
    "46250043_10_M": {
        "name": "Valencia-Vivers",
        "predictions": ROOT / "outputs/metrics/predictions_valencia_vivers.csv",
        "raw": ROOT / "data/raw/pm10_valencia_vivers.csv",
        "dataset": "e1_rr_valencia_vivers",
    },
    "46263999_10_M": {
        "name": "Zarra-EMEP",
        "predictions": ROOT / "outputs/metrics/predictions_zarra_emep.csv",
        "raw": ROOT / "data/raw/pm10_zarra_emep.csv",
        "dataset": "e1_rr_zarra_emep",
    },
}


def load_dm_module():
    path = ROOT / "scripts/05_dm_significance.py"
    spec = importlib.util.spec_from_file_location("p4_dm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load existing DM implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DM = load_dm_module()


def bh_adjust(pvals: pd.Series) -> pd.Series:
    valid = pvals.dropna()
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    if valid.empty:
        return out
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    raw = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    out.loc[order] = np.minimum(monotone, 1.0)
    return out


def load_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"])
    value = "pm10" if "pm10" in raw.columns else "value"
    if value not in raw.columns:
        raise ValueError(f"No PM10 value column in {path}")
    raw[value] = pd.to_numeric(raw[value], errors="coerce")
    raw = raw[["date", value]].dropna().drop_duplicates("date").sort_values("date")
    return raw.rename(columns={value: "pm10"})


def unit_metrics(station_id: str, spec: dict) -> list[dict]:
    pred = pd.read_csv(spec["predictions"])
    pred["origin_date"] = pd.to_datetime(pred["origin_date"])
    pred["date"] = pd.to_datetime(pred["date"])
    pred["horizon"] = pred["horizon"].astype(int)
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    raw = load_raw(spec["raw"])
    history = pd.Series(raw["pm10"].to_numpy(float), index=raw["date"])
    baseline = pred[pred["model"].eq("persistence")].copy()
    rows: list[dict] = []

    for model in sorted(set(pred["model"]) - {"persistence"}):
        for horizon in sorted(pred.loc[pred["model"].eq(model), "horizon"].unique()):
            group = pred[(pred["model"].eq(model)) & (pred["horizon"].eq(horizon))].copy()
            expected_target = group["origin_date"] + pd.to_timedelta(int(horizon), unit="D")
            if not np.all(group["date"].to_numpy() == expected_target.to_numpy()):
                raise ValueError(f"Origin-target mismatch for {station_id}/{model}/h{horizon}")
            fold_as_date = pd.to_datetime(group["fold"], errors="coerce")
            if fold_as_date.isna().any() or not np.all(fold_as_date.to_numpy() == group["origin_date"].to_numpy()):
                raise ValueError(f"Fold-origin mismatch for {station_id}/{model}/h{horizon}")
            base = baseline[baseline["horizon"].eq(horizon)].copy()
            merged = group.merge(
                base[["fold", "date", "y_true", "y_pred"]].rename(
                    columns={"y_true": "y_true_base", "y_pred": "y_pred_base"}
                ),
                on=["fold", "date"],
                how="inner",
                validate="one_to_one",
            )
            if merged.empty or not np.allclose(merged["y_true"], merged["y_true_base"]):
                raise ValueError(f"Baseline mismatch for {station_id}/{model}/h{horizon}")

            thresholds = []
            for origin in merged["origin_date"]:
                train = history[history.index <= origin]
                thresholds.append(float(np.percentile(train.to_numpy(float), 75)) if not train.empty else np.nan)
            merged["p75_threshold"] = thresholds
            merged = merged.dropna(subset=["y_true", "y_pred", "y_pred_base", "p75_threshold"])
            y = merged["y_true"].to_numpy(float)
            yp = merged["y_pred"].to_numpy(float)
            yb = merged["y_pred_base"].to_numpy(float)
            dm_stat, dm_p = DM._dm_hln((y - yb) ** 2 - (y - yp) ** 2, int(horizon))
            rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
            rmse_base = float(np.sqrt(np.mean((y - yb) ** 2)))
            alpha = float(np.var(yp) / np.var(y)) if np.var(y) > 0 else np.nan
            recall = float(np.sum((y > merged["p75_threshold"].to_numpy(float)) & (yp > merged["p75_threshold"].to_numpy(float))) / np.sum(y > merged["p75_threshold"].to_numpy(float))) if np.sum(y > merged["p75_threshold"].to_numpy(float)) else np.nan
            rows.append({
                "station_id": station_id,
                "station_name": spec["name"],
                "model": model,
                "horizon": int(horizon),
                "n_pairs": int(len(merged)),
                "mae": float(np.mean(np.abs(y - yp))),
                "rmse": rmse,
                "rmse_persistence": rmse_base,
                "bias": float(np.mean(yp - y)),
                "skill": float(1.0 - rmse / rmse_base) if rmse_base > 0 else np.nan,
                "alpha": alpha,
                "std_ratio": float(np.std(yp) / np.std(y)) if np.std(y) > 0 else np.nan,
                "correlation": float(np.corrcoef(y, yp)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(yp) > 0 else np.nan,
                "p75_recall": recall,
                "dm_stat": float(dm_stat),
                "dm_pval_raw": float(dm_p),
                "p75_train_support": "YES",
                "row_level_source": str(spec["predictions"]),
            })

    table = pd.DataFrame(rows)
    table["dm_pval_bh"] = np.nan
    # Existing code applies BH within dataset, across all model/horizon tests.
    table.loc[:, "dm_pval_bh"] = bh_adjust(table["dm_pval_raw"])
    table["dm_significant"] = table["dm_pval_bh"] < 0.05
    table["rule_a"] = (table["skill"] > 0) & table["dm_significant"]
    table["rule_b"] = table["rule_a"] & (table["alpha"] >= 0.50) & (table["p75_recall"] >= 0.20)
    table["formal_discordance"] = table["rule_a"] & (table["p75_recall"] >= 0.20) & (table["alpha"] < 0.50)
    return table.to_dict("records")


def remote_inventory(station_ids: list[str]) -> dict[tuple[str, str, int], dict]:
    if not RELEASE.exists():
        raise FileNotFoundError(f"Expected previously downloaded release asset: {RELEASE}")
    release = pd.read_csv(RELEASE, usecols=["dataset", "model", "fold", "origin_date", "horizon", "date", "y_true", "y_pred"])
    dataset_to_station = {}
    for dataset in release["dataset"].drop_duplicates():
        if dataset == "e1_rr_daily":
            dataset_to_station[dataset] = "03014002_10_M"
        elif str(dataset).startswith("e1_rr_"):
            dataset_to_station[dataset] = str(dataset)[6:]
    out = {}
    for dataset, station_id in dataset_to_station.items():
        if station_id not in station_ids or station_id in LOCAL:
            continue
        sub = release[release["dataset"].eq(dataset)]
        for model in MODELS:
            for horizon in range(1, 8):
                rows = sub[(sub["model"].eq(model)) & (sub["horizon"].eq(horizon))]
                if rows.empty:
                    continue
                has_base = model != "sarima" and not sub[(sub["model"].eq("persistence")) & (sub["horizon"].eq(horizon))].empty
                out[(station_id, model, horizon)] = {
                    "row_level": "YES",
                    "baseline": "YES" if has_base else "NO",
                    "origin": "YES",
                    "target": "YES",
                    "fold": "YES",
                    "p75": "NO",
                    "reason": "No preserved full observed training series for exact train-derived P75 support" if has_base else "SARIMA-aligned persistence support missing; full observed training support also unavailable",
                    "source": "public release paper-a-row-level-evidence-v1",
                }
    return out


def main() -> None:
    metadata = pd.read_csv(ROOT / "evidence/paper_a/metadata/station_metadata.csv")
    station_ids = metadata["station_id"].astype(str).tolist()
    local_records: list[dict] = []
    for station_id, spec in LOCAL.items():
        local_records.extend(unit_metrics(station_id, spec))
    local_df = pd.DataFrame(local_records)
    local_keys = {(r["station_id"], r["model"], int(r["horizon"])) for r in local_records}
    remote = remote_inventory(station_ids)

    matrix = []
    for station_id in station_ids:
        for model in MODELS:
            for horizon in range(1, 8):
                key = (station_id, model, horizon)
                if key in local_keys:
                    matrix.append({
                        "station": station_id,
                        "model": model,
                        "horizon": horizon,
                        "y_true": "YES",
                        "y_pred": "YES",
                        "baseline": "YES",
                        "origin": "YES",
                        "target": "YES",
                        "fold": "YES",
                        "p75_train_support": "YES",
                        "exact_support": "YES",
                        "metric_chain_available": "YES",
                        "grade_a_candidate": "YES",
                        "reason_if_not": "",
                        "evidence_source": local_df.loc[(local_df.station_id == station_id) & (local_df.model == model) & (local_df.horizon == horizon), "row_level_source"].iloc[0],
                    })
                elif station_id in LOCAL:
                    matrix.append({
                        "station": station_id, "model": model, "horizon": horizon,
                        "y_true": "YES", "y_pred": "NO", "baseline": "YES",
                        "origin": "YES", "target": "YES", "fold": "YES",
                        "p75_train_support": "YES", "exact_support": "NO",
                        "metric_chain_available": "NO", "grade_a_candidate": "NO",
                        "reason_if_not": "No preserved row-level y_pred for this model at this station",
                        "evidence_source": str(LOCAL[station_id]["predictions"]),
                    })
                else:
                    info = remote.get(key, {})
                    matrix.append({
                        "station": station_id, "model": model, "horizon": horizon,
                        "y_true": info.get("row_level", "NO"), "y_pred": info.get("row_level", "NO"),
                        "baseline": info.get("baseline", "NO"), "origin": info.get("origin", "NO"),
                        "target": info.get("target", "NO"), "fold": info.get("fold", "NO"),
                        "p75_train_support": info.get("p75", "NO"),
                        "exact_support": "NO", "metric_chain_available": "NO", "grade_a_candidate": "NO",
                        "reason_if_not": info.get("reason", "No exact row-level source inventory for this unit"),
                        "evidence_source": info.get("source", "none"),
                    })
    matrix_df = pd.DataFrame(matrix)
    MATRIX_OUT.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(MATRIX_OUT, index=False)

    candidates = {
        "C1_two_stations_five_models": {"stations": ["46250043_10_M", "46263999_10_M"], "models": MODELS},
        "C2_three_stations_two_models": {"stations": list(LOCAL), "models": ["hgb_direct", "ridge_direct"]},
    }
    diag_rows = []
    for name, spec in candidates.items():
        sub = local_df[local_df.station_id.isin(spec["stations"]) & local_df.model.isin(spec["models"])].copy()
        expected = len(spec["stations"]) * len(spec["models"]) * 7
        if len(sub) != expected:
            raise RuntimeError(f"Candidate {name} expected {expected} units, got {len(sub)}")
        summary = {
            "candidate": name,
            "stations": ",".join(spec["stations"]),
            "models": ",".join(spec["models"]),
            "horizons": "1..7",
            "cells": len(sub),
            "row_level_rows": int(sub["n_pairs"].sum()),
            "mae_median": float(sub["mae"].median()),
            "rmse_median": float(sub["rmse"].median()),
            "bias_median": float(sub["bias"].median()),
            "skill_median": float(sub["skill"].median()),
            "alpha_median": float(sub["alpha"].median()),
            "std_ratio_median": float(sub["std_ratio"].median()),
            "correlation_median": float(sub["correlation"].median()),
            "p75_recall_median": float(sub["p75_recall"].median()),
            "positive_skill_units": int((sub["skill"] > 0).sum()),
            "low_alpha_positive_skill_units": int(((sub["skill"] > 0) & (sub["alpha"] < 0.50)).sum()),
            "rule_a_units": int(sub["rule_a"].sum()),
            "rule_b_units": int(sub["rule_b"].sum()),
            "changed_units": int((sub["rule_a"] & ~sub["rule_b"]).sum()),
            "formal_discordant_units": int(sub["formal_discordance"].sum()),
            "event_support": "YES",
            "rank_reversal": "NOT_COMPUTED; no rank-reversal rule is part of this subset audit",
        }
        diag_rows.append(summary)
    pd.DataFrame(diag_rows).to_csv(DIAGNOSTICS_OUT, index=False)
    print(json.dumps({
        "matrix_rows": len(matrix_df),
        "grade_a_units": int((matrix_df.grade_a_candidate == "YES").sum()),
        "failed_units": int((matrix_df.grade_a_candidate == "NO").sum()),
        "local_units": len(local_df),
        "candidates": diag_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
