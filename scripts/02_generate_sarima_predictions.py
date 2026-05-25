#!/usr/bin/env python3
"""Generate sparse-origin SARIMA predictions for daily PM10."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sarima_model import SarimaForecaster


HORIZONS = tuple(range(1, 8))
MIN_TRAIN_ROWS = 365
DEFAULT_PREDICTIONS_OUTPUT = Path("outputs/metrics/predictions_sarima.csv")
DEFAULT_SKILL_OUTPUT = Path("outputs/metrics/skill_sarima.csv")
DESIRABLE_OUTPUT = Path("outputs/tables/sarima_desirable_cells.csv")
FAILURE_LOG = Path("outputs/logs/sarima_failures.log")


def _load_daily_pm10(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = {"date", "pm10"} - set(raw.columns)
    if missing:
        raise ValueError(f"Input must contain date and pm10 columns. Missing: {sorted(missing)}")
    df = raw[["date", "pm10"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    daily = df.set_index("date").reindex(full_index).rename_axis("date").reset_index()
    daily["pm10"] = pd.to_numeric(daily["pm10"], errors="coerce")
    return daily


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float(np.sqrt(np.mean(err**2)))


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float(np.mean(np.abs(err)))


def _iter_origins(daily: pd.DataFrame) -> pd.Series:
    return daily.dropna(subset=["pm10"])["date"].reset_index(drop=True)


def _predict_one_origin(daily: pd.DataFrame, dataset: str, horizon: int, origin: pd.Timestamp) -> list[dict]:
    target_date = origin + pd.Timedelta(days=horizon)
    target = daily.loc[daily["date"].eq(target_date), "pm10"]
    if target.empty or pd.isna(target.iloc[0]):
        return []
    train_daily = daily[(daily["date"] <= origin) & daily["pm10"].notna()].copy()
    if len(train_daily) < MIN_TRAIN_ROWS:
        return []

    y_train = train_daily["pm10"].to_numpy(dtype=float)
    y_true = float(target.iloc[0])
    y_persistence = float(train_daily.iloc[-1]["pm10"])
    rows = [
        {
            "dataset": dataset,
            "model": "persistence",
            "fold": origin.strftime("%Y-%m-%d"),
            "origin_date": origin.strftime("%Y-%m-%d"),
            "horizon": horizon,
            "date": target_date.strftime("%Y-%m-%d"),
            "y_true": y_true,
            "y_pred": y_persistence,
        }
    ]
    try:
        model = SarimaForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
        model.fit(y_train)
        forecast = model.forecast(steps=horizon)
        rows.append(
            {
                "dataset": dataset,
                "model": "sarima",
                "fold": origin.strftime("%Y-%m-%d"),
                "origin_date": origin.strftime("%Y-%m-%d"),
                "horizon": horizon,
                "date": target_date.strftime("%Y-%m-%d"),
                "y_true": y_true,
                "y_pred": float(forecast[horizon - 1]),
            }
        )
    except Exception as exc:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FAILURE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{dataset},h={horizon},origin={origin.date()},{type(exc).__name__}: {exc}\n")
    return rows


def _generate_predictions(daily: pd.DataFrame, dataset: str, origin_step: int, n_jobs: int) -> pd.DataFrame:
    if origin_step < 1:
        raise ValueError("origin_step must be >= 1")
    origins = _iter_origins(daily).iloc[::origin_step].reset_index(drop=True)
    frames = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_predict_one_origin)(daily, dataset, horizon, origin)
        for horizon in HORIZONS
        for origin in origins
    )
    rows = [row for chunk in frames for row in chunk]
    if not rows:
        raise ValueError("No SARIMA predictions were generated.")
    return pd.DataFrame(rows).sort_values(["dataset", "model", "horizon", "origin_date"]).reset_index(drop=True)


def _build_skill_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[predictions["model"].eq("persistence")]
    rows = []
    for (dataset, model, horizon), group in predictions.groupby(["dataset", "model", "horizon"]):
        if model == "persistence":
            continue
        base = baseline[(baseline["dataset"].eq(dataset)) & (baseline["horizon"].eq(horizon))]
        merged = group.merge(
            base[["origin_date", "date", "y_pred"]].rename(columns={"y_pred": "y_pred_baseline"}),
            on=["origin_date", "date"],
            how="inner",
        )
        if merged.empty:
            continue
        rmse_model = _rmse(merged["y_true"], merged["y_pred"])
        rmse_baseline = _rmse(merged["y_true"], merged["y_pred_baseline"])
        mae_model = _mae(merged["y_true"], merged["y_pred"])
        mae_baseline = _mae(merged["y_true"], merged["y_pred_baseline"])
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "skill": 1 - rmse_model / rmse_baseline if rmse_baseline > 0 else np.nan,
                "mae_skill": 1 - mae_model / mae_baseline if mae_baseline > 0 else np.nan,
            }
        )
    if not rows:
        raise ValueError("No SARIMA skill rows generated.")
    return pd.DataFrame(rows).sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def _write_desirable_cells(predictions: pd.DataFrame, skill: pd.DataFrame) -> None:
    rows = []
    for (dataset, model, horizon), group in predictions[predictions["model"].eq("sarima")].groupby(
        ["dataset", "model", "horizon"]
    ):
        observed_var = float(np.var(group["y_true"].to_numpy(dtype=float), ddof=0))
        pred_var = float(np.var(group["y_pred"].to_numpy(dtype=float), ddof=0))
        alpha = pred_var / observed_var if observed_var > 0 else 0.0
        skill_row = skill[(skill["dataset"].eq(dataset)) & (skill["model"].eq(model)) & (skill["horizon"].eq(horizon))]
        rmse_skill = float(skill_row.iloc[0]["skill"]) if not skill_row.empty else np.nan
        if rmse_skill > 0 and alpha >= 0.8:
            rows.append({"dataset": dataset, "model": model, "horizon": horizon, "skill": rmse_skill, "alpha": alpha})
    if rows:
        DESIRABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(DESIRABLE_OUTPUT, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sparse-origin SARIMA predictions.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with date and pm10 columns")
    parser.add_argument("--dataset", required=True, help="Dataset identifier")
    parser.add_argument("--origin-step", type=int, default=14, help="Evaluate every kth observed origin")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs across horizon-origin fits")
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--skill-output", type=Path, default=DEFAULT_SKILL_OUTPUT)
    args = parser.parse_args()

    daily = _load_daily_pm10(args.input)
    predictions = _generate_predictions(daily, args.dataset, args.origin_step, args.n_jobs)
    skill = _build_skill_summary(predictions)
    _write_desirable_cells(predictions, skill)

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.skill_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    skill.to_csv(args.skill_output, index=False)
    print(f"Wrote {args.predictions_output} with {len(predictions)} rows")
    print(f"Wrote {args.skill_output} with {len(skill)} rows")


if __name__ == "__main__":
    main()
