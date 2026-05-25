"""Generate minimal daily E1-RR lags-only predictions.

This script is intentionally restricted to the current E1-RR post-evaluation
work package. It consumes a daily PM10 table with `date` and `pm10`, then writes
canonical inputs for variance-retention diagnostics:

- outputs/metrics/predictions.csv
- outputs/metrics/skill_summary.csv

Design constraints:
- no meteorological inputs
- no E2-MET or E3-PROB logic
- rolling-origin evaluation
- horizon range h=1,...,7 days
- train-only fitting at each origin
- persistence as mandatory baseline
- lightweight sklearn models only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.seasonal_persistence import SeasonalPersistenceModel
from src.models.sarima_model import SarimaForecaster
from src.models.stl_ridge import STLRidgeForecaster


DEFAULT_INPUT = Path("data/raw/pm10_daily.csv")
DEFAULT_PREDICTIONS_OUTPUT = Path("outputs/metrics/predictions.csv")
DEFAULT_SKILL_OUTPUT = Path("outputs/metrics/skill_summary.csv")

HORIZONS = tuple(range(1, 8))
LAGS = tuple(range(0, 7))
MIN_TRAIN_ROWS = 365


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float(np.sqrt(np.mean(err**2)))


def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    err = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float(np.mean(np.abs(err)))


def _load_daily_pm10(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing PM10 input file: {path}")
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


def _make_supervised(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = daily.copy()
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame["pm10"].shift(lag)
    frame["target"] = frame["pm10"].shift(-horizon)
    frame["target_date"] = frame["date"].shift(-horizon)
    return frame


def _iter_origins(daily: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    observed = daily.dropna(subset=["pm10"])
    origins = observed[(observed["date"].dt.year >= start_year) & (observed["date"].dt.year <= end_year)]["date"]
    return origins.reset_index(drop=True)


def _fit_predict_models(
    train: pd.DataFrame,
    train_daily: pd.DataFrame,
    test_row: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> dict[str, float]:
    x_train = train[feature_cols]
    y_train = train["target"]
    x_test = test_row[feature_cols]

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    hgb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=42)

    ridge.fit(x_train, y_train)
    hgb.fit(x_train, y_train)

    sp = SeasonalPersistenceModel(season_length=7)
    history = train_daily["pm10"].dropna().to_numpy()
    sp_pred = float(sp.predict(history, horizon=horizon)[horizon - 1])

    stl_ridge = STLRidgeForecaster(season_length=7, ridge_alpha=1.0, n_lags=7)
    stl_ridge.fit(train_daily)
    stl_pred = float(stl_ridge.predict_horizon(horizon))

    return {
        "ridge_direct": float(ridge.predict(x_test)[0]),
        "hgb_direct": float(hgb.predict(x_test)[0]),
        "stl_ridge_direct": stl_pred,
        "seasonal_naive": sp_pred,
    }


def _fit_predict_sarima(train_daily: pd.DataFrame, horizon: int) -> float:
    y_train = train_daily["pm10"].dropna().to_numpy(dtype=float)
    model = SarimaForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    model.fit(y_train)
    forecast = model.forecast(steps=horizon)
    return float(forecast[horizon - 1])


def _generate_predictions_for_horizon(
    daily: pd.DataFrame,
    dataset: str,
    horizon: int,
    origins: pd.Series,
    origin_step: int,
    sarima_origin_step: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    feature_cols = [f"lag_{lag}" for lag in LAGS]
    supervised = _make_supervised(daily, horizon=horizon)
    complete = supervised.dropna(subset=feature_cols + ["target", "target_date"])
    selected_origins = origins.iloc[::origin_step].reset_index(drop=True)

    for origin_idx, origin in enumerate(selected_origins):
        test = complete[complete["date"] == origin]
        if test.empty:
            continue
        train = complete[complete["target_date"] <= origin].copy()
        if len(train) < MIN_TRAIN_ROWS:
            continue
        train_daily = daily[(daily["date"] <= origin) & daily["pm10"].notna()].copy()
        if len(train_daily) < MIN_TRAIN_ROWS:
            continue

        y_origin = float(test.iloc[0]["lag_0"])
        y_true = float(test.iloc[0]["target"])
        target_date = pd.Timestamp(test.iloc[0]["target_date"])

        rows.append(
            {
                "dataset": dataset,
                "model": "persistence",
                "fold": origin.strftime("%Y-%m-%d"),
                "origin_date": origin.strftime("%Y-%m-%d"),
                "horizon": horizon,
                "date": target_date.strftime("%Y-%m-%d"),
                "y_true": y_true,
                "y_pred": y_origin,
            }
        )

        preds = _fit_predict_models(train, train_daily, test, feature_cols, horizon=horizon)
        if sarima_origin_step > 0 and origin_idx % sarima_origin_step == 0:
            try:
                preds["sarima"] = _fit_predict_sarima(train_daily, horizon=horizon)
            except Exception as exc:
                log_path = Path("outputs/logs/sarima_failures.log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{dataset},h={horizon},origin={origin.date()},{type(exc).__name__}: {exc}\n")
        for model, y_pred in preds.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "fold": origin.strftime("%Y-%m-%d"),
                    "origin_date": origin.strftime("%Y-%m-%d"),
                    "horizon": horizon,
                    "date": target_date.strftime("%Y-%m-%d"),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    print(f"h={horizon}: wrote {len(rows)} prediction rows from {len(selected_origins)} candidate origins")
    return pd.DataFrame(rows)


def _generate_predictions(
    daily: pd.DataFrame,
    dataset: str,
    start_year: int,
    end_year: int,
    origin_step: int,
    sarima_origin_step: int,
    n_jobs: int,
) -> pd.DataFrame:
    if origin_step < 1:
        raise ValueError("origin_step must be >= 1")
    if sarima_origin_step < 0:
        raise ValueError("sarima_origin_step must be >= 0")

    origins = _iter_origins(daily, start_year=start_year, end_year=end_year)
    if origins.empty:
        raise ValueError("No observed origins found in the requested evaluation window.")

    horizon_frames = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_generate_predictions_for_horizon)(daily, dataset, horizon, origins, origin_step, sarima_origin_step)
        for horizon in HORIZONS
    )

    predictions = pd.concat(horizon_frames, ignore_index=True)
    if predictions.empty:
        raise ValueError("No predictions were generated. Check date range, missing data, and MIN_TRAIN_ROWS.")
    return predictions.sort_values(["dataset", "model", "horizon", "origin_date"]).reset_index(drop=True)


def _build_skill_summary(predictions: pd.DataFrame, baseline_model: str = "persistence") -> pd.DataFrame:
    baseline = predictions[predictions["model"] == baseline_model]
    if baseline.empty:
        raise ValueError(f"Baseline model not found: {baseline_model}")

    rows: list[dict] = []
    for (dataset, model, horizon), group in predictions.groupby(["dataset", "model", "horizon"]):
        if model == baseline_model:
            continue

        baseline_group = baseline[(baseline["dataset"] == dataset) & (baseline["horizon"] == horizon)]
        merged = group.merge(
            baseline_group[["origin_date", "date", "y_pred"]].rename(columns={"y_pred": "y_pred_baseline"}),
            on=["origin_date", "date"],
            how="inner",
        )
        if merged.empty:
            continue

        rmse_model = _rmse(merged["y_true"], merged["y_pred"])
        rmse_baseline = _rmse(merged["y_true"], merged["y_pred_baseline"])
        mae_model = _mae(merged["y_true"], merged["y_pred"])
        mae_baseline = _mae(merged["y_true"], merged["y_pred_baseline"])
        skill = 1.0 - rmse_model / rmse_baseline if rmse_baseline > 0 else np.nan
        mae_skill = 1.0 - mae_model / mae_baseline if mae_baseline > 0 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "skill": skill,
                "mae_skill": mae_skill,
            }
        )

    skill = pd.DataFrame(rows)
    if skill.empty:
        raise ValueError("No skill rows generated.")
    return skill.sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate minimal daily E1-RR lags-only predictions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV with date and pm10 columns")
    parser.add_argument("--dataset", default="e1_rr_daily", help="Dataset identifier")
    parser.add_argument("--start-year", type=int, default=2020, help="First origin year")
    parser.add_argument("--end-year", type=int, default=2024, help="Last origin year")
    parser.add_argument(
        "--origin-step",
        type=int,
        default=1,
        help="Evaluate every kth observed origin. Keep 1 for the full rolling-origin run.",
    )
    parser.add_argument(
        "--sarima-origin-step",
        type=int,
        default=14,
        help="Evaluate SARIMA every kth selected origin. Use 0 to disable SARIMA.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs across horizons")
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--skill-output", type=Path, default=DEFAULT_SKILL_OUTPUT)
    args = parser.parse_args()

    daily = _load_daily_pm10(args.input)
    predictions = _generate_predictions(
        daily,
        dataset=args.dataset,
        start_year=args.start_year,
        end_year=args.end_year,
        origin_step=args.origin_step,
        sarima_origin_step=args.sarima_origin_step,
        n_jobs=args.n_jobs,
    )
    skill = _build_skill_summary(predictions)

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.skill_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    skill.to_csv(args.skill_output, index=False)

    print(f"Wrote {args.predictions_output} with {len(predictions)} rows")
    print(f"Wrote {args.skill_output} with {len(skill)} rows")
    print("Date range:", predictions["date"].min(), "to", predictions["date"].max())
    print("Models:", sorted(predictions["model"].unique()))
    print("Horizons:", int(predictions["horizon"].min()), "to", int(predictions["horizon"].max()))
    print("Rows per model/horizon min:", int(predictions.groupby(["model", "horizon"]).size().min()))


if __name__ == "__main__":
    main()
