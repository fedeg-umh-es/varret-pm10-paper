"""KGE horizon-wise diagnostics for the PM10 variance-retention paper.

AUDIT SOURCE: adapted from manuscript_assets/paper_c_kge/paper.tex KGE
definitions and the existing repo outputs used by scripts/13 and scripts/14.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
KGE_OUTPUT_COLUMNS = [
    "station",
    "model",
    "h",
    "r_h",
    "alpha_h",
    "beta_h",
    "KGE_h",
    "KGE_skill_h",
]


def _finite_pair_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def compute_kge_components(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Compute horizon-wise KGE components.

    Formula:
        KGE_h = 1 - sqrt((r_h - 1)^2 + (alpha_h - 1)^2 + (beta_h - 1)^2)

    where r_h is Pearson correlation, alpha_h = SD(y_pred) / SD(y_true),
    and beta_h = mean(y_pred) / mean(y_true). alpha_h == phi_h by construction.

    Degenerate groups return NaN for undefined components: fewer than two
    finite pairs, zero observed standard deviation, zero observed mean, or
    zero predicted standard deviation for correlation.
    """
    true, pred = _finite_pair_arrays(y_true, y_pred)
    if len(true) < 2:
        return {"r_h": np.nan, "alpha_h": np.nan, "beta_h": np.nan, "KGE_h": np.nan}

    sd_true = float(np.std(true, ddof=0))
    sd_pred = float(np.std(pred, ddof=0))
    mean_true = float(np.mean(true))
    mean_pred = float(np.mean(pred))

    alpha = sd_pred / sd_true if sd_true > EPS else np.nan
    beta = mean_pred / mean_true if abs(mean_true) > EPS else np.nan
    r = float(np.corrcoef(pred, true)[0, 1]) if sd_true > EPS and sd_pred > EPS else np.nan

    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = np.nan

    return {"r_h": r, "alpha_h": alpha, "beta_h": beta, "KGE_h": kge}


def compute_kge_skill(kge_model: float, kge_persistence: float) -> float:
    """Compute KGE skill against persistence as 1 - KGE_h / KGE_persistence_h.

    If persistence KGE is zero, near-zero, or non-finite, the ratio is not
    stable and NaN is returned conservatively.
    """
    if not np.isfinite(kge_model) or not np.isfinite(kge_persistence):
        return float("nan")
    if abs(kge_persistence) <= EPS:
        return float("nan")
    return float(1.0 - (kge_model / kge_persistence))


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def kge_horizon_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Build station-model-horizon KGE diagnostics from aligned forecasts.

    Input columns must include dataset, model, horizon, y_true and y_pred.
    Persistence forecasts must be present as model == "persistence" so
    KGE_persistence_h and KGE_skill_h can be computed without rerunning models.
    """
    _require_columns(forecast_df, ["dataset", "model", "horizon", "y_true", "y_pred"], "forecast_df")

    persistence = forecast_df[forecast_df["model"].eq("persistence")]
    if persistence.empty:
        raise ValueError("forecast_df must include persistence rows for KGE_skill_h.")

    persistence_kge: dict[tuple[str, int], float] = {}
    for (dataset, horizon), group in persistence.groupby(["dataset", "horizon"], sort=True):
        components = compute_kge_components(group["y_true"], group["y_pred"])
        persistence_kge[(str(dataset), int(horizon))] = components["KGE_h"]

    rows: list[dict[str, object]] = []
    for (dataset, model, horizon), group in forecast_df.groupby(["dataset", "model", "horizon"], sort=True):
        if model == "persistence":
            continue
        components = compute_kge_components(group["y_true"], group["y_pred"])
        kge_persistence = persistence_kge.get((str(dataset), int(horizon)), np.nan)
        rows.append(
            {
                "dataset": dataset,
                "station": dataset,
                "model": model,
                "h": int(horizon),
                "horizon": int(horizon),
                "n": int(len(group.dropna(subset=["y_true", "y_pred"]))),
                "r_h": components["r_h"],
                "alpha_h": components["alpha_h"],
                "phi_h": components["alpha_h"],
                "beta_h": components["beta_h"],
                "KGE_h": components["KGE_h"],
                "KGE_persistence_h": kge_persistence,
                "KGE_skill_h": compute_kge_skill(components["KGE_h"], kge_persistence),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["dataset", *KGE_OUTPUT_COLUMNS])
    return out.sort_values(["station", "model", "h"]).reset_index(drop=True)

