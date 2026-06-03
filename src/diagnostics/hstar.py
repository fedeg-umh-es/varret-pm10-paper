"""Operational forecast skill horizon utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def compute_hstar(skill_by_horizon: Sequence[float], criterion: str = "strict") -> int:
    """Compute H* from a horizon-wise skill sequence.

    Parameters
    ----------
    skill_by_horizon:
        Ordered horizon-wise skill values starting at h=1.
    criterion:
        `strict` returns the last consecutive horizon from h=1 with positive skill.
        `relax` returns the last horizon anywhere with positive skill.
        `nonnegative` uses skill >= 0 instead of skill > 0.
    """
    skill = np.asarray(skill_by_horizon, dtype=float)
    if skill.ndim != 1 or skill.size == 0:
        raise ValueError("skill_by_horizon must be a non-empty 1D sequence.")

    if criterion == "strict":
        hstar = 0
        for value in skill:
            if value > 0.0:
                hstar += 1
            else:
                break
        return hstar

    if criterion == "relax":
        positive = np.where(skill > 0.0)[0]
        return 0 if positive.size == 0 else int(positive[-1] + 1)

    if criterion == "nonnegative":
        valid = np.where(skill >= 0.0)[0]
        return 0 if valid.size == 0 else int(valid[-1] + 1)

    raise ValueError("criterion must be one of: 'strict', 'relax', 'nonnegative'.")


def build_skill_table(
    metrics_df: pd.DataFrame,
    baseline_model: str = "persistence",
) -> pd.DataFrame:
    required = {"dataset", "horizon", "model", "rmse"}
    missing = required - set(metrics_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    baseline_df = (
        metrics_df.loc[metrics_df["model"] == baseline_model, ["dataset", "horizon", "rmse"]]
        .rename(columns={"rmse": "rmse_baseline"})
        .copy()
    )

    if (baseline_df["rmse_baseline"] <= 0).any():
        raise ValueError("Baseline RMSE must be strictly positive to compute skill.")

    skill_df = metrics_df.merge(
        baseline_df,
        on=["dataset", "horizon"],
        how="left",
        validate="many_to_one",
    )

    if skill_df["rmse_baseline"].isna().any():
        raise ValueError("Missing baseline RMSE for at least one dataset/horizon pair.")

    skill_df["baseline_model"] = baseline_model
    skill_df["skill_vs_baseline"] = 1.0 - (
        skill_df["rmse"] / skill_df["rmse_baseline"]
    )
    skill_df.loc[skill_df["model"] == baseline_model, "skill_vs_baseline"] = 0.0

    return (
        skill_df[
            [
                "dataset",
                "horizon",
                "model",
                "rmse",
                "baseline_model",
                "rmse_baseline",
                "skill_vs_baseline",
            ]
        ]
        .sort_values(["dataset", "horizon", "model"])
        .reset_index(drop=True)
    )


def build_horizons_summary(skill_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "horizon", "model", "skill_vs_baseline", "baseline_model"}
    missing = required - set(skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for (dataset, model), group in skill_df.groupby(["dataset", "model"], sort=True):
        group = group.sort_values("horizon").reset_index(drop=True)

        baseline_model = str(group["baseline_model"].iloc[0])

        if model == baseline_model:
            h = 0
            h_star_relax = 0
            h_star_strict = 0
        else:
            skill_values = group["skill_vs_baseline"].to_list()
            h = compute_hstar(skill_values, criterion="relax")
            h_star_relax = h
            h_star_strict = compute_hstar(skill_values, criterion="strict")

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "H": int(h),
                "H_star_relax": int(h_star_relax),
                "H_star_strict": int(h_star_strict),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )

def _build_fold_skill_table(
    fold_metrics_df: pd.DataFrame,
    baseline_model: str,
) -> pd.DataFrame:
    required = {"dataset", "fold", "horizon", "model", "rmse"}
    missing = required - set(fold_metrics_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    baseline_df = (
        fold_metrics_df.loc[
            fold_metrics_df["model"] == baseline_model,
            ["dataset", "fold", "horizon", "rmse"],
        ]
        .rename(columns={"rmse": "rmse_baseline"})
        .copy()
    )

    skill_df = fold_metrics_df.merge(
        baseline_df,
        on=["dataset", "fold", "horizon"],
        how="left",
        validate="many_to_one",
    )

    if skill_df["rmse_baseline"].isna().any():
        raise ValueError(
            "Missing baseline RMSE for at least one dataset/fold/horizon combination."
        )

    skill_df["baseline_model"] = baseline_model
    skill_df["undefined_skill_flag"] = skill_df["rmse_baseline"] <= 0

    skill_df["skill_vs_baseline"] = np.where(
        skill_df["rmse_baseline"] > 0,
        1.0 - (skill_df["rmse"] / skill_df["rmse_baseline"]),
        np.nan,
    )
    skill_df.loc[
        (skill_df["model"] == baseline_model) & (~skill_df["undefined_skill_flag"]),
        "skill_vs_baseline"
    ] = 0.0

    return (
        skill_df[
            [
                "dataset",
                "fold",
                "horizon",
                "model",
                "rmse",
                "baseline_model",
                "rmse_baseline",
                "skill_vs_baseline",
                "undefined_skill_flag",
            ]
        ]
        .sort_values(["dataset", "horizon", "model", "fold"])
        .reset_index(drop=True)
    )


def build_fold_skill_table(
    fold_metrics_df: pd.DataFrame,
    baseline_model: str = "persistence",
) -> pd.DataFrame:
    return _build_fold_skill_table(fold_metrics_df, baseline_model=baseline_model)


def build_fold_dispersion_summary(fold_skill_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "fold", "horizon", "model", "skill_vs_baseline"}
    missing = required - set(fold_skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for (dataset, model, horizon), group in fold_skill_df.groupby(
        ["dataset", "model", "horizon"], sort=True
    ):
        s = group["skill_vs_baseline"].dropna().astype(float)
        n_undefined = int(group["skill_vs_baseline"].isna().sum())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "mean_skill": float(s.mean()) if not s.empty else float("nan"),
                "std_skill": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
                "min_skill": float(s.min()) if not s.empty else float("nan"),
                "q25_skill": float(s.quantile(0.25)) if not s.empty else float("nan"),
                "median_skill": float(s.median()) if not s.empty else float("nan"),
                "q75_skill": float(s.quantile(0.75)) if not s.empty else float("nan"),
                "max_skill": float(s.max()) if not s.empty else float("nan"),
                "n_folds": int(len(group)),
                "n_valid_folds": int(len(s)),
                "n_undefined_skill_folds": n_undefined,
                "share_positive_skill": float((s > 0).mean()) if not s.empty else float("nan"),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model", "horizon"])
        .reset_index(drop=True)
    )


def build_trajectory_summary(skill_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "horizon", "model", "skill_vs_baseline"}
    missing = required - set(skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for (dataset, model), group in skill_df.groupby(["dataset", "model"], sort=True):
        group = group.sort_values("horizon").reset_index(drop=True)
        skill = group["skill_vs_baseline"].astype(float)
        horizons = group["horizon"].astype(int)

        peak_idx = int(skill.idxmax())
        last_skill = float(skill.iloc[-1])
        max_skill = float(skill.max())

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "max_skill": max_skill,
                "argmax_skill": int(group.loc[peak_idx, "horizon"]),
                "skill_range": float(max_skill - float(skill.min())),
                "skill_drop_last_vs_peak": float(max_skill - last_skill),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )


def _last_h_where(horizons: list[int], mask: list[bool]) -> int:
    """Return the last horizon (relax semantics) where mask is True, or 0."""
    candidates = [h for h, m in zip(horizons, mask) if m]
    return candidates[-1] if candidates else 0


def build_robust_horizons_summary(fold_dispersion_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "model",
        "horizon",
        "median_skill",
        "share_positive_skill",
        "q25_skill",
    }
    missing = required - set(fold_dispersion_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for (dataset, model), group in fold_dispersion_df.groupby(
        ["dataset", "model"], sort=True
    ):
        group = group.sort_values("horizon").reset_index(drop=True)
        horizons = group["horizon"].tolist()
        median = group["median_skill"].tolist()
        share = group["share_positive_skill"].tolist()
        q25 = group["q25_skill"].tolist()

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "last_h_positive_median": _last_h_where(horizons, [v > 0 for v in median]),
                "last_h_share_ge_0_50": _last_h_where(horizons, [v >= 0.50 for v in share]),
                "last_h_share_ge_0_75": _last_h_where(horizons, [v >= 0.75 for v in share]),
                "last_h_q25_nonneg": _last_h_where(horizons, [v >= 0.0 for v in q25]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
