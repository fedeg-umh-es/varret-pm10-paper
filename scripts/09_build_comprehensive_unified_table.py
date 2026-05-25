#!/usr/bin/env python3
"""Merge diagnostic outputs into a master station-model-horizon table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


VARRET = Path("outputs/tables/variance_retention_all_stations.csv")
DM = Path("outputs/tables/dm_significance_all_stations.csv")
EXCEEDANCE = Path("outputs/tables/exceedance_all_stations.csv")
MURPHY = Path("outputs/tables/murphy_decomposition_all_stations.csv")
CONCENTRATION = Path("outputs/tables/concentration_scale_summary.csv")
OUTPUT = Path("outputs/tables/master_diagnostic_table.csv")


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    if not VARRET.exists():
        raise FileNotFoundError(f"Missing base variance-retention table: {VARRET}")
    master = pd.read_csv(VARRET)

    dm = _read_optional(DM)
    if not dm.empty:
        cols = ["dataset", "model", "horizon", "dm_pval_bh", "dm_significant", "dm_stat", "n_pairs"]
        master = master.merge(dm[[c for c in cols if c in dm.columns]], on=["dataset", "model", "horizon"], how="left")

    murphy = _read_optional(MURPHY)
    if not murphy.empty:
        cols = ["dataset", "model", "horizon", "bias_sq", "cond_bias_sq", "irreducible_sq", "rho", "std_pred_ug", "std_true_ug"]
        master = master.merge(
            murphy[[c for c in cols if c in murphy.columns]],
            on=["dataset", "model", "horizon"],
            how="left",
            suffixes=("", "_murphy"),
        )

    exceedance = _read_optional(EXCEEDANCE)
    if not exceedance.empty:
        pivot = exceedance.pivot_table(
            index=["dataset", "model", "horizon"],
            columns="threshold_type",
            values=["recall", "precision", "f1", "flag_rate", "base_rate"],
            aggfunc="mean",
        )
        pivot.columns = [f"{metric}_{threshold}" for metric, threshold in pivot.columns]
        master = master.merge(pivot.reset_index(), on=["dataset", "model", "horizon"], how="left")

    concentration = _read_optional(CONCENTRATION)
    if not concentration.empty:
        keep = [c for c in concentration.columns if c in {"dataset", "obs_mean_ug", "obs_std_ug", "observed_range_ug"}]
        master = master.merge(concentration[keep].drop_duplicates("dataset"), on="dataset", how="left")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(master)} rows and {len(master.columns)} columns")


if __name__ == "__main__":
    main()
