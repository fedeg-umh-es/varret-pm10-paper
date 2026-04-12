"""
Build canonical table
=====================
Final results table for Paper A.

Rows: horizon (h = 1, 6, 24, 48)
Columns per model: skill, var_pct, skill_vp, recall_p75, precision_p75, flag_rate_p75
Plus shared: rmse_persistence, var_obs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description="Build canonical table")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path(cfg['paths']['metrics_dir'])
    tables_dir  = Path(cfg['paths']['tables_dir'])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    df_ro     = pd.read_csv(metrics_dir / "metrics_rolling_origin_by_horizon.csv")
    df_events = pd.read_csv(metrics_dir / "metrics_events_rolling_origin.csv")

    horizons = sorted(df_ro['h'].unique())
    models   = sorted(df_ro['model'].unique())

    decimals = cfg['evaluation']['round_decimals']

    rows = []
    for h in horizons:
        row = {'h': h}

        # Shared: persistence RMSE and obs variance (model-independent)
        ref = df_ro[(df_ro['h'] == h) & (df_ro['model'] == models[0])]
        if len(ref) > 0:
            row['rmse_persistence'] = round(float(ref['rmse_persistence'].values[0]), decimals['rmse'])
            row['var_obs']          = round(float(ref['var_obs'].values[0]),          decimals['rmse'])

        for model in models:
            md = df_ro[(df_ro['model'] == model) & (df_ro['h'] == h)]
            if len(md) == 0:
                continue

            pfx = f"{model}_"
            row[f'{pfx}rmse']     = round(float(md['rmse'].values[0]),     decimals['rmse'])
            row[f'{pfx}skill']    = round(float(md['skill'].values[0]),    decimals['skill'])
            row[f'{pfx}var_pct']  = round(float(md['var_pct'].values[0]),  decimals['var_pct'])
            row[f'{pfx}skill_vp'] = round(float(md['skill_vp'].values[0]), decimals['skill_vp'])

            # Event metrics at P75
            ev75 = df_events[
                (df_events['model'] == model) &
                (df_events['h'] == h) &
                (df_events['threshold'] == 75)
            ]
            if len(ev75) > 0:
                row[f'{pfx}recall_p75']    = round(float(ev75['recall_events'].values[0]),    decimals['recall'])
                row[f'{pfx}precision_p75'] = round(float(ev75['precision_events'].values[0]), decimals['recall'])
                row[f'{pfx}flag_rate_p75'] = round(float(ev75['flag_rate'].values[0]),        decimals['recall'])

            # Event metrics at P90
            ev90 = df_events[
                (df_events['model'] == model) &
                (df_events['h'] == h) &
                (df_events['threshold'] == 90)
            ]
            if len(ev90) > 0:
                row[f'{pfx}recall_p90']    = round(float(ev90['recall_events'].values[0]),    decimals['recall'])
                row[f'{pfx}precision_p90'] = round(float(ev90['precision_events'].values[0]), decimals['recall'])

        rows.append(row)

    df_canonical = pd.DataFrame(rows)

    canonical_path = tables_dir / "table_canonical_full.csv"
    df_canonical.to_csv(canonical_path, index=False)
    print(f"Saved → {canonical_path}")
    print("\nCanonical Table:")
    print(df_canonical.to_string())


if __name__ == "__main__":
    main()
