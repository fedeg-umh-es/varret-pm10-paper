"""
Build canonical table
=====================
E1: Main results table with all metrics per horizon.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml


def main():
    """Build canonical table."""
    parser = argparse.ArgumentParser(description="Build canonical table")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    metrics_dir = Path(cfg['paths']['metrics_dir'])
    tables_dir = Path(cfg['paths']['tables_dir'])
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("Loading metrics...")
    df_ro = pd.read_csv(metrics_dir / "metrics_rolling_origin_by_horizon.csv")

    # Try to load event metrics (optional)
    events_path = metrics_dir / "metrics_events_rolling_origin.csv"
    df_events = None
    if events_path.exists():
        df_events = pd.read_csv(events_path)
    else:
        print("Note: Event metrics not found, skipping recall columns")
    
    # Build table: rows = horizon, columns = metrics per model
    print("\nBuilding canonical table...")
    
    # Get unique horizons
    horizons = sorted(df_ro['h'].unique())
    models = sorted(df_ro['model'].unique())
    
    rows = []
    for h in horizons:
        row = {'h': h}
        
        # Persistence baseline (RMSE only)
        persist_data = df_ro[(df_ro['model'] == models[0]) & (df_ro['h'] == h)]
        if len(persist_data) > 0:
            row['rmse_persistence'] = persist_data['rmse_persistence'].values[0]
            row['var_obs'] = persist_data['var_obs'].values[0]
        
        # For each model
        for model in models:
            model_data = df_ro[(df_ro['model'] == model) & (df_ro['h'] == h)]
            if len(model_data) > 0:
                prefix = f"{model}_"
                row[f'{prefix}rmse'] = model_data['rmse'].values[0]
                row[f'{prefix}skill'] = model_data['skill'].values[0]
                row[f'{prefix}var_pred'] = model_data['var_pred'].values[0]
                row[f'{prefix}var_pct'] = model_data['var_pct'].values[0]
                row[f'{prefix}skill_vp'] = model_data['skill_vp'].values[0]
                
                # Add recall from events (if available)
                if df_events is not None:
                    event_data = df_events[
                        (df_events['model'] == model) &
                        (df_events['h'] == h) &
                        (df_events['threshold'] == 75)
                    ]
                    if len(event_data) > 0:
                        row[f'{prefix}recall_p75'] = event_data['recall_events'].values[0]

                    event_data = df_events[
                        (df_events['model'] == model) &
                        (df_events['h'] == h) &
                        (df_events['threshold'] == 90)
                    ]
                    if len(event_data) > 0:
                        row[f'{prefix}recall_p90'] = event_data['recall_events'].values[0]
        
        rows.append(row)
    
    df_canonical = pd.DataFrame(rows)
    
    # Round for readability
    decimals = cfg['evaluation']['round_decimals']
    for col in df_canonical.columns:
        if col != 'h':
            dtype = df_canonical[col].dtype
            if dtype in ['float64', 'float32']:
                if 'rmse' in col or 'var' in col:
                    df_canonical[col] = df_canonical[col].round(decimals['rmse'])
                elif 'skill_vp' in col:
                    df_canonical[col] = df_canonical[col].round(decimals['skill_vp'])
                elif 'skill' in col:
                    df_canonical[col] = df_canonical[col].round(decimals['skill'])
                elif 'recall' in col:
                    df_canonical[col] = df_canonical[col].round(decimals['recall'])
    
    # Save
    canonical_path = tables_dir / "table_canonical_full.csv"
    df_canonical.to_csv(canonical_path, index=False)
    print(f"Saved to {canonical_path}")
    print("\nCanonical Table:")
    print(df_canonical.to_string())


if __name__ == "__main__":
    main()
