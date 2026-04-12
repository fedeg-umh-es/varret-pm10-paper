"""
Compare rolling-origin vs holdout
==================================
E2: Protocol robustness. Checks if rolling-origin and holdout agree.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml


def main():
    """Compare protocols."""
    parser = argparse.ArgumentParser(description="Compare protocols")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    metrics_dir = Path(cfg['paths']['metrics_dir'])
    
    # Load metrics
    print("Loading metrics...")
    df_ro = pd.read_csv(metrics_dir / "metrics_rolling_origin_by_horizon.csv")
    df_ho = pd.read_csv(metrics_dir / "metrics_holdout_by_horizon.csv")
    
    # Compare
    print("\nComparing protocols...")
    comparison = []
    
    for model in df_ro['model'].unique():
        for h in df_ro['h'].unique():
            # Rolling-origin
            ro_data = df_ro[(df_ro['model'] == model) & (df_ro['h'] == h)]
            if len(ro_data) == 0:
                continue
            
            # Holdout
            ho_data = df_ho[(df_ho['model'] == model) & (df_ho['h'] == h)]
            if len(ho_data) == 0:
                continue
            
            for metric in ['rmse', 'skill', 'var_pct', 'skill_vp']:
                ro_val = ro_data[metric].values[0] if metric in ro_data.columns else None
                ho_val = ho_data[metric].values[0] if metric in ho_data.columns else None
                
                if ro_val is not None and ho_val is not None:
                    diff_abs = abs(ho_val - ro_val)
                    diff_pct = 100 * diff_abs / abs(ro_val) if ro_val != 0 else 0
                    
                    comparison.append({
                        'protocol': 'rolling_origin_vs_holdout',
                        'model': model,
                        'h': h,
                        'metric': metric,
                        'ro_value': ro_val,
                        'ho_value': ho_val,
                        'diff_absolute': diff_abs,
                        'diff_pct': diff_pct
                    })
    
    df_comparison = pd.DataFrame(comparison)
    
    # Save
    comp_path = metrics_dir / "metrics_protocol_comparison.csv"
    df_comparison.to_csv(comp_path, index=False)
    print(f"Saved to {comp_path}")
    
    # Summary
    print("\nSummary of differences (%):")
    print(df_comparison.groupby('metric')['diff_pct'].describe())


if __name__ == "__main__":
    main()
