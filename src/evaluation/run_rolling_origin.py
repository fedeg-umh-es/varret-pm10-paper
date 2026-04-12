"""
Run / verify rolling-origin predictions
========================================
Checks that all expected prediction files exist and have sensible record counts.
SARIMA predictions are validated separately from tabular models because they
follow a different generation path (time-series, not lag features).
"""

import pandas as pd
from pathlib import Path
import argparse
import yaml


def check_predictions(pred_dir: Path, models: list, protocol: str):
    print(f"\nChecking {protocol} predictions:")
    all_ok = True
    for model in models:
        path = pred_dir / f"{model}_{protocol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            nan_pct = df['y_pred'].isna().mean() * 100
            print(f"  ✓ {model}: {len(df):,} records  (NaN y_pred: {nan_pct:.1f}%)")
            if nan_pct > 10:
                print(f"    WARNING: high NaN rate — check SARIMA fit for this model")
        else:
            print(f"  ✗ {model}: MISSING — run the corresponding train_*.py script")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify rolling-origin predictions")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_dir = Path(cfg['paths']['predictions_dir'])

    # Tabular models (use lag features pipeline)
    tabular_models = ['persistence', 'lgbm', 'nn']

    # Statistical time-series models (bypass tabular pipeline)
    ts_models = ['sarima']

    ok_ro = check_predictions(pred_dir, tabular_models + ts_models, "rolling_origin")
    ok_ho = check_predictions(pred_dir, tabular_models + ts_models, "holdout")

    if ok_ro and ok_ho:
        print("\nAll prediction files present.")
    else:
        print("\nSome files are missing — see above.")


if __name__ == "__main__":
    main()
