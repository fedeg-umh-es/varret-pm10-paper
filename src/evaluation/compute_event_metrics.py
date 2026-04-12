"""
Compute event metrics: exceedance recall
========================================
Evaluates model ability to predict high PM10 events.

THRESHOLD POLICY (Paper A official):
  Rolling-origin: P75/P90 recomputed on each fold's training window.
  Holdout: P75/P90 computed on holdout training split.

  This prevents inflation of recall/precision caused by non-stationarity
  between fold-0 train and later test periods.

Outputs per (protocol, model, horizon, threshold):
  base_rate_test, recall_events, precision_events, flag_rate,
  skill, var_pct, skill_vp
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

def compute_event_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          threshold: float) -> dict:
    """Recall, precision, F1, flag_rate for exceedance events."""
    event_true = y_true > threshold
    event_pred = y_pred > threshold

    tp = np.sum(event_true & event_pred)
    fn = np.sum(event_true & ~event_pred)
    fp = np.sum(~event_true & event_pred)

    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    flag_rate = float(np.mean(event_pred))
    base_rate = float(np.mean(event_true))

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'flag_rate': flag_rate,
        'base_rate_test': base_rate,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pred(pred_dir: Path, name: str) -> pd.DataFrame | None:
    path = pred_dir / name
    return pd.read_parquet(path) if path.exists() else None


def _load_models(pred_dir: Path, suffix: str) -> dict:
    candidate_models = [
        ('lgbm',   f"lgbm_{suffix}.parquet"),
        ('nn',     f"nn_{suffix}.parquet"),
        ('lstm',   f"lstm_{suffix}.parquet"),
        ('sarima', f"sarima_{suffix}.parquet"),
    ]
    loaded = {}
    for label, fname in candidate_models:
        df = _load_pred(pred_dir, fname)
        if df is not None:
            loaded[label] = df
            print(f"  Loaded {fname}")
        else:
            print(f"  Skipping {fname} (not found)")
    return loaded


# ---------------------------------------------------------------------------
# Rolling-origin: per-fold thresholds, averaged across folds
# ---------------------------------------------------------------------------

def _compute_rolling_origin(
    pred_dir: Path,
    observations: np.ndarray,
    splits: dict,
    horizons: list,
    percentiles: list,
) -> list:
    pred_persist = _load_pred(pred_dir, "persistence_rolling_origin.parquet")
    if pred_persist is None:
        raise FileNotFoundError("persistence_rolling_origin.parquet not found")

    loaded_models = _load_models(pred_dir, "rolling_origin")

    # Accumulate per-fold metrics then average
    # Structure: {(model, h, p): [metric_dicts]}
    fold_metrics: dict = {}

    for fold_info in splits['folds']:
        fold      = fold_info['fold']
        train_idx = fold_info['train_idx']
        test_idx  = fold_info['test_idx']

        # Fold-specific thresholds from this fold's training window
        obs_train = observations[train_idx]
        fold_thresholds = {p: float(np.percentile(obs_train, p)) for p in percentiles}

        y_true = observations[test_idx]

        # Persistence for this fold
        pers_fold = pred_persist[pred_persist['fold'] == fold]

        for h in horizons:
            pers_h   = pers_fold[pers_fold['horizon'] == h]
            y_persist = pers_h['y_pred'].values
            indices_h = pers_h['sample_idx'].values

            # y_true aligned to persistence indices
            y_true_h = observations[indices_h]
            rmse_persist = np.sqrt(np.mean((y_persist - y_true_h) ** 2))

            for model_label, df_pred in loaded_models.items():
                df_fold_h = df_pred[
                    (df_pred['fold'] == fold) & (df_pred['horizon'] == h)
                ] if 'fold' in df_pred.columns else df_pred[df_pred['horizon'] == h]

                y_pred = df_fold_h['y_pred'].values

                if len(y_pred) != len(y_true_h):
                    print(f"  WARNING: fold {fold} {model_label} h={h}: "
                          f"length mismatch ({len(y_pred)} vs {len(y_true_h)}), skipping")
                    continue

                rmse_model = np.sqrt(np.mean((y_pred - y_true_h) ** 2))
                skill      = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0.0
                var_pct    = (100 * np.var(y_pred) / np.var(y_true_h)
                              if np.var(y_true_h) > 0 else 0.0)
                skill_vp   = skill * (var_pct / 100)

                for p in percentiles:
                    thr = fold_thresholds[p]
                    em  = compute_event_metrics(y_true_h, y_pred, thr)

                    key = (model_label, h, p)
                    if key not in fold_metrics:
                        fold_metrics[key] = []
                    fold_metrics[key].append({
                        'recall':          em['recall'],
                        'precision':       em['precision'],
                        'f1':              em['f1'],
                        'flag_rate':       em['flag_rate'],
                        'base_rate_test':  em['base_rate_test'],
                        'skill':           skill,
                        'var_pct':         var_pct,
                        'skill_vp':        skill_vp,
                        'threshold_value': thr,
                    })

    # Average across folds
    rows = []
    for (model_label, h, p), fold_list in fold_metrics.items():
        avg = {k: float(np.mean([d[k] for d in fold_list])) for k in fold_list[0]}
        rows.append({
            'protocol':          'rolling_origin',
            'model':             model_label,
            'h':                 h,
            'threshold':         p,
            'threshold_value':   avg['threshold_value'],
            'base_rate_test':    avg['base_rate_test'],
            'recall_events':     avg['recall'],
            'precision_events':  avg['precision'],
            'flag_rate':         avg['flag_rate'],
            'skill':             avg['skill'],
            'var_pct':           avg['var_pct'],
            'skill_vp':          avg['skill_vp'],
        })

    return rows


# ---------------------------------------------------------------------------
# Holdout: single threshold from holdout train split
# ---------------------------------------------------------------------------

def _compute_holdout(
    pred_dir: Path,
    observations: np.ndarray,
    splits: dict,
    horizons: list,
    percentiles: list,
) -> list:
    train_idx = splits['train_idx']
    test_idx  = splits['test_idx']

    obs_train      = observations[train_idx]
    ho_thresholds  = {p: float(np.percentile(obs_train, p)) for p in percentiles}
    print(f"  Holdout thresholds: {ho_thresholds}")

    pred_persist = _load_pred(pred_dir, "persistence_holdout.parquet")
    if pred_persist is None:
        raise FileNotFoundError("persistence_holdout.parquet not found")

    loaded_models = _load_models(pred_dir, "holdout")

    rows = []
    for h in horizons:
        pers_h    = pred_persist[pred_persist['horizon'] == h]
        y_persist = pers_h['y_pred'].values
        indices_h = pers_h['sample_idx'].values
        y_true_h  = observations[indices_h]
        rmse_persist = np.sqrt(np.mean((y_persist - y_true_h) ** 2))

        for model_label, df_pred in loaded_models.items():
            df_h   = df_pred[df_pred['horizon'] == h]
            y_pred = df_h['y_pred'].values

            if len(y_pred) != len(y_true_h):
                print(f"  WARNING: {model_label} h={h}: "
                      f"length mismatch ({len(y_pred)} vs {len(y_true_h)}), skipping")
                continue

            rmse_model = np.sqrt(np.mean((y_pred - y_true_h) ** 2))
            skill      = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0.0
            var_pct    = (100 * np.var(y_pred) / np.var(y_true_h)
                          if np.var(y_true_h) > 0 else 0.0)
            skill_vp   = skill * (var_pct / 100)

            for p in percentiles:
                thr = ho_thresholds[p]
                em  = compute_event_metrics(y_true_h, y_pred, thr)

                rows.append({
                    'protocol':          'holdout',
                    'model':             model_label,
                    'h':                 h,
                    'threshold':         p,
                    'threshold_value':   thr,
                    'base_rate_test':    em['base_rate_test'],
                    'recall_events':     em['recall'],
                    'precision_events':  em['precision'],
                    'flag_rate':         em['flag_rate'],
                    'skill':             skill,
                    'var_pct':           var_pct,
                    'skill_vp':          skill_vp,
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute event metrics")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open("config/horizons.yaml") as f:
        horizons = yaml.safe_load(f)['horizons']
    with open("config/thresholds.yaml") as f:
        percentiles = yaml.safe_load(f)['event_detection']['percentiles']

    processed_dir = Path(cfg['paths']['processed_dir'])
    pred_dir      = Path(cfg['paths']['predictions_dir'])
    metrics_dir   = Path(cfg['paths']['metrics_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Loading observations...")
    observations = pd.read_parquet(
        processed_dir / "pm10_preprocessed.parquet"
    )['pm10_normalized'].values

    with open(processed_dir / "splits_rolling_origin.json") as f:
        splits_ro = json.load(f)
    with open(processed_dir / "splits_holdout.json") as f:
        splits_ho = json.load(f)

    all_rows = []

    # Rolling-origin (fold-specific thresholds)
    print("\nComputing event metrics (rolling_origin) — fold-specific thresholds...")
    rows_ro = _compute_rolling_origin(
        pred_dir, observations, splits_ro, horizons, percentiles
    )
    all_rows.extend(rows_ro)
    df_ro = pd.DataFrame(rows_ro)
    ro_path = metrics_dir / "metrics_events_rolling_origin.csv"
    df_ro.to_csv(ro_path, index=False)
    print(f"Saved → {ro_path} ({len(df_ro)} rows)")

    # Holdout (threshold from holdout train)
    print("\nComputing event metrics (holdout) — holdout-train threshold...")
    rows_ho = _compute_holdout(
        pred_dir, observations, splits_ho, horizons, percentiles
    )
    all_rows.extend(rows_ho)
    df_ho = pd.DataFrame(rows_ho)
    ho_path = metrics_dir / "metrics_events_holdout.csv"
    df_ho.to_csv(ho_path, index=False)
    print(f"Saved → {ho_path} ({len(df_ho)} rows)")

    # Combined
    df_full = pd.DataFrame(all_rows)
    full_path = metrics_dir / "metrics_events_full.csv"
    df_full.to_csv(full_path, index=False)
    print(f"\nSaved combined → {full_path} ({len(df_full)} rows)")

    # --- Diagnostic pivots ---
    ro75 = df_full[
        (df_full['protocol'] == 'rolling_origin') & (df_full['threshold'] == 75)
    ]

    for metric, label in [
        ('base_rate_test',   'Base rate test @ P75  (rolling_origin, avg across folds)'),
        ('recall_events',    'Recall @ P75           (rolling_origin)'),
        ('precision_events', 'Precision @ P75        (rolling_origin)'),
        ('flag_rate',        'Flag rate @ P75        (rolling_origin)'),
    ]:
        print(f"\n{label}:")
        if metric == 'base_rate_test':
            # model-independent — show for first model only
            first_model = ro75['model'].iloc[0]
            tbl = (
                ro75[ro75['model'] == first_model]
                [['h', metric]]
                .set_index('h')
                .round(3)
            )
        else:
            tbl = (
                ro75.pivot_table(index='h', columns='model', values=metric)
                .round(3)
            )
        print(tbl)


if __name__ == "__main__":
    main()
