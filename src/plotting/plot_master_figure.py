"""
Plot master figures — Paper A final version
============================================
Figure 1: Skill & Variance (Panel A + Panel B)
Figure 2: Skill_VP & Event metrics (Panel C + Panel D)

Models shown: LightGBM, SARIMA
Protocol: rolling-origin (fold-recalibrated thresholds for event panels)

Output files:
  outputs/figures/figure1_skill_variance.png
  outputs/figures/figure2_skillvp_events.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(df, model, col):
    """Return (h_values, col_values) sorted by horizon."""
    d = df[df['model'] == model].sort_values('h')
    return d['h'].values, d[col].values


def _base_fig(cfg):
    """Create a 1×2 figure with shared style."""
    FIGSIZE   = (7, 3.2)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=cfg['plotting']['dpi'])
    return fig, axes


FS = dict(title=9, label=9, tick=8, legend=8)
MODELS  = ['lgbm', 'sarima']
LABELS  = {'lgbm': 'LightGBM', 'sarima': 'SARIMA'}
COLORS  = {'lgbm': '#1f77b4', 'sarima': '#d62728'}
MARKERS = {'lgbm': 'o', 'sarima': 's'}
LW      = 1.8
MS      = 5


# ---------------------------------------------------------------------------
# Figure 1 — Skill & Variance
# ---------------------------------------------------------------------------

def plot_figure1(df_ro, horizons, cfg, figures_dir):
    fig, axes = _base_fig(cfg)
    fig.suptitle(
        'Skill and Variance Capture vs Forecast Horizon',
        fontsize=FS['title'], fontweight='bold'
    )

    # Panel A — Skill
    ax = axes[0]
    for m in MODELS:
        h_vals, y = _get(df_ro, m, 'skill')
        ax.plot(h_vals, y, marker=MARKERS[m], label=LABELS[m],
                color=COLORS[m], linewidth=LW, markersize=MS)
    ax.axhline(0, color='#888888', linestyle='--', linewidth=1.1, label='Skill = 0')
    ax.set_xlabel('Horizon (h)', fontsize=FS['label'])
    ax.set_ylabel('Skill', fontsize=FS['label'])
    ax.set_title('(A) Skill vs Horizon', fontsize=FS['title'], fontweight='bold')
    ax.set_xticks(horizons)
    ax.tick_params(labelsize=FS['tick'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS['legend'])
    ax.set_ylim([-0.05, 1.05])

    # Panel B — Variance %
    ax = axes[1]
    for m in MODELS:
        h_vals, y = _get(df_ro, m, 'var_pct')
        ax.plot(h_vals, y, marker=MARKERS[m], label=LABELS[m],
                color=COLORS[m], linewidth=LW, markersize=MS)
    ax.axhline(100, color='#2ca02c', linestyle='--', linewidth=1.1, label='100% (full variance)')
    ax.set_xlabel('Horizon (h)', fontsize=FS['label'])
    ax.set_ylabel('Variance captured (%)', fontsize=FS['label'])
    ax.set_title('(B) Variance % vs Horizon', fontsize=FS['title'], fontweight='bold')
    ax.set_xticks(horizons)
    ax.tick_params(labelsize=FS['tick'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS['legend'])
    ax.set_ylim([-5, 115])

    plt.tight_layout()
    path = figures_dir / "figure1_skill_variance.png"
    plt.savefig(path, dpi=cfg['plotting']['dpi'], bbox_inches='tight')
    print(f"Saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Skill_VP & Event metrics
# ---------------------------------------------------------------------------

def plot_figure2(df_ro, ev75, horizons, cfg, figures_dir):
    fig, axes = _base_fig(cfg)
    fig.suptitle(
        'Skill$_{VP}$ and Exceedance Detection vs Forecast Horizon',
        fontsize=FS['title'], fontweight='bold'
    )

    # Panel C — Skill_VP
    ax = axes[0]
    for m in MODELS:
        h_vals, y = _get(df_ro, m, 'skill_vp')
        ax.plot(h_vals, y, marker=MARKERS[m], label=LABELS[m],
                color=COLORS[m], linewidth=LW, markersize=MS)
    ax.axhline(0, color='#888888', linestyle='--', linewidth=1.1)
    ax.set_xlabel('Horizon (h)', fontsize=FS['label'])
    ax.set_ylabel('Skill$_{VP}$', fontsize=FS['label'])
    ax.set_title('(C) Skill$_{VP}$ vs Horizon', fontsize=FS['title'], fontweight='bold')
    ax.set_xticks(horizons)
    ax.tick_params(labelsize=FS['tick'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS['legend'])

    # Panel D — Recall & Precision @ P75
    ax = axes[1]
    for m in MODELS:
        df_m = ev75[ev75['model'] == m].sort_values('h')
        h_vals = df_m['h'].values
        rec    = df_m['recall_events'].values
        prec   = df_m['precision_events'].values

        ax.plot(h_vals, rec, marker=MARKERS[m], label=f'{LABELS[m]} recall',
                color=COLORS[m], linewidth=LW, markersize=MS)
        ax.plot(h_vals, prec, marker=MARKERS[m], linestyle='--',
                label=f'{LABELS[m]} precision',
                color=COLORS[m], linewidth=1.2, markersize=4, alpha=0.7)

    ax.axhline(0.5, color='#888888', linestyle=':', linewidth=0.9)
    ax.set_xlabel('Horizon (h)', fontsize=FS['label'])
    ax.set_ylabel('Recall / Precision @ P75', fontsize=FS['label'])
    ax.set_title('(D) Event Recall & Precision vs Horizon', fontsize=FS['title'], fontweight='bold')
    ax.set_xticks(horizons)
    ax.tick_params(labelsize=FS['tick'])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS['legend'])

    plt.tight_layout()
    path = figures_dir / "figure2_skillvp_events.png"
    plt.savefig(path, dpi=cfg['plotting']['dpi'], bbox_inches='tight')
    print(f"Saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Paper A figures")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open("config/horizons.yaml") as f:
        horizons = sorted(yaml.safe_load(f)['horizons'])

    metrics_dir = Path(cfg['paths']['metrics_dir'])
    figures_dir = Path(cfg['paths']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    df_ro     = pd.read_csv(metrics_dir / "metrics_rolling_origin_by_horizon.csv")
    df_events = pd.read_csv(metrics_dir / "metrics_events_rolling_origin.csv")

    df_ro     = df_ro[df_ro['model'].isin(MODELS)]
    df_events = df_events[df_events['model'].isin(MODELS)]
    ev75      = df_events[df_events['threshold'] == 75]

    plot_figure1(df_ro, horizons, cfg, figures_dir)
    plot_figure2(df_ro, ev75, horizons, cfg, figures_dir)


if __name__ == "__main__":
    main()
