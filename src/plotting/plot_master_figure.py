"""
Plot master figure
==================
Main visualization: Skill, Var%, Skill_VP, Recall vs horizon.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Generate master figure."""
    parser = argparse.ArgumentParser(description="Plot master figure")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    with open("config/horizons.yaml") as f:
        horizons_cfg = yaml.safe_load(f)
    horizons = sorted(horizons_cfg['horizons'])
    
    metrics_dir = Path(cfg['paths']['metrics_dir'])
    figures_dir = Path(cfg['paths']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print("Loading metrics...")
    df_ro = pd.read_csv(metrics_dir / "metrics_rolling_origin_by_horizon.csv")
    df_events = pd.read_csv(metrics_dir / "metrics_events_rolling_origin.csv")
    
    # Filter to LightGBM and LSTM (exclude persistence)
    df_ro = df_ro[df_ro['model'].isin(['lgbm', 'lstm'])]
    df_events = df_events[df_events['model'].isin(['lgbm', 'lstm'])]
    
    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=cfg['plotting']['figsize'], dpi=cfg['plotting']['dpi'])
    fig.suptitle('Paper A: Ghost Skill & Variance Collapse Diagnostics', fontsize=14, fontweight='bold')
    
    colors = cfg['plotting']['colors']
    models = ['lgbm', 'lstm']
    model_labels = {'lgbm': 'LightGBM', 'lstm': 'LSTM'}
    model_colors = {m: colors[m] for m in models}
    
    # Panel 1: Skill vs Horizon (Ghost Skill Effect)
    ax = axes[0, 0]
    for model in models:
        data = df_ro[df_ro['model'] == model]
        ax.plot(data['h'], data['skill'], 'o-', label=model_labels[model], 
               color=model_colors[model], linewidth=2, markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Baseline (skill=0)', alpha=0.7)
    ax.fill_between(horizons, -1, 0, alpha=0.1, color='red', label='Ghost Skill Zone')
    ax.set_xlabel('Horizon (hours)', fontsize=11)
    ax.set_ylabel('Skill', fontsize=11)
    ax.set_title('Panel 1: Skill vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(horizons)
    
    # Panel 2: Variance % vs Horizon (Variance Collapse)
    ax = axes[0, 1]
    for model in models:
        data = df_ro[df_ro['model'] == model]
        ax.plot(data['h'], data['var_pct'], 's-', label=model_labels[model],
               color=model_colors[model], linewidth=2, markersize=8)
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='Full Variance', alpha=0.7)
    ax.fill_between(horizons, 0, 100, alpha=0.1, color='green')
    ax.fill_between(horizons, 100, 200, alpha=0.1, color='red', label='Variance Expansion')
    ax.set_xlabel('Horizon (hours)', fontsize=11)
    ax.set_ylabel('Variance Percentage (%)', fontsize=11)
    ax.set_title('Panel 2: Var% vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(horizons)
    
    # Panel 3: Skill_VP vs Horizon (Diagnostic Metric)
    ax = axes[1, 0]
    for model in models:
        data = df_ro[df_ro['model'] == model]
        ax.plot(data['h'], data['skill_vp'], '^-', label=model_labels[model],
               color=model_colors[model], linewidth=2, markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Horizon (hours)', fontsize=11)
    ax.set_ylabel('Skill_VP', fontsize=11)
    ax.set_title('Panel 3: Skill_VP vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(horizons)
    
    # Panel 4: Recall of Exceedances (P75)
    ax = axes[1, 1]
    for model in models:
        data = df_events[(df_events['model'] == model) & (df_events['threshold'] == 75)]
        ax.plot(data['h'], data['recall_events'], 'D-', label=model_labels[model],
               color=model_colors[model], linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Horizon (hours)', fontsize=11)
    ax.set_ylabel('Recall of Events (P75)', fontsize=11)
    ax.set_title('Panel 4: Event Recall vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(horizons)
    ax.set_ylim([0, 1])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path = figures_dir / "master_figure.png"
    plt.savefig(output_path, dpi=cfg['plotting']['dpi'], bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
