#!/usr/bin/env python3
"""
Generate:
1. skill_alpha_quadrant_counts.csv
2. figure_skill_alpha_five_models.pdf (Premium Horizon-Trajectory plot of Skill vs. Alpha for 5 models)
3. figure_skill_alpha_five_models_caption.txt (Figure caption)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_CSV = BASE_DIR / "outputs/tables/master_diagnostic_table.csv"
OUT_COUNTS_CSV = BASE_DIR / "outputs/tables/skill_alpha_quadrant_counts.csv"
OUT_FIG_PDF = BASE_DIR / "outputs/figures/figure_skill_alpha_five_models.pdf"
OUT_FIG_PNG = BASE_DIR / "outputs/figures/figure_skill_alpha_five_models.png"
OUT_CAPTION_TXT = BASE_DIR / "outputs/figures/figure_skill_alpha_five_models_caption.txt"

# Models and Families
TARGET_MODELS = [
    "hgb_direct",
    "ridge_direct",
    "sarima",
    "seasonal_naive",
    "stl_ridge_direct"
]

FAMILY_MAP = {
    "hgb_direct": "Direct ML (HGB)",
    "ridge_direct": "Direct ML (Ridge)",
    "sarima": "Statistical (SARIMA)",
    "seasonal_naive": "Naive (Seasonal)",
    "stl_ridge_direct": "Decomposition (STL+Ridge)"
}

COLORS = {
    "hgb_direct": "#008080",       # Teal
    "ridge_direct": "#2E4053",     # Slate Blue
    "sarima": "#8E44AD",           # Purple
    "seasonal_naive": "#F39C12",   # Gold/Orange
    "stl_ridge_direct": "#E74C3C"  # Crimson/Red
}

# Plot markers for horizons
HORIZON_MARKERS = {
    1: "o",
    2: "^",
    3: "s",
    4: "p",
    5: "h",
    6: "8",
    7: "D"
}

def main():
    print("======================================================================")
    print("GENERATING PREMIUM HORIZON-TRAJECTORY SKILL-ALPHA FIGURES")
    print("======================================================================")

    # 1. Load Data
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Missing master diagnostic table: {MASTER_CSV}")
    
    df = pd.read_csv(MASTER_CSV)
    df = df[df["model"].isin(TARGET_MODELS)].copy()

    # 2. Compute Quadrant Counts
    quadrant_rows = []
    for model in TARGET_MODELS:
        mdf = df[df["model"] == model]
        q1 = ((mdf["skill"] > 0) & (mdf["alpha"] < 0.5)).sum()
        q2 = ((mdf["skill"] > 0) & (mdf["alpha"] >= 0.5)).sum()
        q3 = ((mdf["skill"] <= 0) & (mdf["alpha"] < 0.5)).sum()
        q4 = ((mdf["skill"] <= 0) & (mdf["alpha"] >= 0.5)).sum()
        
        row = {
            "model": model,
            "model_family": FAMILY_MAP[model],
            "total_cells": len(mdf),
            "high_skill_collapsed": q1,
            "high_skill_retained": q2,
            "low_skill_collapsed": q3,
            "low_skill_retained": q4
        }
        quadrant_rows.append(row)

    df_counts = pd.DataFrame(quadrant_rows)
    OUT_COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_counts.to_csv(OUT_COUNTS_CSV, index=False)
    print(f"Wrote quadrant counts table to {OUT_COUNTS_CSV}")

    # 3. Create Premium Horizon-Trajectory Plot
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=300)

    # A. Shaded band for near-ideal variance band (0.8 <= alpha <= 1.2)
    ax.axvspan(0.8, 1.2, color="#E8F8F5", alpha=0.5, label="Near-Ideal Variance Band [0.8, 1.2]", zorder=1)

    # B. Reference lines
    ax.axhline(0.0, color="#5D6D7E", linestyle="--", linewidth=1.0, label="Persistence Reference (Skill = 0.0)", zorder=2)
    ax.axvline(0.5, color="#7F8C8D", linestyle=":", linewidth=1.0, label="Collapse Boundary (alpha = 0.5)", zorder=2)
    ax.axvline(1.0, color="#27AE60", linestyle="-.", linewidth=1.0, label="Perfect Variance (alpha = 1.0)", zorder=2)

    # C. Plot Faint Background Scatter Cloud (All 595 points)
    for model in TARGET_MODELS:
        mdf = df[df["model"] == model]
        ax.scatter(
            mdf["alpha"],
            mdf["skill"],
            color=COLORS[model],
            alpha=0.08,
            s=20,
            edgecolors="none",
            zorder=3
        )

    # D. Compute and Plot Median Trajectories (Connected lines h=1 to h=7)
    for model in TARGET_MODELS:
        mdf = df[df["model"] == model]
        # Group by horizon and compute median
        traj = mdf.groupby("horizon")[["alpha", "skill"]].median().sort_index().reset_index()
        
        # Plot trajectory line
        ax.plot(
            traj["alpha"],
            traj["skill"],
            color=COLORS[model],
            linewidth=3.0,
            linestyle="-",
            zorder=4,
            alpha=0.9
        )
        
        # Plot markers for each horizon on the trajectory line
        for _, row in traj.iterrows():
            h = int(row["horizon"])
            ax.scatter(
                row["alpha"],
                row["skill"],
                color=COLORS[model],
                marker=HORIZON_MARKERS[h],
                s=75,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                alpha=1.0
            )

        # Label the endpoints of the trajectory (h=1 and h=7)
        h1_row = traj[traj["horizon"] == 1].iloc[0]
        h7_row = traj[traj["horizon"] == 7].iloc[0]
        
        # Horizontal & vertical offsets for label placement based on model shape
        offset_x = 0.02
        offset_y = 0.02
        if model == "stl_ridge_direct":
            offset_y = -0.04
        elif model == "seasonal_naive":
            offset_x = 0.03
            offset_y = -0.02

        ax.text(
            h1_row["alpha"] + offset_x, h1_row["skill"] + offset_y,
            r"$h=1$", fontsize=8, fontweight="semibold", color=COLORS[model],
            zorder=6, va="center"
        )
        ax.text(
            h7_row["alpha"] + offset_x, h7_row["skill"] + offset_y,
            r"$h=7$", fontsize=8, fontweight="semibold", color=COLORS[model],
            zorder=6, va="center"
        )

    # E. Add Quadrant Label Text (with semi-transparent white bounding boxes)
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.90, edgecolor="#D5D8DC", linewidth=0.8)
    
    # Top-Left: High Skill, Collapsed
    ax.text(0.04, 0.28, "I: High Skill / Collapsed\n(HGB, Ridge, SARIMA)", fontsize=9.5, fontweight="semibold", color="#2C3E50", bbox=bbox_props, va="top", ha="left")
    # Top-Right: High Skill, Retained
    ax.text(0.80, 0.28, "II: High Skill / Retained", fontsize=9.5, fontweight="semibold", color="#27AE60", bbox=bbox_props, va="top", ha="left")
    # Bottom-Left: Low Skill, Collapsed
    ax.text(0.04, -2.0, "III: Low Skill / Collapsed", fontsize=9.5, fontweight="semibold", color="#7F8C8D", bbox=bbox_props, va="bottom", ha="left")
    # Bottom-Right: Low Skill, Retained
    ax.text(1.20, -2.0, "IV: Low Skill / Retained\n(STL+Ridge, Seasonal)", fontsize=9.5, fontweight="semibold", color="#C0392B", bbox=bbox_props, va="bottom", ha="left")

    # Styling
    ax.set_xlabel(r"Variance Retention Coefficient ($\alpha = s_{\hat{y}} / s_{y}$)", fontsize=12, fontweight="semibold", labelpad=8)
    ax.set_ylabel(r"Forecasting Skill (Persistence-Relative $1 - \text{MSE}/\text{MSE}_{\text{pers}}$)", fontsize=12, fontweight="semibold", labelpad=8)
    ax.set_title("The Predictability-Variance Frontier in PM10 Forecasting", fontsize=14, fontweight="bold", pad=15)
    
    # Adjust axis ranges to tightly fit data while leaving space for legend and labels
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(-2.2, 0.35)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#BDC3C7", alpha=0.5)

    # Create dummy handles for the legend to represent lines & markers cleanly
    legend_handles = []
    for model in TARGET_MODELS:
        legend_handles.append(
            plt.Line2D(
                [], [],
                color=COLORS[model],
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=FAMILY_MAP[model]
            )
        )
    
    # Add near-ideal band to legend
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, color="#E8F8F5", alpha=0.6, label="Near-Ideal Band [0.8, 1.2]")
    )

    # Legend in lower left (completely empty region of the scatter plot)
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    
    # Save PDF & PNG
    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_FIG_PNG, format="png", bbox_inches="tight")
    plt.close()
    print(f"Wrote PDF figure to {OUT_FIG_PDF}")
    print(f"Wrote PNG figure to {OUT_FIG_PNG}")

    # 4. Generate Figure Caption Text File
    caption_content = """Figure 5: The Predictability-Variance Frontier and horizon trajectories across 17 stations and 7 forecast horizons (N=595 cells).

The figure overlays a faint scatter plot of all individual station-horizon cells (small semi-transparent markers) with thick solid lines tracking the median trajectory of each model family as the forecast horizon h increases from 1 to 7. 
- Quadrant I (High Skill / Collapsed): Both direct machine learning models (HGB in teal, Ridge in slate blue) and the seasonal ARMA model (SARIMA in purple) start at h=1 with moderate variance retention (alpha approx. 0.4-0.5) and immediately collapse to extreme variance suppression (alpha < 0.15) at longer horizons, moving leftwards along the top.
- Quadrant IV (Low Skill / Retained): Seasonal persistence (Seasonal Naive in gold) maintains alpha approx. 1.0 perfectly across all horizons but is skillless. STL+Ridge (crimson) successfully retains dynamic variance (alpha approx. 1.25-1.72) but suffers a catastrophic skill collapse, starting at skill approx. -1.74 (h=1) and recovering to -0.98 (h=7), remaining far below the persistence baseline in 100% of cells.
- The shaded green band represents the Near-Ideal variance retention range [0.8, 1.2].

This visualization physically demonstrates the 'Predictability-Variance Dilemma' in PM10 forecasting: deterministic models optimized under MSE suppress dynamic variance as horizon increases and predictability decreases to minimize squared errors, whereas forcing variance retention through pre-decomposition (STL+Ridge) without a corresponding increase in true predictability inflates systematic and conditional errors, causing catastrophic skill failure.
"""
    OUT_CAPTION_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CAPTION_TXT, "w", encoding="utf-8") as f:
        f.write(caption_content)
    print(f"Wrote figure caption to {OUT_CAPTION_TXT}")

    print("======================================================================")
    print("FINISHED PIPELINE GENERATION SUCCESSFUL")
    print("======================================================================")

if __name__ == "__main__":
    main()
