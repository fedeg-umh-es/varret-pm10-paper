#!/usr/bin/env python3
"""
Generate:
1. skill_alpha_quadrant_counts.csv
2. figure_skill_alpha_five_models.pdf (Scatter plot of Skill vs. Alpha for 5 models)
3. figure_skill_alpha_five_models_caption.txt (Figure caption)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path("/Users/federicogarciacrespi/Public/varret-pm10-paper")
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

def main():
    print("======================================================================")
    print("GENERATING SKILL-ALPHA FIGURES AND QUADRANT TABLES")
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

    # 3. Create Scatter Plot (Matplotlib/Seaborn)
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

    # Shaded band for near-ideal variance band (0.8 <= alpha <= 1.2)
    ax.axvspan(0.8, 1.2, color="#E8F8F5", alpha=0.6, label="Near-Ideal Variance Band [0.8, 1.2]", zorder=1)

    # Reference lines
    ax.axhline(0.0, color="#2C3E50", linestyle="--", linewidth=1.2, label="Persistence Reference (Skill = 0.0)", zorder=2)
    ax.axvline(0.5, color="#7F8C8D", linestyle=":", linewidth=1.2, label="Collapse Boundary (alpha = 0.5)", zorder=2)
    ax.axvline(1.0, color="#27AE60", linestyle="-.", linewidth=1.2, label="Perfect Variance Preservation (alpha = 1.0)", zorder=2)

    # Scatter points for each model
    for model in TARGET_MODELS:
        mdf = df[df["model"] == model]
        ax.scatter(
            mdf["alpha"],
            mdf["skill"],
            color=COLORS[model],
            label=FAMILY_MAP[model],
            alpha=0.7,
            edgecolors="none",
            s=45,
            zorder=3
        )

    # Add quadrant label text
    # Coordinates in data space
    # Top-Left: High Skill, Collapsed
    ax.text(0.1, 0.28, "I: High Skill / Collapsed\n(HGB, Ridge, SARIMA)", fontsize=10, fontweight="semibold", color="#2C3E50", alpha=0.8)
    # Top-Right: High Skill, Retained
    ax.text(0.8, 0.28, "II: High Skill / Retained", fontsize=10, fontweight="semibold", color="#27AE60", alpha=0.8)
    # Bottom-Left: Low Skill, Collapsed
    ax.text(0.1, -1.8, "III: Low Skill / Collapsed", fontsize=10, fontweight="semibold", color="#7F8C8D", alpha=0.8)
    # Bottom-Right: Low Skill, Retained
    ax.text(1.2, -1.8, "IV: Low Skill / Retained\n(STL+Ridge, Seasonal)", fontsize=10, fontweight="semibold", color="#C0392B", alpha=0.8)

    # Styling
    ax.set_xlabel(r"Variance Retention Coefficient ($\alpha = s_{\hat{y}} / s_{y}$)", fontsize=12, fontweight="semibold")
    ax.set_ylabel(r"Forecasting Skill (Persistence-Relative $1 - \text{MSE}/\text{MSE}_{\text{pers}}$)", fontsize=12, fontweight="semibold")
    ax.set_title("The Predictability-Variance Frontier in PM10 Forecasting", fontsize=14, fontweight="bold", pad=15)
    
    # Adjust axis ranges
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(-2.2, 0.35)

    # Legend
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=9.5)

    plt.tight_layout()
    
    # Save PDF & PNG
    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_FIG_PNG, format="png", bbox_inches="tight")
    plt.close()
    print(f"Wrote PDF figure to {OUT_FIG_PDF}")
    print(f"Wrote PNG figure to {OUT_FIG_PNG}")

    # 4. Generate Figure Caption Text File
    caption_content = """Figure 5: The Predictability-Variance Frontier and model-family clusterings across 17 stations and 7 forecast horizons (N=595 cells). 

The scatter plot maps persistence-relative forecasting skill (y-axis) against the variance retention coefficient alpha (x-axis, standard deviation of predictions divided by observations). 
- Quadrant I (High Skill / Collapsed): Both direct machine learning models (HGB in teal, Ridge in slate blue) and the seasonal ARMA model (SARIMA in purple) cluster tightly in this quadrant, achieving high forecasting skill but collapsing forecast variance below alpha < 0.5 (median HGB alpha: 0.151, SARIMA: 0.095). 
- Quadrant IV (Low Skill / Retained): Seasonal persistence (Seasonal Naive in gold) preserves variance perfectly (alpha = 1.000) but lacks skill. Additive decomposition regression (STL+Ridge in crimson) succeeds in restoring dynamic variance (alpha = 1.399) but suffers a catastrophic loss of skill (median: -1.107), falling substantially below the persistence baseline in all 119 cells.
- The shaded green region represents the Near-Ideal variance retention band [0.8, 1.2].

This distribution highlights the 'Predictability-Variance Dilemma' in PM10 forecasting: deterministic models optimized under mean squared error loss suppress dynamic variance in low-predictability horizons to minimize errors, whereas forcing variance preservation without a corresponding increase in true predictability inflates systematic and conditional error components, causing catastrophic skill failure.
"""
    OUT_CAPTION_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CAPTION_TXT, "w", encoding="utf-8") as f:
        f.write(caption_content)
    print(f"Wrote figure caption to {OUT_CAPTION_TXT}")

    # Also copy file outputs directly to root folder if requested by user review
    # but the outputs directory is the standard location for git control.
    print("======================================================================")
    print("FINISHED PIPELINE GENERATION SUCCESSFUL")
    print("======================================================================")

if __name__ == "__main__":
    main()
