# 02_FIGURES_SEPARATE — Figure Manifest & Quality Audit

This directory contains standalone, high-resolution figures prepared for separate upload per *Environmetrics* author guidelines.

## Figure Manifest

| Figure # | Packaged Filenames | Original Repository Path | Source Data Table | Producer Script | Format | Pixel Dimensions | Effective DPI | Journal Guideline Requirement | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Figure 1** | `Figure_1.pdf`<br>`Figure_1.png` | `figures/fig1_horizon_evaluation_divergence.pdf`<br>`figures/fig1_horizon_evaluation_divergence.png` | `outputs/figure_source_tables/fig1_horizon_evaluation_divergence.csv` | `scripts/46_generate_canonical_figures.py` | Vector PDF<br>Raster PNG | Vector (Scalable)<br>$3421 \times 1110$ px | Vector (Lossless)<br>300 DPI | Vector preferred / 600–800 DPI minimum | `PASS` (Vector PDF provides infinite vector resolution; PNG provides 300 DPI web preview) |
| **Figure 2** | `Figure_2.pdf`<br>`Figure_2.png` | `figures/fig2_sarima48_fold_stability.pdf`<br>`figures/fig2_sarima48_fold_stability.png` | `outputs/figure_source_tables/fig2_sarima48_fold_stability.csv` | `scripts/46_generate_canonical_figures.py` | Vector PDF<br>Raster PNG | Vector (Scalable)<br>$3410 \times 1230$ px | Vector (Lossless)<br>300 DPI | Vector preferred / 600–800 DPI minimum | `PASS` (Vector PDF provides infinite vector resolution; PNG provides 300 DPI web preview) |

## Visual Accessibility & Grayscale Audit
- **Figure 1**: Uses distinct high-contrast markers (circles vs. squares) and line styles (solid vs. dashed) for LightGBM vs. SARIMA, ensuring complete distinguishability in grayscale.
- **Figure 2**: Uses distinct hatched bar patterns and high-contrast color coding for individual folds and the persistence baseline, fully distinguishable in grayscale.
