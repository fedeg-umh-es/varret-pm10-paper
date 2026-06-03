# Figure and Table Traceability

This document maps the main `varret-pm10` closing artifacts to their canonical
CSV inputs and producer scripts. Canonical artifacts live in `outputs/`; root
figure copies are Overleaf convenience exports only.

## Scope

- Paper: PM10 variance-retention diagnostic.
- Closing evidence package: all-stations PM10, 17 stations, horizons `h=1..7`.
- Main diagnostics: persistence-relative `Skill(h)`, variance retention
  `alpha(h)`, `Skill_VP(h)`, collapse flags, exceedance relevance, and
  Murphy-style error decomposition.
- Excluded from main closeout: KGE companion outputs, DPR/DILATE, probabilistic
  forecasting, and `paper2H` H* claims.

## Canonical Tables

| Artifact | Producer | Source inputs | Status |
|---|---|---|---|
| `outputs/tables/variance_retention_all_stations.csv` | `scripts/build_unified_variance_table.py` | station-level `outputs/tables/variance_retention_*.csv` | final |
| `outputs/tables/dm_significance_all_stations.csv` | `scripts/build_unified_dm_table.py` | station-level `outputs/metrics/dm_*.csv` | final |
| `outputs/tables/exceedance_all_stations.csv` | `scripts/04_build_unified_exceedance_table.py` | station-level `outputs/metrics/exceedance_*.csv` | final |
| `outputs/tables/murphy_decomposition_all_stations.csv` | `scripts/07_murphy_decomposition.py` | `outputs/metrics/predictions_all_stations.csv` | final |
| `outputs/tables/concentration_scale_summary.csv` | `scripts/08_concentration_scale.py` | `outputs/metrics/predictions_all_stations.csv` | supporting |
| `outputs/tables/master_diagnostic_table.csv` | `scripts/09_build_comprehensive_unified_table.py` | variance retention, DM, exceedance, Murphy, concentration tables | final |
| `outputs/tables/model_family_diagnostic_summary.csv` | `scripts/13_build_five_model_diagnostic_summary.py` | `master_diagnostic_table.csv`, `exceedance_all_stations.csv`, `murphy_decomposition_all_stations.csv` | final |
| `outputs/tables/model_family_diagnostic_summary.tex` | `scripts/13_build_five_model_diagnostic_summary.py` | same as above | final |
| `outputs/tables/model_family_diagnostic_summary.md` | `scripts/13_build_five_model_diagnostic_summary.py` | same as above | final |
| `outputs/tables/skill_alpha_quadrant_counts.csv` | `scripts/14_generate_skill_alpha_figure.py` | `master_diagnostic_table.csv` | final |
| `outputs/tables/collapse_rates_summary.csv` | `scripts/make_collapse_rates_summary.py` | `variance_retention_all_stations.csv` | final |
| `outputs/tables/station_ml_only_collapse_rates.csv` | `scripts/generate_figures_3_to_6.py` | `variance_retention_all_stations.csv` | final |
| `outputs/tables/threshold_sensitivity.csv` | `scripts/06_threshold_sensitivity.py` | variance-retention diagnostics | supporting |
| `outputs/tables/alpha_threshold_sensitivity.csv` | `scripts/make_alpha_threshold_sensitivity.py` | `master_diagnostic_table.csv` | supporting |

## Canonical Figures

| Artifact | Producer | Source inputs | Status |
|---|---|---|---|
| `outputs/figures/figure1_reporting_gap_audit.pdf/png` | `scripts/10_build_figures.py` | embedded reporting-audit counts | supporting motivation |
| `outputs/figures/figure2_skill_variance_retention.pdf/png` | `scripts/10_build_figures.py` | `outputs/tables/variance_retention_summary.csv` | single-station provenance |
| `outputs/figures/figure3_skill_profiles.pdf/png` | `scripts/generate_figures_3_to_6.py` | `outputs/tables/variance_retention_all_stations.csv` | final |
| `outputs/figures/figure4_alpha_profiles.pdf/png` | `scripts/generate_figures_3_to_6.py` | `outputs/tables/variance_retention_all_stations.csv` | final |
| `outputs/figures/figure5_scatter_skill_alpha.pdf/png` | `scripts/generate_figures_3_to_6.py` | `outputs/tables/variance_retention_all_stations.csv` | final |
| `outputs/figures/figure6_station_collapse_rates.pdf/png` | `scripts/generate_figures_3_to_6.py` | `outputs/tables/variance_retention_all_stations.csv` | final |
| `outputs/figures/figure7_station_map_spain.pdf/png` | `scripts/generate_figures_3_to_6.py` | `outputs/tables/variance_retention_all_stations.csv`, station metadata | final |
| `outputs/figures/figure_skill_alpha_five_models.pdf/png` | `scripts/14_generate_skill_alpha_figure.py` | `outputs/tables/master_diagnostic_table.csv` | final |
| `outputs/figures/figure_exceedance_recall.pdf/png` | `scripts/plot_exceedance_figure.py` | `outputs/tables/exceedance_all_stations.csv` | final/supporting |
| `outputs/figures/figure_murphy_decomposition.pdf/png` | `scripts/plot_murphy_decomposition.py` | `outputs/tables/murphy_decomposition_all_stations.csv` | final/supporting |
| `outputs/figures/figure_threshold_sensitivity.pdf/png` | `scripts/plot_threshold_sensitivity.py` | `outputs/tables/threshold_sensitivity.csv` | supporting |
| `outputs/figures/station_map_ml_only_collapse_rate.pdf/png` | `scripts/generate_figures_3_to_6.py` | `outputs/tables/station_ml_only_collapse_rates.csv` | supporting |

## Non-Canonical or Companion Outputs

- Root-level `figure*.pdf/png` files: Overleaf convenience copies.
- `results/kge_*`, `figures/fig_*kge*`, `scripts/38_*` to `scripts/43_*`,
  `docs/companion_kge/`: KGE companion note package.
- `scripts/p34_utils/`: legacy/companion utilities; excluded from the main
  variance-retention run order because they contain external absolute paths.
- `tmp/`, `.pytest_cache/`, and local PDF review files: not submission
  artifacts.
