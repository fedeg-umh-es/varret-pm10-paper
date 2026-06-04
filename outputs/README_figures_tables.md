# VARRET PM10 — Figures and Tables

**Generated**: 2026-06-04  
**Script**: `python3 scripts/make_figures_tables.py`  
**Branch**: `figures-tables-espr`

## Reproduction

```bash
cd /path/to/varret-pm10-paper
python3 scripts/make_figures_tables.py
```

Requirements: `pandas`, `matplotlib`, `numpy` (see `requirements.txt`).

## Figures generated (12/12)

| ID | File | Description |
|----|------|-------------|
| fig01 | `fig01_station_map.{pdf,png}` | PM10 station map — NE Spain, colored by class, sized by collapse rate |
| fig02 | `fig02_evaluation_workflow.{pdf,png}` | Leakage-free rolling-origin evaluation workflow (conceptual) |
| fig03 | `fig03_station_horizon_heatmap.{pdf,png}` | Station × horizon heatmap of variance retention (α) — Direct ML |
| fig04 | `fig04_horizon_distribution.{pdf,png}` | Horizon-wise boxplots of skill and α (all models × stations) |
| fig05 | `fig05_skill_variance_plane.{pdf,png}` | Skill vs. variance retention scatter — all cells, by model family |
| fig06 | `fig06_station_collapse_rates.{pdf,png}` | Station-level collapse rates (% cells with α < 0.5) |
| fig07 | `fig07_exceedance_diagnostics.{pdf,png}` | Exceedance detection: recall and F1 by model family and threshold |
| fig08 | `fig08_murphy_decomposition.{pdf,png}` | Murphy-style MSE decomposition by model family |
| fig09 | `fig09_episode_timeseries.{pdf,png}` | Representative episode window (h=1, observed vs. forecasts) |
| fig10 | `fig10_meteo_delta_skill_by_horizon.{pdf,png}` | Meteorological value-added: Δ skill by horizon (Madrid + Ireland) |
| fig11 | `fig11_meteo_phi_comparison.{pdf,png}` | Variance retention with vs. without meteorological covariates |
| fig12 | `fig12_meteo_ablation.{pdf,png}` | Covariate ablation — wind, temp, humidity, all combined (Madrid) |

## Tables generated (9 tables, 17 files)

| ID | File | Rows | Description |
|----|------|------|-------------|
| table01 | `table01_model_family_diagnostic_summary.{csv,tex}` | 4 | Model family aggregates: skill, α, collapse%, SkillVP, DM significance |
| table02 | `table02_horizon_diagnostic_summary.{csv,tex}` | 28 | Horizon × model family: skill, α, collapse%, retained% |
| table03 | `table03_station_diagnostic_summary.{csv,tex}` | 17 | Station-level: skill, α, collapse%, best family |
| table04 | `table04_skill_retention_quadrants.{csv,tex}` | 7 | Skill–α quadrant counts and percentages |
| table05 | `table05_exceedance_summary.{csv,tex}` | 12 | Exceedance recall/precision/F1 by threshold and model family |
| table06 | `table06_murphy_decomposition_summary.{csv,tex}` | 4 | Murphy MSE decomposition by model family |
| table07 | `table07_figure_source_mapping.csv` | 12 | Traceability: figure → input data → variables |
| table08 | `table08_meteo_value_added_summary.{csv,tex}` | 2 | Meteorological value-added: Madrid vs. Ireland comparison |
| table09 | `table09_meteo_dm_significance.{csv,tex}` | 36 | Diebold-Mariano significance: meteo vs. lags-only |

## Inputs used

### Primary (NE Spain — 17 stations)
- `outputs/tables/master_diagnostic_table.csv` (595 rows, 17 stations, 5 models, 7 horizons)
- `outputs/tables/murphy_decomposition_all_stations.csv` (714 rows)
- `outputs/tables/exceedance_all_stations.csv` (2142 rows)
- `outputs/metrics/predictions_all_stations.csv` (895,737 rows)

### External meteorology (from benchmark-pm-hstar)
- `data/external/madrid_meteo/table_delta_lags_meteo_vs_lags_only.csv` (24 horizons)
- `data/external/madrid_meteo/master_meteorology_diagnostic_table.csv` (384 rows, 8 stations × 24 horizons × 2 conditions)
- `data/external/madrid_meteo/meteorology_ablation_summary.csv` (5 conditions)
- `data/external/madrid_meteo/dm_lags_meteo_vs_lags_only.csv` (4 horizons)
- `data/external/ireland_meteo/table_delta_skill_meteo_vs_lags.csv` (192 rows, 8 stations × 24 horizons)
- `data/external/ireland_meteo/dm_lags_meteo_vs_lags_only.csv` (32 cells)

## Outputs omitted

None — all 12 figures and 9 tables were generated successfully.

## Meteorological value-added evidence

Meteorological data from two independent networks (Madrid, Ireland) is used from the
`benchmark-pm-hstar` companion repo to test whether adding meteorological covariates
(wind, temperature, humidity, pressure, precipitation, solar radiation) resolves the
variance collapse observed in lags-only models.

**Key findings:**
- Median Δ skill (RMSE) = +4.25 pp (Madrid), +2.10 pp (Ireland) — positive but marginal
- φ_h (variance retention) increases from 0.63 → 0.67 with meteorology — collapse persists
- DM significance: 7/36 tested cells significant (p < 0.05), all favouring meteo
- Ablation: individual covariates contribute +1.5–1.8 pp; combined +2.4 pp
- Meteorological covariates do NOT resolve variance collapse

## Claims supported by these outputs

- Aggregated metrics hide heterogeneity across stations, horizons, and models
- Variance collapse (α < 0.5) is pervasive across Direct ML models (99.2%)
- Persistence-relative skill is positive but coexists with collapsed variance (57.65% of cells)
- Exceedance recall is low across all model families
- Murphy decomposition shows irreducible error dominates
- Station-level and horizon-level diagnostics reveal operationally relevant patterns
- **Meteorological covariates provide marginal improvement (+2–4 pp) but do not resolve collapse**
- **The collapse phenomenon is reproducible across three independent networks (NE Spain, Madrid, Ireland)**

## Claims NOT supported

- Causal attribution of collapse to specific model features (ablation is correlational)
- External validity beyond Mediterranean/Atlantic European climate
- Any claim about specific model hyperparameters or training details

## Note

This script does NOT modify the manuscript `.tex` files.
Integration into LaTeX will be done separately.
