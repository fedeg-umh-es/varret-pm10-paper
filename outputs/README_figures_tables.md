# VARRET PM10 — Figures and Tables

**Generated**: 2026-06-03  
**Script**: `python3 scripts/make_figures_tables.py`  
**Branch**: `figures-tables-espr`

## Reproduction

```bash
cd /path/to/varret-pm10-paper
python3 scripts/make_figures_tables.py
```

Requirements: `pandas`, `matplotlib`, `numpy` (see `requirements.txt`).

## Figures generated (9/9)

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

## Tables generated (7 tables, 13 files)

| ID | File | Rows | Description |
|----|------|------|-------------|
| table01 | `table01_model_family_diagnostic_summary.{csv,tex}` | 4 | Model family aggregates: skill, α, collapse%, SkillVP, DM significance |
| table02 | `table02_horizon_diagnostic_summary.{csv,tex}` | 28 | Horizon × model family: skill, α, collapse%, retained% |
| table03 | `table03_station_diagnostic_summary.{csv,tex}` | 17 | Station-level: skill, α, collapse%, best family |
| table04 | `table04_skill_retention_quadrants.{csv,tex}` | 7 | Skill–α quadrant counts and percentages |
| table05 | `table05_exceedance_summary.{csv,tex}` | 12 | Exceedance recall/precision/F1 by threshold and model family |
| table06 | `table06_murphy_decomposition_summary.{csv,tex}` | 4 | Murphy MSE decomposition by model family |
| table07 | `table07_figure_source_mapping.csv` | 9 | Traceability: figure → input data → variables |

## Inputs used

- `outputs/tables/master_diagnostic_table.csv` (595 rows, 17 stations, 5 models, 7 horizons)
- `outputs/tables/murphy_decomposition_all_stations.csv` (714 rows)
- `outputs/tables/exceedance_all_stations.csv` (2142 rows)
- `outputs/metrics/predictions_all_stations.csv` (895,737 rows)

## Outputs omitted

None — all 9 figures and 7 tables were generated successfully.

## Meteorological data

**Not available in this repo.** No figures or tables reference meteorological covariates.
All diagnostics use persistence-relative skill, variance retention (α), SkillVP, collapse flags,
exceedance metrics, and Murphy decomposition only.

## Claims supported by these outputs

- Aggregated metrics hide heterogeneity across stations, horizons, and models
- Variance collapse (α < 0.5) is pervasive across Direct ML models
- Persistence-relative skill is positive but coexists with collapsed variance
- Exceedance recall is low across all model families
- Murphy decomposition shows irreducible error dominates
- Station-level and horizon-level diagnostics reveal operationally relevant patterns

## Claims NOT supported

- Meteorological value-added (no meteo data)
- Causal attribution of collapse to specific model features
- External validity beyond NE Spain / these 17 stations
- Any claim about specific model hyperparameters or training details

## Note

This script does NOT modify the manuscript `.tex` files.
Integration into LaTeX will be done separately.
