# Canonical Run Order

This run order closes the main `varret-pm10` paper as an all-stations PM10
variance-retention diagnostic. It does not reopen KGE, DPR, DILATE,
probabilistic forecasting, or new model-family experiments.

## 0. Environment

```bash
cd /Users/federicogarciacrespi/Public/varret-pm10-paper
python3 -m pip install -r requirements.txt
```

## 1. Single-station provenance workflow

Use this only when refreshing the original E1-RR provenance artifacts:

```bash
python3 scripts/run_p33_pipeline.py
```

This updates:

- `outputs/metrics/predictions.csv`
- `outputs/metrics/skill_summary.csv`
- `outputs/tables/variance_retention_summary.csv`
- `outputs/reports/run_summary.txt`
- `outputs/figures/figure1_reporting_gap_audit.*`
- `outputs/figures/figure2_skill_variance_retention.*`

## 2. All-stations canonical workflow

The closing evidence package uses the all-stations artifacts already present
under `outputs/metrics/` and `outputs/tables/`.

Run the aggregation and diagnostic steps:

```bash
python3 scripts/build_unified_predictions_table.py
python3 scripts/build_unified_variance_table.py
python3 scripts/build_unified_dm_table.py
python3 scripts/04_build_unified_exceedance_table.py
python3 scripts/07_murphy_decomposition.py --predictions outputs/metrics/predictions_all_stations.csv --output outputs/tables/murphy_decomposition_all_stations.csv
python3 scripts/08_concentration_scale.py --predictions outputs/metrics/predictions_all_stations.csv --output outputs/tables/concentration_scale_summary.csv
python3 scripts/09_build_comprehensive_unified_table.py
python3 scripts/make_collapse_rates_summary.py
python3 scripts/make_alpha_threshold_sensitivity.py
python3 scripts/06_threshold_sensitivity.py
```

Run the final tables and figures:

```bash
python3 scripts/13_build_five_model_diagnostic_summary.py
python3 scripts/14_generate_skill_alpha_figure.py
python3 scripts/10_generate_all_figures.py
```

## 3. Validation

```bash
pytest
```

## 4. Canonical outputs

Primary tables:

- `outputs/tables/master_diagnostic_table.csv`
- `outputs/tables/model_family_diagnostic_summary.csv`
- `outputs/tables/collapse_rates_summary.csv`
- `outputs/tables/skill_alpha_quadrant_counts.csv`
- `outputs/tables/exceedance_all_stations.csv`
- `outputs/tables/murphy_decomposition_all_stations.csv`
- `outputs/tables/variance_retention_all_stations.csv`

Primary figures:

- `outputs/figures/figure3_skill_profiles.pdf`
- `outputs/figures/figure4_alpha_profiles.pdf`
- `outputs/figures/figure5_scatter_skill_alpha.pdf`
- `outputs/figures/figure6_station_collapse_rates.pdf`
- `outputs/figures/figure7_station_map_spain.pdf`
- `outputs/figures/figure_exceedance_recall.pdf`
- `outputs/figures/figure_murphy_decomposition.pdf`
- `outputs/figures/figure_skill_alpha_five_models.pdf`

## 5. Explicitly excluded from main closeout

- `scripts/p34_utils/`: legacy/companion utilities with external absolute paths.
- `results/kge_*`, `scripts/38_*` to `scripts/43_*`, `docs/companion_kge/`:
  companion KGE package only.
- Root-level figure copies: Overleaf convenience exports, not canonical sources.
