# Companion KGE Note Package

This package consolidates the post-evaluation KGE companion note for the PM10 variance-retention benchmark. It is a documentation and writing package only: it does not reopen model training, forecasting, feature construction, or the main paper pipeline.

## Purpose

The companion note asks whether the KGE component `alpha_h` is redundant with the paper's variance-retention diagnostic `phi_h`, or whether the other KGE components, `r_h` and `beta_h`, add ranking information beyond `phi_h` in multi-horizon PM10 forecast evaluation.

The closed result is that `alpha_h == phi_h` empirically in the generated KGE table, while KGE rankings diverge materially from Skill and from the `phi_h`-only reading. The robustness closeout labels this a `robust complementary diagnostic`.

## Canonical Inputs

- `outputs/metrics/predictions_all_stations.csv`
- `outputs/tables/master_diagnostic_table.csv`

These are the same evaluated rolling-origin outputs and unified diagnostic table used by the current paper workflow. Forecasts were not regenerated for the KGE analysis.

## Main Scripts

- `scripts/38_compute_kge_horizon.py`
- `scripts/39_rank_comparison_kge_vs_phi.py`
- `scripts/40_figures_kge.py`
- `scripts/41_tie_sensitivity_rankings.py`
- `scripts/42_horizon_breakdown_kge.py`
- `scripts/43_station_type_breakdown_kge.py`

## Final Outputs

- `results/kge_horizon_table.csv`
- `results/rank_correlation_kge_phi.csv`
- `results/tie_sensitivity_rankings.csv`
- `results/horizon_breakdown_kge.csv`
- `results/station_type_breakdown_kge.csv`
- `results/kge_interpretation.md`
- `results/kge_robustness_closure.md`
- `figures/fig_rank_correlation_heatmap.pdf`
- `figures/fig_rank_correlation_heatmap.png`
- `figures/fig_horizon_divergence_kge.pdf`
- `figures/fig_horizon_divergence_kge.png`

## Regeneration Order

```bash
python3 scripts/38_compute_kge_horizon.py
python3 scripts/39_rank_comparison_kge_vs_phi.py
python3 scripts/40_figures_kge.py
python3 scripts/41_tie_sensitivity_rankings.py
python3 scripts/42_horizon_breakdown_kge.py
python3 scripts/43_station_type_breakdown_kge.py
```

## Minimal Dependencies

The scripts use the existing repository dependency set: Python, pandas, numpy, matplotlib, and pytest for tests. No external KGE package is required.

