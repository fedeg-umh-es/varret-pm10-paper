# Companion KGE Runbook

## Relevant Branches

- `main`: contains the initial KGE horizon-wise diagnostic commit.
- `kge-robustness-closeout`: contains robustness closeout and this companion-note consolidation.

## Relevant Commits

- `a070f94 feat(kge): add horizon-wise KGE diagnostics, rank comparison and figures`
- `411cb99 feat(kge): add robustness closeout by ties, horizon and station type`

## Inputs

- `outputs/metrics/predictions_all_stations.csv`
- `outputs/tables/master_diagnostic_table.csv`

No base forecasts were regenerated. All scripts operate on existing post-evaluation outputs.

## Commands

```bash
python3 scripts/38_compute_kge_horizon.py
python3 scripts/39_rank_comparison_kge_vs_phi.py
python3 scripts/40_figures_kge.py
python3 scripts/41_tie_sensitivity_rankings.py
python3 scripts/42_horizon_breakdown_kge.py
python3 scripts/43_station_type_breakdown_kge.py
```

## Tests

```bash
python3 -m pytest tests/test_kge_diagnostics.py tests/test_kge_robustness_outputs.py
```

Last verified result: `9 passed`.

## Success Criteria Reached

- KGE table contains 595 station-model-horizon cells.
- Rank comparison table contains 119 station-horizon comparisons.
- `alpha_h == phi_h` with maximum absolute difference equal to 0.
- Robustness closure verdict is `robust complementary diagnostic`.
- Editorial recommendation is `keep as companion note`.

