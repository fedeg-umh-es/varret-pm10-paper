# Companion KGE Limitations

- This companion note does not replace the main paper. It inherits the main benchmark and adds a post-evaluation diagnostic layer.
- KGE does not replace baseline-relative Skill, variance retention, Skill_VP, exceedance diagnostics, or Murphy-style decomposition already used in the paper workflow.
- The finding is post-evaluation. It diagnoses already generated rolling-origin forecasts and does not imply a change in model training or forecast generation.
- The result depends on this benchmark: 17 MITECO PM10 stations, five model families, horizons 1 to 7, and the current rolling-origin outputs.
- The station-type breakdown uses categories available in `outputs/tables/master_diagnostic_table.csv`; it should not be generalized to station typologies outside the benchmark.
- The horizon pattern is descriptive. It should not be used to claim a causal mechanism for why KGE divergence changes by lead time.
- KGE component interpretation assumes the definitions implemented in `src/kge_diagnostics.py`: Pearson correlation, standard-deviation ratio, mean ratio, and Euclidean KGE distance from the ideal point.
- The companion note does not establish out-of-domain generalization to other pollutants, cities, temporal resolutions, meteorological covariate designs, or probabilistic forecasts.

