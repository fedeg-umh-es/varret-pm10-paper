# KGE robustness closure

## 1. New Scripts Executed

- scripts/41_tie_sensitivity_rankings.py
- scripts/42_horizon_breakdown_kge.py
- scripts/43_station_type_breakdown_kge.py

## 2. Inputs Used

- outputs/metrics/predictions_all_stations.csv
- outputs/tables/master_diagnostic_table.csv
- results/kge_horizon_table.csv
- results/rank_correlation_kge_phi.csv

## 3. Tie Sensitivity

| rank_method | n_station_horizon_cells | median_spearman_skill_vs_kge | median_spearman_phi_vs_r | rank_reversal_rate | top_skill_top_kge_mismatch_cells | tie_affected_cells | tie_affected_pct | tied_metric_instances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| average | 119 | 0.3 | -0.5 | 1 | 104 | 0 | 0 | 0 |
| dense | 119 | 0.3 | -0.5 | 1 | 104 | 0 | 0 | 0 |
| min | 119 | 0.3 | -0.5 | 1 | 104 | 0 | 0 | 0 |

Result: ranking-method conclusion is stable across average, dense and min tie handling.

## 4. Horizon Breakdown

| h | median_skill_h | median_phi_h | median_r_h | median_beta_h | median_KGE_h | median_spearman_skill_vs_kge | median_spearman_phi_vs_r | rank_reversal_cells | rank_reversal_rate | top_skill_top_kge_mismatch_cells | n_station_horizon_cells | n_station_model_horizon_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0583157 | 0.709576 | 0.504203 | 1.00795 | 0.343477 | 1 | -0.6 | 17 | 1 | 7 | 17 | 85 |
| 2 | 0.141538 | 0.5158 | 0.292564 | 1.00429 | 0.122557 | 0.3 | -0.8 | 17 | 1 | 14 | 17 | 85 |
| 3 | 0.171974 | 0.417496 | 0.22763 | 1.00568 | 0.0246445 | 0.3 | -0.7 | 17 | 1 | 16 | 17 | 85 |
| 4 | 0.197364 | 0.373823 | 0.184765 | 1.00545 | -0.0452487 | 0.1 | -0.5 | 17 | 1 | 16 | 17 | 85 |
| 5 | 0.188699 | 0.358645 | 0.164801 | 1.00195 | -0.0954876 | 0.3 | 0.1 | 17 | 1 | 17 | 17 | 85 |
| 6 | 0.203883 | 0.370996 | 0.159519 | 1.0051 | -0.111664 | 0.1 | 0.1 | 17 | 1 | 17 | 17 | 85 |
| 7 | 0.188788 | 0.371119 | 0.152289 | 1.00286 | -0.134712 | 0.3 | 0 | 17 | 1 | 17 | 17 | 85 |

Result: KGE complementarity is descriptively strongest in long horizons by median Skill-vs-KGE rank divergence.

## 5. Station-Type Breakdown

| station_type | n_stations | n_station_horizon_cells | median_spearman_skill_vs_kge | median_spearman_phi_vs_r | rank_reversal_rate | top_skill_top_kge_mismatch_cells |
| --- | --- | --- | --- | --- | --- | --- |
| Rural Near-city/Industrial | 1 | 7 | 0.3 | 0 | 1 | 6 |
| Rural Remote EMEP | 1 | 7 | 0.4 | -0.8 | 1 | 6 |
| Rural Remote/Background | 1 | 7 | 0.5 | -0.5 | 1 | 5 |
| Rural Remote/Industrial | 1 | 7 | 0.3 | 0.3 | 1 | 6 |
| Suburban/Industrial | 3 | 21 | 0.1 | -0.5 | 1 | 17 |
| Suburban/Traffic | 1 | 7 | 0.3 | 0.4 | 1 | 6 |
| Urban/Background | 4 | 28 | 0.1 | -0.6 | 1 | 26 |
| Urban/Industrial | 1 | 7 | 0.3 | 0 | 1 | 7 |
| Urban/Residential | 1 | 7 | 0.3 | -0.8 | 1 | 6 |
| Urban/Traffic | 3 | 21 | 0.3 | -0.5 | 1 | 19 |

Result: divergence is transversal across station types; small subgroups should not be overinterpreted.

## 6. Final Verdict

robust complementary diagnostic

## 7. Editorial Recommendation

keep as companion note
