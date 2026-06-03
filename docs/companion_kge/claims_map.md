# Companion KGE Claims Map

| claim_id | claim | evidence_files | status | notes |
| --- | --- | --- | --- | --- |
| KGE-C1 | `alpha_h` is empirically identical to `phi_h` in the KGE diagnostic table. | `results/kge_horizon_table.csv`; `results/kge_interpretation.md` | supported | Maximum absolute difference is 0. This is an algebraic identity under the implemented definition. |
| KGE-C2 | KGE adds ranking information beyond Skill in this benchmark. | `results/rank_correlation_kge_phi.csv`; `results/kge_interpretation.md`; `figures/fig_rank_correlation_heatmap.pdf` | supported | Median Spearman Skill-vs-KGE is 0.3 and top-1 Skill/KGE differs in 104 station-horizon cells. |
| KGE-C3 | The ranking divergence is not explained by ranking tie conventions. | `results/tie_sensitivity_rankings.csv`; `results/tie_sensitivity_summary.md`; `results/kge_robustness_closure.md` | supported | Average, dense, and min ranking methods give the same qualitative conclusion; no tie-affected cells were detected. |
| KGE-C4 | KGE complementarity is visible across horizons and is descriptively strongest at long horizons. | `results/horizon_breakdown_kge.csv`; `figures/fig_horizon_divergence_kge.pdf`; `results/kge_robustness_closure.md` | supported | This is descriptive and should not be framed causally. |
| KGE-C5 | The divergence between Skill and KGE is transversal across observed station-type categories. | `results/station_type_breakdown_kge.csv`; `figures/fig_station_type_divergence_kge.pdf`; `results/kge_robustness_closure.md` | supported with caution | Some station-type groups contain one station, so subgroup interpretation must remain restrained. |
| KGE-C6 | KGE should be treated as a complementary post-evaluation diagnostic, not a replacement for Skill, variance retention, or exceedance diagnostics. | `docs/companion_kge/limitations.md`; `docs/companion_kge/relationship_to_main_paper.md` | interpretive | This follows from the scope of the analysis, not from a new experiment. |

