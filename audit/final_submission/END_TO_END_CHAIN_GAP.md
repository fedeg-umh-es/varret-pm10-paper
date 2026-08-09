# Paper A end-to-end reproducibility chain gap audit

Audit date: 2026-08-10

No replacement pipeline was written or executed. The table below records
whether the existing evidence and scripts can connect the required stages
without refitting, guessing or tuning.

| Stage | Required input | Available input | Existing script | Output | Verified | Gap |
|---|---|---|---|---|---|---|
| Row-level predictions → matched persistence | `y_true`, `y_pred`, origin, target, horizon, station and aligned persistence | Public release has model/persistence rows, but not SARIMA-aligned persistence before 2020; two local station files have partial historical rows | `scripts/02_generate_sarima_predictions.py`, `scripts/05_dm_significance.py` | matched prediction pairs | NO | Missing 17-station SARIMA-aligned baseline; generator would refit and is not run. |
| Matched predictions → RMSE/skill | Exact model and baseline pairs for every station/model/horizon | Aggregate skill exists; non-SARIMA skill is row-level recoverable; SARIMA 119 cells are not | `scripts/06_build_skill_tables.py` | skill table | NO | `06_build_skill_tables.py` is a placeholder; SARIMA source pair is missing. |
| Row-level predictions → alpha | Matched `y_true`/`y_pred` groups | Public release plus variance tables | `scripts/build_unified_variance_table.py` and independent consolidation logic | variance-retention table | PARTIAL | Public documentation supports alpha for all 595, but the complete current chain is not packaged with one verified manifest. |
| Row-level predictions → P75 recall | Origin-aware observed history, model predictions and threshold rule | Public release and event script | `scripts/03_exceedance_analysis.py` | exceedance table | PARTIAL | Code exists, but public consolidation did not independently reconcile event values. |
| Row-level predictions → DM/HLN/BH | Exact model/persistence loss pairs and family definition | DM aggregate plus script | `scripts/05_dm_significance.py` | DM table | PARTIAL | Script requires missing SARIMA baseline; aggregate values were not independently recomputed in consolidation. |
| Metrics → master diagnostic table | Variance, skill, DM and event tables keyed by station/model/horizon | 595-row master aggregate | `scripts/09_build_comprehensive_unified_table.py` | master table | PARTIAL | Script merges diagnostics but does not independently regenerate skill or verify the recovered source chain. |
| Master table → Rule A/B | `skill`, `dm_significant`, `alpha`, `recall_p75` | Master table and local decision script | `scripts/15_decision_change_analysis.py` | 277, 8, 269, 97.1% | PARTIAL | Deterministic aggregate transformation exists locally, but it cannot repair missing row-level provenance and is not present in the public evidence tag. |
| Rule A/B → formal discordance | Rule A, recall P75 and alpha | Master table and local verifier | `audit/decision_change/verify_decision_change.py` | 101 cases | PARTIAL | Exact aggregate check passes locally; upstream all-station row-level chain remains incomplete. |
| Canonical table → figures/tables | Versioned source table and figure generators | Aggregate/source tables and plotting scripts | figure scripts and paper package generators | publication figures/tables | PARTIAL | Deterministic outputs exist, but current run identity and full row-level source reconciliation are not complete. |
| Entire chain → canonical 595-row table | All stages above with exact hashes and protocol manifest | Local canonical table and public partial package | no complete entrypoint | canonical table | NO | Aggregate agreement is not accepted as proof of Grade-A row-level provenance. |

## Reproduction decision

The missing work is not merely orchestration. The row-level input required
for the SARIMA baseline comparison is absent, and the existing run manifest
describes a conflicting older protocol. A wrapper could therefore only
produce an apparently matching output by depending on the aggregate table
or undocumented assumptions, which is explicitly disallowed.

`REPRODUCIBILITY_WRAPPER_CREATED = NO`

`REPRODUCIBILITY_CHAIN_COMPLETE = NO`

`END_TO_END_CHAIN_STATUS = BLOCKED`

## Independent local aggregate check

The local integrity checker still reproduces the aggregate contract:

- 595 rows;
- Rule A 277;
- Rule B 8;
- changes 269;
- 97.1%;
- formal discordance 101;
- rho -0.863.

This is an aggregate integrity result, not an end-to-end Grade-A recovery.
