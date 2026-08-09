# P4 final remote Grade-A audit

Audit date: 2026-08-10

## Scope and governing canon

The canonical project specification used for this audit is:

`docs/p4_legacy_closeout/P4_PROJECT_CANON.md`

It is version 1.4, dated 2026-08-08. An older copy exists in the adjacent
`P4_Ghost_Skill_Dynamic_Fidelity` repository; that copy was not used as the
source of truth.

The canon defines Grade A as complete evidence containing row-level
predictions, observed values, model and baseline, station, origin, horizon,
fold, reproducible metrics, source tables, scripts, and manifests. It also
requires all-station predictions, station metadata, variance-retention
outputs, event metrics, figure source tables, and reproducible aggregation
for a strong 17-station claim.

## Local integrity gate

`python3 audit/lightgbm_robustness/verify_canonical_integrity.py` passed
16/16 checks before this audit. The local aggregate contract is therefore
intact, but this does not by itself establish Grade A provenance.

## Public remote inspected

Remote: `github`

Repository: `fedeg-umh-es/varret-pm10-paper`

Public refs inspected with `git ls-remote github`:

- `refs/heads/codex/p4-lightgbm-ems-gap-audit` →
  `12fb7bb8446f8f606dc2f75bb689a4561a61178c`
- `refs/heads/main` → `4a49b08b041c578ec5981dc1472125b2af0a4d59`
- `refs/heads/paper/paper-a-evidence-package` →
  `8f7e345baa9a58dd3af3de5022f69a433ce42fc0`
- `refs/tags/paper-a-row-level-evidence-v1` →
  `4e3a97adb21fb43f3dade44fb50554bb15a48173`

The row-level asset is publicly distributed by the non-draft release
`paper-a-row-level-evidence-v1`:

`https://github.com/fedeg-umh-es/varret-pm10-paper/releases/tag/paper-a-row-level-evidence-v1`

Asset: `predictions_all_stations.csv`

- size: 77,080,400 bytes;
- rows: 895,737 plus header;
- SHA-256:
  `2551b3a2acf549e94cd92e39386b96aca2358cca41fed7f85c933d35c77ef823`;
- columns: `dataset, model, fold, origin_date, horizon, date, y_true, y_pred`;
- coverage: 17 datasets, 7 horizons, five evaluated models plus persistence.

## Requirement audit

| Requirement | Canonical expectation | Remote artifact | Path/ref | Schema verified | Reproducible | Status | Notes |
|---|---|---|---|---:|---:|---|---|
| Row-level predictions | All-station prediction rows | Public release asset | `paper-a-row-level-evidence-v1/predictions_all_stations.csv` | YES | PARTIAL | PARTIAL | Asset is public and complete in row count, but the documented SARIMA baseline pairing is incomplete. |
| `y_true` | Observed target per prediction row | Release asset | same asset | YES | YES | PASS | Non-null `y_true` column present. |
| `y_pred` | Prediction per row | Release asset | same asset | YES | YES | PASS | Non-null `y_pred` column present. |
| Baseline | Persistence aligned to each model comparison | Persistence rows in asset | `model=persistence` | PARTIAL | NO for all cells | PARTIAL | Persistence rows exist, but the 2018–2019 SARIMA-aligned persistence rows are absent from the released combined asset. |
| Station | Station identity per row | `dataset` plus station metadata | `evidence/paper_a/metadata/station_metadata.csv` | YES | YES | PASS | 17 datasets reconcile to the public metadata. |
| Origin | Forecast origin | `origin_date` and `fold` columns | release asset | YES | YES | PASS | Dates are present at row level. |
| Horizon | Horizon per row | `horizon` column | release asset | YES | YES | PASS | h=1,...,7 present. |
| Fold | Fold/origin grouping | `fold` column | release asset | YES | PARTIAL | PARTIAL | A date-valued fold field is present; the public run manifest does not describe the current seven-horizon protocol consistently. |
| 17-station metadata | Reconciliation of all stations | Station metadata CSV | `evidence/paper_a/metadata/station_metadata.csv` | YES | YES | PASS | Station IDs, names, provinces, types and coordinates are present. |
| Error metrics | RMSE/skill and significance status | Master and DM aggregates | `evidence/paper_a/aggregates/master_diagnostic_table.csv`, `dm_significance_all_stations.csv` | YES | PARTIAL | PARTIAL | Aggregate values exist, but exact SARIMA skill is not independently recoverable from the release asset; DM values were schema-validated, not independently recomputed in the consolidation. |
| Variance retention | Alpha by station/model/horizon | Aggregate and station source tables | `variance_retention_all_stations.csv`, `source_tables/` | YES | YES | PASS | The public documentation reports independent alpha recomputation for all 595 cells. |
| P75 recall | Event metric by station/model/horizon | Exceedance aggregate plus event script | `exceedance_all_stations.csv`, `scripts/03_exceedance_analysis.py` | YES | PARTIAL | PARTIAL | Artifact and code exist, but the consolidated public package documents no independent numerical recomputation of event outputs. |
| Rule A/B | Explicit derivation of eligibility rules | Aggregate columns/prose only | master aggregate and manuscript | PARTIAL | NO | PARTIAL | No versioned public script was found that derives Rule A, Rule B, 277, 8, 269 and 97.1% from the released evidence. |
| Aggregation script | Predictions → metrics → eligibility → tables | Partial metric builders | `scripts/03_exceedance_analysis.py`, `05_dm_significance.py`, `09_build_comprehensive_unified_table.py` | PARTIAL | NO for complete chain | FAIL | The available scripts do not form a complete, current, end-to-end Rule A/B reproduction chain. `06_build_skill_tables.py` is a placeholder. |
| Run manifest | Current protocol, versions and hashes | Reproduction manifest | `outputs/reproduction/run_manifest_rolling_origin.json` | YES | NO | FAIL | Manifest describes a different older protocol: horizons 1, 6, 24, 48 and LightGBM/SARIMA, not the current five-model, seven-horizon paper protocol. |
| Figure source tables | Deterministic sources for main figures | Aggregate/source tables and plotting scripts | `evidence/paper_a/aggregates/`, `source_tables/`, `scripts/14_generate_skill_alpha_figure.py` | PARTIAL | PARTIAL | PARTIAL | Source material exists, but the complete current manuscript figure chain is not reconciled to a single current run manifest. |
| Rank reversal table | Canonical P4 output where required | `results/rank_reversal_summary.csv` | tag root `results/rank_reversal_summary.csv` | YES | PARTIAL | NOT_APPLICABLE_TO_CURRENT_CLAIM | It is present, but Paper A currently claims eligibility change rather than rank reversal; it was audited and not added to the manuscript. |
| Manifest/provenance | Source/version/hash reconciliation | Provenance CSV and README | `evidence/paper_a/manifests/provenance_manifest.csv`, `evidence/paper_a/README.md` | YES | PARTIAL | PARTIAL | File hashes and recovery provenance are documented, but the source is recovered from a local copy and not a clean re-run provenance chain. |

## Canonical-output classification

| Canonical output | Classification | Remote evidence |
|---|---|---|
| `predictions_row_level` | `REMOTE_PRESENT` / `PARTIAL` for full current claim | Public release asset; SARIMA-aligned baseline limitation remains. |
| `metrics_by_station_horizon` | `REMOTE_PRESENT` / `PARTIAL` | Master and DM aggregates exist; full independent recomputation is incomplete. |
| `variance_retention_by_station_horizon` | `REMOTE_PRESENT` | 595-row aggregate and 17 station source tables. |
| `event_metrics_by_station_horizon` | `REMOTE_PRESENT` / `PARTIAL` | Exceedance aggregate and generation script exist; independent numerical recomputation was not completed in the consolidation. |
| `rank_reversal_table` | `REMOTE_PRESENT` | `results/rank_reversal_summary.csv`; not required for the current eligibility-change claim. |
| `station_metadata` | `REMOTE_PRESENT` | 17-station metadata CSV. |
| `run_manifest` | `REMOTE_PRESENT` / `CONFLICTING` | Public manifest describes an older, different protocol. |
| `figure_source_tables` | `REMOTE_PRESENT` / `PARTIAL` | Deterministic source material exists, but the current end-to-end reconciliation is incomplete. |

## Remote reproducibility test

The public asset was downloaded and its published SHA-256 verified. Its
schema and coverage recover:

- 895,737 row-level records;
- 17 station datasets;
- five evaluated models plus persistence;
- seven horizons;
- non-null observed and predicted values.

The remote chain does not, however, recover all current paper results
without relying on unreleased or non-recomputed information:

1. The release documentation explicitly states that the 119 SARIMA
   station-by-horizon skill values are not independently reproducible from
   the combined release asset because the aligned persistence rows for the
   earlier period are absent. The source `skill_sarima_{station}.csv` files
   are not present in the public tag.
2. The public package includes DM and event aggregates and their builders,
   but its README states that their numerical values were schema-validated,
   not independently recomputed during consolidation.
3. No versioned public rule-derivation script was found that connects the
   released row-level evidence to Rule A, Rule B, 277, 8, 269, 97.1%, and
   the 101 formal discordant cases.
4. The public rolling-origin manifest describes a conflicting older protocol
   with four hourly horizons and LightGBM/SARIMA, so it cannot serve as the
   manifest for the current daily five-model, seven-horizon paper claim.

The local 16/16 aggregate integrity result remains valid, but it cannot
override these remote provenance and reproducibility limitations.

## Grade-A verdict

`GRADE_A_REMOTE_BLOCKED_BY_INCOMPLETE_REPRODUCIBILITY`

The strong 17-station claim is not Grade-A verified under the governing
canon. This is an evidence-grade decision, not a finding that the local
aggregate table is numerically corrupted.

### Minimum missing artifact / chain

At minimum, the public evidence package must add a current, versioned and
reproducible chain containing:

1. the SARIMA-aligned persistence/prediction source needed to reproduce all
   119 SARIMA skill cells (or the corresponding `skill_sarima_{station}.csv`
   outputs for all 17 stations);
2. a current manifest matching the daily five-model, seven-horizon
   protocol; and
3. a versioned deterministic aggregation/decision script that derives the
   metrics, Rule A/B counts and formal discordance from those sources.

The public DM and P75 outputs must also be independently reconciled by that
chain before the claim can be promoted to `GRADE_A_REMOTE_VERIFIED`.

## Execution gate

Because the verdict is not `GRADE_A_REMOTE_VERIFIED`, the mandated hard
stop applies:

- Phase B was not executed;
- no manuscript files were modified;
- no Data Availability statement was modified;
- no package was rebuilt;
- no commit was created;
- no push was performed.

Final gate status: `SUBMISSION_BLOCKED_BY_EVIDENCE_GRADE`.
