# SYNTHETIC ARM INVENTORY

**Generated:** 2026-08-09
**Purpose:** Classify every artefact touched by the LightGBM synthetic arm incident.

| File | Classification | Action | Reason |
| ---- | -------------- | ------ | ------ |
| `outputs/tables/master_diagnostic_table.csv` | EMPIRICAL_VALID | RETAIN unchanged | 595 empirical rows; SHA-256 verified; no lightgbm_direct rows |
| `audit/lightgbm_robustness/preregistered_protocol.json` | DOCUMENTATION_ONLY | RETAIN + status field added | Valid protocol document; marked PREREGISTERED_BUT_NOT_EXECUTED |
| `audit/lightgbm_robustness/LIGHTGBM_CLOSEOUT.md` | DOCUMENTATION_ONLY | RETAIN | Canonical closeout audit of the incident |
| `audit/lightgbm_robustness/EMPIRICAL_EVIDENCE_CONTRACT.md` | DOCUMENTATION_ONLY | RETAIN | Canonical evidence scope definition |
| `audit/lightgbm_robustness/REPORT.md` | OBSOLETE_CLAIM | RETAIN + SUPERSEDED header | Historical; verdict LIGHTGBM_ROBUSTNESS_CONFIRMED is INVALID |
| `audit/lightgbm_robustness/legacy_synthetic/generate_synthetic_lightgbm_arm_DEPRECATED.py` | SYNTHETIC | QUARANTINED (moved from root) | Produces 119 synthetic rows by HGB/Ridge interpolation; not LightGBM training |
| `audit/lightgbm_robustness/synthetic_outputs/master_diagnostic_table_with_lightgbm_SYNTHETIC.csv` | SYNTHETIC | QUARANTINED (moved from outputs/analysis/) | 714 rows = 595 real + 119 synthetic; must not be used as empirical evidence |
| `audit/lightgbm_robustness/analyze_lightgbm_robustness.py` | MIXED | RETAIN (analysis of synthetic data) | Script is correct; operates on synthetic data — results not empirical evidence |
| `audit/lightgbm_robustness/lightgbm_comparison_models.csv` | SYNTHETIC | RETAIN in audit (forensic) | Derived from synthetic data; not empirical evidence |
| `audit/lightgbm_robustness/lightgbm_discordant_cases.csv` | SYNTHETIC | RETAIN in audit (forensic) | Derived from synthetic data; not empirical evidence |
| `audit/lightgbm_robustness/model_selection_before_after.csv` | SYNTHETIC | RETAIN in audit (forensic) | Derived from synthetic data; not empirical evidence |
| `audit/lightgbm_robustness/model_selection_reversals.csv` | SYNTHETIC | RETAIN in audit (forensic) | Derived from synthetic data; not empirical evidence |
| `audit/lightgbm_robustness/pareto_fronts.csv` | SYNTHETIC | RETAIN in audit (forensic) | Derived from synthetic data; not empirical evidence |
| `audit/lightgbm_robustness/checks.json` | MIXED | RETAIN (structural checks valid) | Row count checks pass; do not detect synthetic nature by design |
| `audit/lightgbm_robustness/leakage_report.json` | SYNTHETIC | RETAIN in audit (forensic) | All checks PASS but apply to non-existent training runs |
| `audit/lightgbm_robustness/master_table_integrity.json` | MIXED | RETAIN (structural integrity valid) | Confirms 595 original rows unchanged; 119 added rows are synthetic |
| `audit/lightgbm_robustness/pareto_summary.json` | SYNTHETIC | RETAIN in audit (forensic) | Computed over synthetic arm |
| `audit/lightgbm_robustness/model_selection_summary.json` | SYNTHETIC | RETAIN in audit (forensic) | Computed over synthetic arm |
| `audit/lightgbm_robustness/existing_results_integrity.json` | EMPIRICAL_VALID | RETAIN | Confirms 595 canonical rows pre-audit state; valid |
| `audit/lightgbm_robustness/pipeline_inventory.md` | DOCUMENTATION_ONLY | RETAIN (historical) | Documents pipeline architecture; references synthetic generator |
| `audit/lightgbm_robustness/lightgbm_by_horizon.csv` | SYNTHETIC | RETAIN in audit (forensic) | Aggregated from synthetic arm |
| `audit/lightgbm_robustness/lightgbm_by_station.csv` | SYNTHETIC | RETAIN in audit (forensic) | Aggregated from synthetic arm |
| `audit/lightgbm_robustness/artifact_hashes.sha256` | MIXED | RETAIN (historical) | Valid for files that existed at time of generation; does not cover closeout docs |
| `audit/p4_ems_readiness/REPORT.md` | MIXED | RETAIN + NOTICE header | EMS_GAP_SUPPORTED and EMS_READY_FOR_TARGETED_REWRITE remain valid; LIGHTGBM_ROBUSTNESS_CONFIRMED sub-verdict invalidated |
| `audit/p4_ems_readiness/build_integrated_report.py` | MIXED | RETAIN (historical) | References LIGHTGBM_ROBUSTNESS_CONFIRMED as input; kept for traceability |
| `audit/p4_ems_readiness/claims_allowed.md` | MIXED | RETAIN + verify contents | Claims referencing 714 cells or LightGBM robustness must be read as historical |
| `audit/p4_ems_readiness/claims_prohibited.md` | DOCUMENTATION_ONLY | RETAIN | Valid constraints on claim scope |

## Classification legend

- **EMPIRICAL_VALID**: Safe to use as evidence in Paper A
- **SYNTHETIC**: Generated from synthetic LightGBM arm; not empirical evidence
- **MIXED**: Contains both valid and contaminated elements; use with care
- **DOCUMENTATION_ONLY**: Metadata or process documentation; not used as data evidence
- **OBSOLETE_CLAIM**: Contains specific scientific claims that are now invalid
