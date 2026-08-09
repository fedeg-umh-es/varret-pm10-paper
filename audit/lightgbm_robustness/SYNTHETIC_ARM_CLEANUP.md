# SYNTHETIC ARM CLEANUP REPORT — Paper A / P4

**Date:** 2026-08-09
**Branch:** codex/p4-lightgbm-ems-gap-audit
**Initial HEAD:** 679e7e7953f2a42e8f5f4781cf74f11c5dd3a92d
**Final HEAD:** (see commit below)

---

## 1. Initial state

- Branch: `codex/p4-lightgbm-ems-gap-audit`
- HEAD: `679e7e7` (audit(p4): close LightGBM arm — SYNTHETIC_ARM verdict)
- Previous closeout: `LIGHTGBM_CLOSEOUT.md` confirmed that `lightgbm_direct`
  was never trained; 119 cells were arithmetic interpolations of HGB/Ridge.
- `LIGHTGBM_ROBUSTNESS_CONFIRMED` had been declared invalid.
- Contaminated table `master_diagnostic_table_with_lightgbm.csv` still lived in `outputs/analysis/`.
- Generator `generate_lightgbm_arm.py` still lived at root of audit dir.

---

## 2. Synthetic artefacts identified

| Artefact | Type |
|---|---|
| `outputs/analysis/master_diagnostic_table_with_lightgbm.csv` (714 rows) | PRIMARY SYNTHETIC |
| `audit/lightgbm_robustness/generate_lightgbm_arm.py` | SYNTHETIC GENERATOR |
| `audit/lightgbm_robustness/lightgbm_comparison_models.csv` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/lightgbm_discordant_cases.csv` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/model_selection_before_after.csv` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/model_selection_reversals.csv` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/pareto_fronts.csv` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/pareto_summary.json` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/model_selection_summary.json` | DERIVED SYNTHETIC |
| `audit/lightgbm_robustness/leakage_report.json` | IRRELEVANT (no real training) |
| `audit/lightgbm_robustness/REPORT.md` | OBSOLETE CLAIM document |
| `audit/p4_ems_readiness/REPORT.md` | MIXED (WP-A sub-verdict invalid) |

---

## 3. Actions performed

| Action | Target | Reason |
|---|---|---|
| `git mv` to quarantine | `generate_lightgbm_arm.py` → `legacy_synthetic/generate_synthetic_lightgbm_arm_DEPRECATED.py` | Neutralize accidental re-execution |
| `git mv` to quarantine | `outputs/analysis/master_diagnostic_table_with_lightgbm.csv` → `synthetic_outputs/master_diagnostic_table_with_lightgbm_SYNTHETIC.csv` | Remove synthetic table from outputs tree |
| STATUS header prepended | `audit/lightgbm_robustness/REPORT.md` | Mark LIGHTGBM_ROBUSTNESS_CONFIRMED as INVALID |
| NOTICE header prepended | `audit/p4_ems_readiness/REPORT.md` | Mark WP-A sub-verdict as invalid; retain WP-B |
| `execution_status` field added | `preregistered_protocol.json` | Mark as PREREGISTERED_BUT_NOT_EXECUTED |
| Created | `SYNTHETIC_ARM_INVENTORY.md` | Classify all 26 affected artefacts |
| Created | `EMPIRICAL_EVIDENCE_CONTRACT.md` | Canonical evidence scope definition |
| Created | `legacy_synthetic/SYNTHETIC_README.md` | Quarantine directory documentation |
| Created | `synthetic_outputs/SYNTHETIC_DATA_README.md` | Synthetic data quarantine documentation |
| Created | `verify_canonical_integrity.py` | Machine-verifiable integrity test suite |
| Executed | `verify_canonical_integrity.py` | 16/16 PASS |
| Created | `canonical_integrity_checks.json` | Machine-readable test results |

---

## 4. Canonical empirical evidence

```
outputs/tables/master_diagnostic_table.csv
SHA-256: 6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6
595 rows | 5 models | 17 stations | 7 horizons | 0 duplicate keys
lightgbm_direct rows: 0
```

---

## 5. Claims corrected

| Location | Original claim | Status |
|---|---|---|
| `audit/lightgbm_robustness/REPORT.md` | `LIGHTGBM_ROBUSTNESS_CONFIRMED` | SUPERSEDED header added |
| `audit/p4_ems_readiness/REPORT.md` | WP-A: `LIGHTGBM_ROBUSTNESS_CONFIRMED` | NOTICE header added |
| `preregistered_protocol.json` | Implicit: protocol was executed | `execution_status: PREREGISTERED_BUT_NOT_EXECUTED` added |

No claims in `scripts/`, README, or manuscript files were found referencing LightGBM robustness.

---

## 6. Tests

| Check | Result |
|---|---|
| canonical_rows_595 | PASS |
| lightgbm_rows_in_canonical_0 | PASS |
| canonical_models_5 | PASS |
| stations_17 | PASS |
| horizons_7 | PASS |
| duplicate_keys_0 | PASS |
| canonical_sha256 | PASS |
| rule_a_277 | PASS |
| rule_b_8 | PASS |
| changes_269 | PASS |
| pct_97.1% | PASS |
| rho_alpha_skill_-0.863 | PASS |
| discordant_101 | PASS |
| synthetic_moved_to_quarantine | PASS |
| synthetic_NOT_in_outputs_analysis | PASS |
| no_canonical_script_loads_synthetic | PASS |

**16/16 PASS — 0 FAIL — 0 BLOCKED**

---

## 7. Remaining synthetic artefacts and why retained

All remaining synthetic artefacts in `audit/lightgbm_robustness/` are retained for forensic traceability of the incident:

- `legacy_synthetic/` — original generator (neutralized, needed to reconstruct what happened)
- `synthetic_outputs/` — 714-row synthetic table (forensic only, clearly labeled)
- `lightgbm_*.csv`, `pareto_fronts.csv`, `model_selection_*.csv`, `pareto_summary.json`, `model_selection_summary.json`, `leakage_report.json` — derived synthetic artefacts in audit directory; cannot be discovered by default analysis scripts

None of these artefacts are in `outputs/tables/` or `outputs/analysis/`.

---

## 8. Material blockers

**None.**

All core results (277/8/269/97.1%/rho=-0.863/101) verified exclusively from 595 empirical rows.
No canonical analysis script loads the synthetic table.
No synthetic data remains in the `outputs/` tree.

---

## 9. Final verdict

```
SYNTHETIC_ARM_CLEANED
```

---

## 10. Paper readiness

```
READY_FOR_PAPER_A = YES
```

Conditions satisfied:
- [x] 595-row canonical table is the only empirical source (`outputs/tables/master_diagnostic_table.csv`)
- [x] LightGBM excluded from empirical evidence (moved to quarantine + contract documented)
- [x] No analysis script defaults to the 119-row synthetic table
- [x] All invalid claims marked SUPERSEDED or NOTICE headers added
- [x] All 6 core results trace exclusively to the 595 empirical rows (16/16 PASS)

Next allowed action: **REWRITE_FOR_EMS**
