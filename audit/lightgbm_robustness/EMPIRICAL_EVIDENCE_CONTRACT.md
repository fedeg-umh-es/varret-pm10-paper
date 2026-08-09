# EMPIRICAL EVIDENCE CONTRACT — Paper A / P4

**Effective date:** 2026-08-09
**Status:** CANONICAL

---

## Canonical empirical table

```
outputs/tables/master_diagnostic_table.csv
```

**SHA-256:** `6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6`

---

## Empirical scope

- **595 rows** (station × model × horizon)
- **5 empirical models:** `hgb_direct`, `ridge_direct`, `sarima`, `seasonal_naive`, `stl_ridge_direct`
- **17 stations:** MITECO/Cataluña/Valencia/Andalucía PM10 network
- **7 horizons:** h = 1, 2, 3, 4, 5, 6, 7 days ahead
- **0 duplicate keys** (station_id × model × horizon)

---

## Verified core results (595-row traceable)

| Result | Value | Status |
|---|---|---|
| Rule A (skill > 0 AND dm_significant) | 277 cells | VERIFIED_FROM_595 |
| Rule B (Rule A AND alpha >= 0.50 AND recall_p75 >= 0.20) | 8 cells | VERIFIED_FROM_595 |
| Decision changes (Rule A - Rule B) | 269 cells | VERIFIED_FROM_595 |
| Decision change proportion | 97.1% | VERIFIED_FROM_595 |
| Spearman rho(alpha, skill) | -0.863 (p = 5.4e-178) | VERIFIED_FROM_595 |
| Discordant cases (Rule A AND recall >= 0.20 AND alpha < 0.50) | 101 cells | VERIFIED_FROM_595 |

---

## Excluded evidence

All `lightgbm_direct` rows are excluded from empirical evidence:

- 119 rows in `audit/lightgbm_robustness/synthetic_outputs/master_diagnostic_table_with_lightgbm_SYNTHETIC.csv`
- All metrics in `audit/lightgbm_robustness/lightgbm_*.csv`
- All aggregated results in `audit/lightgbm_robustness/*.json` that include the synthetic arm
- All model selection and Pareto results derived from the 714-row table

---

## Historical artifacts retained (not to be used as scientific evidence)

| Artifact | Retained for |
|---|---|
| `preregistered_protocol.json` | Protocol design traceability; status: PREREGISTERED_BUT_NOT_EXECUTED |
| `legacy_synthetic/generate_synthetic_lightgbm_arm_DEPRECATED.py` | Forensic traceability of the incident |
| `synthetic_outputs/master_diagnostic_table_with_lightgbm_SYNTHETIC.csv` | Forensic traceability |
| `REPORT.md` | Historical document; marked SUPERSEDED |
| All other `audit/lightgbm_robustness/` CSVs and JSONs | Forensic record |

---

## Scientific status

```
LIGHTGBM_NOT_EXECUTED
LIGHTGBM_NOT_PART_OF_EMPIRICAL_EVIDENCE
EMS_GAP_SUPPORTED (WP-B, 595-row derived)
EMS_READY_FOR_TARGETED_REWRITE
```

---

## Allowed manuscript claim

> "A LightGBM robustness protocol was preregistered but was not executed.
> A subsequently generated synthetic arm was identified during closeout
> and is excluded from empirical evidence."

---

## Prohibited manuscript claims

1. "LightGBM confirmed robustness."
2. "LightGBM reproduced the pattern across 119 cells."
3. "LightGBM provided independent empirical evaluations."
4. "LightGBM generalized the result beyond HGB."
5. Any claim citing the 714-row table as empirical evidence.

---

*This contract supersedes all LightGBM robustness claims in previous audit reports.*
