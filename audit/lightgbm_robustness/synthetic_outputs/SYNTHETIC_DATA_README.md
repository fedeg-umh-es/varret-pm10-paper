# SYNTHETIC DATA — DO NOT USE AS EMPIRICAL EVIDENCE

**STATUS: SUPERSEDED — SYNTHETIC ARM, NOT EMPIRICAL EVIDENCE**

## File: master_diagnostic_table_with_lightgbm_SYNTHETIC.csv

This file was produced by `generate_synthetic_lightgbm_arm_DEPRECATED.py`
and contains 714 rows: 595 real empirical rows + 119 synthetic rows for
`lightgbm_direct`.

The 119 `lightgbm_direct` rows are deterministic arithmetic interpolations
of `hgb_direct` and `ridge_direct` values. They are NOT the output of a
trained LightGBM model.

## Canonical empirical table

The only valid empirical table is:

    outputs/tables/master_diagnostic_table.csv
    595 rows | 5 empirical models | 17 stations | 7 horizons
    SHA-256: 6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6

## Retained for

Forensic traceability of the synthetic arm incident only.
