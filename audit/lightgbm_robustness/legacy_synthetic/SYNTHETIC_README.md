# SYNTHETIC ARM — LEGACY ARCHIVE

**STATUS: SUPERSEDED — SYNTHETIC ARM, NOT EMPIRICAL EVIDENCE**

This directory contains the legacy artefacts of the LightGBM robustness arm
that was preregistered but never executed with real model training.

## What happened

`generate_synthetic_lightgbm_arm_DEPRECATED.py` was initially labeled as
a LightGBM arm generator. During closeout audit on 2026-08-09, it was
confirmed that the script generates all 119 metric cells by deterministic
arithmetic interpolation of `hgb_direct` and `ridge_direct` values,
with max deviation from the interpolation formula of 8.3e-17 (machine precision).

No LightGBM model was trained. No PM10 data was processed by LightGBM.

## Files retained for traceability

- `generate_synthetic_lightgbm_arm_DEPRECATED.py`: the original generator (do not run)

## Do not use these artefacts as scientific evidence

See `audit/lightgbm_robustness/LIGHTGBM_CLOSEOUT.md` for full audit.
See `audit/lightgbm_robustness/EMPIRICAL_EVIDENCE_CONTRACT.md` for canonical evidence scope.
