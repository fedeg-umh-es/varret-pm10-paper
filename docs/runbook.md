# Runbook

The active manuscript source is Overleaf. This repository provides the
reproducible code, tables, figures, and audit trail that support the manuscript.

## Canonical workflow

The closing empirical package is the all-stations PM10 diagnostic. The
single-station E1-RR workflow remains available for provenance, but the final
tables and figures are generated from the canonical all-stations outputs.

1. Confirm station inventory and selected MITECO stations:
   `data/miteco_inventory.csv`, `data/miteco_selected.csv`.
2. Generate or refresh rolling-origin predictions with past-only folds.
3. Build persistence-relative skill summaries:
   `python3 scripts/06_build_skill_tables.py`.
4. Build variance-retention diagnostics:
   `python3 scripts/07_build_variance_retention_table.py`.
5. Build unified station tables and diagnostics:
   `python3 scripts/build_unified_predictions_table.py`,
   `python3 scripts/build_unified_variance_table.py`,
   `python3 scripts/build_unified_dm_table.py`,
   `python3 scripts/04_build_unified_exceedance_table.py`,
   `python3 scripts/09_build_comprehensive_unified_table.py`.
6. Build final manuscript tables:
   `python3 scripts/13_build_five_model_diagnostic_summary.py`,
   `python3 scripts/14_generate_skill_alpha_figure.py`.
7. Build final figures:
   `python3 scripts/10_generate_all_figures.py`.
8. Run `pytest` to validate the diagnostics layer before submission.

## Q1 closeout rules

- Keep the paper framed as a post-evaluation diagnostic, not as a new model.
- Keep `Skill(h)`, `alpha(h)`, and `Skill_VP(h)` conceptually separate.
- Use `Skill_VP` only as an auxiliary diagnostic summary.
- Keep event/exceedance and Murphy decomposition only if they directly support
  the operational relevance of variance collapse.
- Do not import H* claims from `paper2H` into this manuscript.
- Do not add model families solely to expand the leaderboard.
- Keep KGE outputs in the companion package; do not use them as the main paper's
  closing evidence.

## Artifact policy

- Canonical figures: `outputs/figures/`.
- Canonical tables: `outputs/tables/`.
- Canonical metrics/predictions: `outputs/metrics/`.
- Root-level figure/PDF copies are Overleaf convenience exports and should not
  be the only source for any manuscript claim.
- `scripts/p34_utils/` is legacy/companion material and is excluded from the
  main run order.
