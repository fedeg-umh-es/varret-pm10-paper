# P4 EMS final scope-gate build report

## Canonical evidence

- Integrity: 16/16 PASS.
- Source: frozen `master_diagnostic_table.csv`.
- Rows/models/stations/horizons: 595 / 5 / 17 / 7.
- LightGBM empirical rows: 0.
- Primary eligibility: Rule A 277, Rule B 8, changes 269, 97.1%.
- Formal discordance: 101 (54 HGB, 43 Ridge, 4 SARIMA).

## Scope and production controls

No models, forecasts, stations, datasets, folds, canonical thresholds, or
new literature searches were added. The only additional analysis remains
the deterministic Rule-B sensitivity from the existing canonical audit
artifact. The 486-abstract reporting block and the AI-assisted graphical
abstract draft are excluded from the v3 submission archive.

## Final validation

- Abstract: 90 words under the repository's LaTeX-stripped word count.
- LaTeX compilation: PASS, 10 pages, no errors, undefined citations,
  undefined references, missing figures, or missing tables.
- Visual QA: PASS; `FIGURE_2_VISUAL_QA = PASS`.
- Overleaf v3 archive validation: PASS (`unzip -t`).
- Independent Overleaf v3 compilation: PASS, 10 pages.

Author actions remain: manually create the graphical abstract, synchronise
the public repository with the final submission version, and obtain final
author approval.
