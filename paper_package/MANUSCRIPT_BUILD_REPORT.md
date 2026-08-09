# P4 EMS pre-submission revision build report

## Canonical evidence

- Integrity: 16/16 PASS.
- Source: frozen `master_diagnostic_table.csv`.
- Rows/models/stations/horizons: 595 / 5 / 17 / 7.
- LightGBM empirical rows: 0.
- Primary eligibility: Rule A 277, Rule B 8, changes 269, 97.1%.
- Formal discordance: 101 (54 HGB, 43 Ridge, 4 SARIMA).

## Scope controls

No models, forecasts, stations, datasets, folds, or canonical thresholds
were changed. The only additional analysis is deterministic Rule-B
sensitivity from the existing canonical audit artifact. The 486-abstract
reporting block was removed from the Paper-A manuscript because it
introduces a separate provenance line.

## Production status

- Abstract: 90 words under the repository's LaTeX-stripped word count.
- LaTeX compilation: PASS, 10 pages, no errors, undefined citations,
  undefined references, missing figures, or missing tables.
- Visual QA: PASS; `FIGURE_2_VISUAL_QA = PASS`.
- Overleaf v2 archive validation: PASS (`unzip -t`).
- Independent Overleaf v2 compilation: PASS, 10 pages.

Current scientific verdict target:
`MANUSCRIPT_PACKAGE_READY_WITH_DOCUMENTED_LIMITATIONS`.
