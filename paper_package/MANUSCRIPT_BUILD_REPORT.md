# P4 result-set-first narrative rebuild report

## Canonical evidence

- Integrity: 16/16 PASS.
- Source: frozen `master_diagnostic_table.csv`.
- Rows/models/stations/horizons: 595 / 5 / 17 / 7.
- LightGBM empirical rows: 0.
- Primary eligibility: Rule A 277, Rule B 8, changes 269, 97.1%.
- Formal discordance: 101 (54 HGB, 43 Ridge, 4 SARIMA).

## Scope controls

No models, forecasts, stations, datasets, folds, canonical thresholds, or
new literature searches were added. The result set was assembled and
visually inspected before narrative editing. The 486-abstract reporting
block and the AI-assisted graphical abstract draft are excluded from the
v4 submission archive.

## Final validation

- Abstract: 138 words under the repository's LaTeX-stripped word count.
- Result-set freeze: PASS; candidate Figure 2 designs were inspected
  before narrative editing.
- LaTeX compilation: PASS, 10 pages, no errors, undefined citations,
  undefined references, missing figures, or missing tables.
- Visual QA: PASS; `FIGURE_2_VISUAL_QA = PASS`.
- PIER audit: PASS.
- Narrative SKIP test: PASS.
- Overleaf v4 archive validation: PASS (`unzip -t`).
- Independent Overleaf v4 compilation: PASS, 10 pages.

Author prose review remains required; no push is performed by this task.
