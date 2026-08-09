# P4 EMS presentation V5.1 build report

## Canonical evidence

- Integrity: 16/16 PASS.
- Source: canonical `master_diagnostic_table.csv`.
- Rows/models/stations/horizons: 595 / 5 / 17 / 7.
- LightGBM empirical rows: 0.
- Primary eligibility: Rule A 277, Rule B 8, changes 269, 97.1%.
- Formal discordance: 101 (54 HGB, 43 Ridge, 4 SARIMA).

## Scope controls

No models, forecasts, stations, datasets, folds, canonical thresholds, or
new literature searches were added. The result set was assembled and
visually inspected before narrative editing. The 486-abstract reporting
block and the AI-assisted graphical abstract draft are excluded from the
v5 submission archive.

## Final validation

- Abstract: 139 lexical tokens in PDF text extraction, below the 150-word limit.
- Result-set freeze: PASS; figures and tables were inspected before the
  localized presentation edits.
- LaTeX compilation: PASS, 11 pages, no errors, undefined citations,
  undefined references, missing figures, or missing tables.
- Visual QA: PASS; `FIGURE_2_VISUAL_QA = PASS`, with no Results floats
  after the Discussion heading and no Supplementary/References interleaving.
- PIER audit: PASS; no global prose rewrite was performed.
- Narrative SKIP test: PASS.
- Overleaf v5.1 archive validation: PASS (`unzip -t`).
- Independent Overleaf v5.1 compilation: PASS, 11 pages.

Author prose review remains required; no push is performed by this task.
