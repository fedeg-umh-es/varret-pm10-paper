# WP-B — Search Log: EMS Error-Fidelity-Selection Gap Audit

## Objective

Map the corpus of Environmental Modelling & Software (EMS) 2015-2026 for papers
that address the joint treatment of:
- **A** forecast error criterion (skill, RMSE, DM significance)
- **B** dynamic forecast fidelity (variance retention, amplitude, oversmoothing)
- **C** recall of extreme events
- **D** joint model selection decision rule using A ∧ B ∧ C

## Search Protocol

- **Date:** 2026-08-06
- **Engine:** WebSearch (Claude proxy)
- **Primary scope:** Environmental Modelling & Software (ISSN 1364-8152)
- **Secondary scope:** Adjacent journals (HESS, Nature Communications, Hydrol. Process., Atmos. Environ.)
- **Period:** 2015-2026
- **Queries:** 23 total across 9 thematic dimensions
- **Scite API:** UNAVAILABLE (quota exhausted in prior session)
- **Elicit API:** UNAVAILABLE (requires paid subscription)

## Thematic Dimensions Searched

| Dim | Topic | Queries |
|-----|-------|---------|
| D1 | Model selection + RMSE/MAE + multiple horizons | 1, 14 |
| D2 | Variance retention / variance ratio in forecasts | 2, 11 |
| D3 | Air quality ML forecast comparison in EMS | 3, 6 |
| D4 | Diebold-Mariano test in environmental forecasting | 5, 23 |
| D5 | Rolling-origin validation reversal | 10, 12 |
| D6 | Oversmoothing / variance underestimation | 7 |
| D7 | Extreme events + recall + model selection | 8, 22 |
| D8 | NSE/KGE/skill score critique | 13, 21 |
| D9 | Joint / multi-criteria evaluation | 9, 15, 16, 17, 18, 19, 20 |

## Key Papers Found

### In EMS (primary target journal)

| Paper | Year | DOI | Relevance to Gap |
|-------|------|-----|-----------------|
| Williams 2025 — NSE/KGE critique | 2025 | 10.1016/j.envsoft.2025.106665 | INDIRECT — argues for error-only, addresses error-vs-variability debate |
| Comment on Williams 2025 | 2026 | 10.1016/j.envsoft.2026.000162* | INDIRECT — defends variability metrics, no operational joint rule |

*DOI estimated from PII S1364815226000162.

### In Adjacent Journals

| Paper | Year | Journal | DOI/URL | Relevance |
|-------|------|---------|---------|-----------|
| MFM — Model Fidelity Metric | 2026 | HESS | https://hess.copernicus.org/articles/30/2651/2026/ | HIGH — proposes combined error+variability metric (hydrology) |
| Standard assessments misleading | 2021 | Nat. Comms. | 10.1038/s41467-021-23771-z | MEDIUM — model ranking sensitivity to criteria (climate) |
| Rolling-Origin Reverses Rankings | 2026 | arXiv (preprint) | arXiv:2603.20315 | VERY HIGH — same protocol, PM10, but no joint fidelity criteria |
| GCMeval multi-criteria | 2020 | Clim. Services | 10.1016/j.cliser.2020.100167 | LOW — climate model selection, not time series forecast fidelity |

## Gap Coverage Assessment

The search found **NO EMS paper** that:
1. Uses variance retention (alpha = Var(y_pred)/Var(y_true)) as a model selection criterion
2. Uses recall of extremes (recall_p75) jointly with error criteria for model acceptance decisions
3. Quantifies the rate of model selection reversals when applying joint error+fidelity criteria
4. Studies this in a multi-station (≥10), multi-horizon air quality forecasting context

The Williams 2025 debate covers the error-vs-variability dimension at the level of metric choice
philosophy but does NOT operationalize a joint rule or quantify decision-change rates.

The MFM paper (HESS 2026) is the closest work but is:
- Outside EMS (different journal)
- Applied to land surface hydrology, not air quality time series
- Focused on model calibration evaluation, not operational forecast model selection

## Limitations

1. **Corpus completeness:** Web search cannot guarantee exhaustive coverage of all EMS papers.
   A Scopus/Web of Science systematic search with full-text access would be definitive.
2. **Paywall:** Many EMS papers were not accessible for full-text screening; abstract-level
   screening only for paywalled items.
3. **Language:** English-only queries.
4. **Scite/Elicit unavailable:** Would have enabled broader structured search.
