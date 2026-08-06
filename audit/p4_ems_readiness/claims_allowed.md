# Claims Allowed — Paper A (P4 Audit)

These claims are supported by the verified 595-cell evidence base
and the WP-B corpus audit. Each claim must be traceable to the
canonical `master_diagnostic_table.csv` (SHA-256 verified).

## Tier 1: Core Empirical Claims (directly supported by 595 cells)

**C-A1.** In a rolling-origin evaluation of PM10 daily forecasting across 17 stations
and 7 horizons (595 model-horizon-station cells), 277/595 cells (46.6%) pass the
standard error criterion (Rule A: skill > 0 AND DM-HLN significant at 5% FDR-BH).

**C-A2.** Of the 277 Rule-A-passing cells, only 8 (2.9%) also satisfy the joint
operational criterion (Rule B: Rule A AND alpha ≥ 0.50 AND recall_p75 ≥ 0.20),
yielding a decision-change rate of 97.1% (269/277).

**C-A3.** 101 cells pass Rule A and recall_p75 ≥ 0.20 but fail alpha < 0.50
(discordant cases detectable only with the variance retention criterion).

**C-A4.** The dominant reason for decision change is variance collapse (alpha < 0.50)
alone or in combination with insufficient recall, not recall failure alone.

**C-A5.** Adding alpha to (skill + recall_p75) in a logistic model predicts decision
change among Rule-A passers with higher AUC than skill + recall_p75 alone
(incremental AUC from alpha confirmed by script 15 output).

**C-A6.** The 595-cell result is free of data leakage: expanding window is strictly
causal; threshold P75 estimated from training data only; no test-time parameter tuning.

## Tier 2: Gap Claims (supported by WP-B corpus audit, partially)

**C-B1.** Environmental Modelling & Software (2015-2026) does not contain articles
that operationalize variance retention (alpha = Var(ŷ)/Var(y)) AND recall of threshold
exceedances jointly as a model selection rule in air quality time series forecasting.

**C-B2.** The Williams (2025) EMS debate addresses the error-vs-variability metric
choice at a philosophical level without proposing or validating a joint operational
selection rule.

**C-B3.** The Model Fidelity Metric (MFM, HESS 2026) proposes a combined error+variability
metric in hydrology but does not address air quality time series or quantify model
selection decision-change rates.

## Tier 3: Qualified Claims (require caveats)

**C-Q1.** The 97.1% decision-change rate applies to the specific thresholds
(alpha ≥ 0.50, recall_p75 ≥ 0.20) and the specific corpus (17 Spanish PM10 stations
2018-2024). Sensitivity analysis across a 3×3 threshold grid shows the rate ranges
from [sensitivity_min]% to [sensitivity_max]% — specific values from
`decision_sensitivity_matrix.csv`.

**C-Q2.** The gap claim (C-B1) is based on a web-search corpus, not a full
Scopus/Web of Science systematic review. The probability of a missed EMS paper with
the exact alpha + recall joint rule is assessed as LOW given the specificity of
the construct, but is not zero.

**C-Q3.** The LightGBM robustness arm was pre-registered but could not be executed
due to missing station data (14/17 stations absent from `data/raw/`). The WP-A verdict
BLOCKED_BY_PIPELINE_OR_DATA means the LightGBM claim cannot be made.
