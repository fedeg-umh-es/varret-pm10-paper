# Claims Prohibited — Paper A (P4 Audit)

These claims are NOT supported by the current evidence base and must not appear
in the manuscript or any public communication about Paper A.

## Category 1: LightGBM Claims (WP-A BLOCKED)

- "LightGBM confirms the collapse phenomenon generalizes to gradient boosting."
- "LightGBM shows robustness of the alpha criterion across model families."
- "The 119 LightGBM cells would have shown higher concordance."
- "The result holds for 714 cells (5 models + LightGBM)."
- Any quantitative claim about LightGBM rule_a, rule_b, alpha, or recall values.

**Reason:** WP-A verdict BLOCKED_BY_PIPELINE_OR_DATA. LightGBM was not executed.
Config pre-registered but no predictions generated.

## Category 2: Corpus Completeness Claims (WP-B PARTIAL)

- "EMS has NEVER addressed forecast fidelity." — FALSE: Williams 2025 + Comment 2026 in EMS.
- "No paper in the literature combines error and variability metrics." — FALSE: MFM (HESS 2026).
- "Paper A is the first to use rolling-origin in PM10 forecasting." — FALSE: arXiv:2603.20315.
- "The gap is confirmed by a systematic Scopus/WoS review." — FALSE: web search only.

**Reason:** These claims overstate what WP-B found. WP-B is PARTIALLY supported,
not exhaustively confirmed.

## Category 3: Causal/Mechanistic Claims (beyond evidence)

- "Models with positive skill are operationally useful for decision support."
- "Variance collapse is caused by [specific mechanism]."
- "The 97.1% rate proves the alpha criterion is universally necessary."
- "Forecasters using only RMSE/skill will select operationally inferior models."

**Reason:** The evidence demonstrates an empirical association (high decision-change rate);
it does not establish causation or generalizability beyond the observed corpus.

## Category 4: Claims Requiring Unrealized Experiments

- "LightGBM confirms the findings from HGB and Ridge."
- "All tree-based boosting models collapse in the same way."
- "The result is robust to model architecture (boosting vs. linear)."

**Reason:** These require the LightGBM arm which is blocked.

## Category 5: Claims About Other Research Lines

- Any claim about P1 (Pharmacology/PLS-SEM), SCADA, WUE, LLM regulatory, or water balance.
- Any cross-paper synthesis not grounded in the 595 PM10 cells.

**Reason:** Out of scope per superprompt constraints.
