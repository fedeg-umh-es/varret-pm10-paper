# Rule-B threshold sensitivity

Source: existing `audit/decision_change/rule_b_sensitivity.csv`, verified against the frozen 595-row canonical table.

Primary thresholds: alpha = 0.50 and P75 recall = 0.20. Primary result: Rule A = 277, Rule B = 8, changes = 269, change proportion = 97.1%.

The displayed neighbourhood uses alpha thresholds {0.40, 0.50, 0.60} and P75-recall thresholds {0.10, 0.20, 0.30}. Across these nine combinations, the proportion of Rule-A cells changing ranges from 91.3% to 98.2%; retained Rule-B cells range from 24 to 5. The qualitative conclusion that fidelity materially changes eligibility persists, so the sensitivity verdict is **ROBUST** within this deterministic neighbourhood.

This is a post-evaluation audit of frozen rows, not a forecasting experiment, threshold optimisation, or claim of universal threshold validity.
