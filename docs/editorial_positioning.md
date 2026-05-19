# Editorial Positioning Decision

## Current Decision

This analysis is currently treated as a **post-evaluation module supporting P33**, not as an independent manuscript.

The current evidence is methodologically useful because it shows that positive persistence-relative skill can coexist with low forecast variance retention. However, the present scope remains too narrow to justify opening a standalone paper front.

## Current Role in the Research Line

The variance-retention analysis should be used as:

- a diagnostic extension of the E1-RR daily lags-only evaluation;
- a reproducible support artifact for the P33 variance-retention / ghost-skill story;
- a bounded methodological result showing that RMSE-based skill can overstate operational credibility when forecasts are overly smoothed.

It should not currently be framed as:

- a new standalone manuscript;
- a general theory of forecast skill;
- a universal replacement for standard skill scores;
- evidence about E2-MET or E3-PROB;
- evidence about meteorological marginal value or probabilistic forecast reliability.

## Rationale

The current analysis is based on:

- one PM10 daily dataset;
- one E1-RR daily lags-only setup;
- persistence as baseline;
- two direct lags-only models (`ridge_direct` and `hgb_direct`);
- horizon-wise diagnostics for `h = 1,...,7`;
- variance retention, `alpha`, and `Skill_VP` as auxiliary diagnostics.

This is sufficient for a reproducible post-evaluation module, but not yet sufficient for a standalone manuscript without risking fragmentation of the research pipeline.

## Conditions for Becoming a Standalone Paper

This module may become a standalone paper only if it is extended beyond the current single-dataset E1-RR daily lags-only case. Minimum extensions would include some combination of:

- additional PM10 stations or datasets;
- additional pollutants or sites;
- multiple model families beyond the current minimal lags-only setup;
- systematic comparison between raw skill ranking and variance-adjusted diagnostic ranking;
- sensitivity analysis of the `alpha` thresholds;
- clearer formalization of `Skill_VP` as an auxiliary diagnostic rather than an accuracy metric;
- robustness across evaluation windows;
- explicit comparison with the original E1-RR result package.

## Claim Guardrails

Acceptable wording:

> The post-evaluation indicates that positive persistence-relative skill may coexist with low variance retention, suggesting that RMSE-based improvements should be interpreted cautiously when forecasts are highly smoothed.

Avoid wording such as:

> Skill_VP proves ghost skill.

> The models are invalid.

> There is no predictability.

> Skill_VP should replace standard forecast skill metrics.

> This result generalizes to E2-MET, E3-PROB, or meteorological forecasting more broadly.

## Operational Rule

Until the module is explicitly expanded, the working label is:

> **E1-RR variance-retention post-evaluation module supporting P33.**

Not:

> **new independent paper.**
