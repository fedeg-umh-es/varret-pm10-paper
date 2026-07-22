# Editorial Positioning Decision

## Update (2026-07-22): predictability-bound line promoted to standalone manuscript

The E1-RR variance-retention diagnostic (Skill$_{VP}$, single dataset) remains a
**post-evaluation module supporting P33**, per the original decision below,
unchanged.

A related but distinct line of work — verifying whether the horizon at which
persistence-relative skill remains positive ($H_{\max}$) is being
misreported as the predictability horizon ($\hstar$) — has been extended
beyond what the "Conditions for Becoming a Standalone Paper" section below
anticipated, and is promoted to an independent manuscript:
`paper_b_predictability_bound.tex`.

Conditions now met, against the list below:
- additional PM10 stations: 3 (Elx-Agroalimentari, Valencia-Vivers,
  Zarra-EMEP), spanning industrial/suburban, urban, and rural-remote site
  types, instead of the single E1-RR daily dataset;
- multiple model families beyond the minimal lags-only setup: `hgb_direct`,
  `ridge_direct`, and sparse-origin `sarima` compared as candidates, with
  `seasonal_naive`/`stl_ridge_direct` retained as weak references;
- a formal theoretical construction replacing `Skill_VP`/`alpha` for this
  line specifically: a Yule-Walker linear-predictability ceiling ($p=14$
  lags), algebraic on the autocorrelation function, with two weaker
  alternatives (AR(1), hybrid) reported for comparison and explicitly
  rejected;
- robustness across the evaluation window: block bootstrap (5,000 resamples)
  on every cell where point-estimate skill exceeded the linear ceiling (8 of
  21 station-horizon cells); none is statistically robust.

This module is **not** the E1-RR variance-retention/Skill$_{VP}$ story and
does not claim variance collapse, ghost skill in the Skill$_{VP}$ sense, or
anything about E2-MET/E3-PROB. It is a distinct claim: that a positive skill
value at $H_{\max}$ is a right-censored lower bound on $\hstar$, not a point
estimate, and that this specific finding (elx-agroalimentari, valencia-vivers,
zarra-emep, $h\le7$) is consistent with the autocorrelation-implied linear
ceiling rather than anomalous relative to it.

The claim guardrails below apply, unmodified, to the E1-RR/Skill$_{VP}$ module.
`paper_b_predictability_bound.tex` has its own explicit Limitations section
and should not be cited as evidence for claims outside its own scope (three
stations, $h\le7$, linear predictability ceiling only — not a bound on
non-linear or exogenous-covariate models).

---

## Original decision (E1-RR variance-retention / Skill$_{VP}$ module)

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
