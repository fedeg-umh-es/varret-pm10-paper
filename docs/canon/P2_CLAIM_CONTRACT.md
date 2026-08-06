# P2_CLAIM_CONTRACT

**Version:** `1.0`
**Date:** 2026-08-06
**Binds:** `docs/canon/P2_PROJECT_CANON.md` v2.0
**Scope:** every claim P2 may make, in any manuscript, abstract, figure caption,
release note or talk, before and after the empirical gate.

A claim not listed here is not authorised. Claims are stated in the exact form
in which they may be written.

---

## C0. Allowed — principal methodological claim

> We propose a paired rolling-origin evaluation that decomposes the loss
> reduction relative to persistence into a contribution associated with one-lag
> dependence, a contribution associated with additional finite linear memory,
> and a model-specific residual gain.

**Status:** allowed now. It describes a construction, not a finding.

**Conditions:**

- The decomposition must be shown on the MSE scale and the additive identity
  numerically verified (`MSE_IDENTITY_VERIFIED`).
- "Paired" may be written only where `origin_date`, `target_date`, `y_true`,
  forecast availability and squared-error cases are shared across all compared
  methods on the reported support.
- The support and its `n_paired` must be stated wherever the claim appears with
  numbers.

---

## C1. Allowed — methodological gap claim (scoped)

> Within the reviewed corpus of 20 works, no evaluation was identified that
> integrates persistence, AR(1), AR(p)/Yule–Walker estimation, paired
> rolling-origin comparison, and explicit treatment of the incomplete calendar,
> in order to attribute predictive improvement to finite linear memory and
> residual gain.

**Status:** allowed **only in this scoped form**.

**Conditions:**

- The scope qualifier ("within the reviewed corpus of 20 works") is mandatory
  and may not be dropped, softened to "in the literature", or upgraded to "no
  study has".
- The 20-work corpus must be enumerated in the manuscript or its supplement.
- The claim is a statement about a bounded review, not about the literature.

**Strengthening step (not a blocker):** a systematic Scopus / Web of Science
search would let the scope qualifier be widened. Suggested query:

```
("Yule-Walker" OR "AR(p)") AND ("forecast skill" OR "predictive skill")
AND ("persistence" OR "baseline")
```

filtered to Environmental Science + Atmospheric Science, last 15 years; plus
`"linear predictability" AND "persistence" AND "air pollution"`. An empty or
irrelevant return makes the gap traceable and citable. Until that search is run
and archived in the repository, **only the scoped form is authorised**.

---

## C2. Pending — empirical claim

> The relative importance of the components varies by station and by horizon.

**Status:** **not yet demonstrated.** May not be written until produced by the
paired rolling-origin decomposition tables under
`docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md`, with block-bootstrap
uncertainty.

**Release conditions:** `P_SENSITIVITY_COMPLETED`, `BLOCK_BOOTSTRAP_COMPLETED`,
`RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION`. Variation must be visible
beyond bootstrap uncertainty, not merely present in point estimates.

---

## C3. Allowed — interpretive claim

> Beating persistence does not by itself demonstrate non-linear structure or
> exogenous information, because part of the gain may come from exploiting
> temporal dependence more effectively.

**Status:** allowed as a framing thesis.

**Condition — novelty tone.** This is a logical consequence already implied in
the classical meteorological literature on the choice of reference forecast for
skill scores. In the Discussion it must be presented as an **empirical result in
this domain**, not as a new epistemological insight. Do not write "we show for
the first time that…" about the epistemology; the first-time element is the
PM10 measurement, not the logic.

---

## C4. Prohibited

> ~~The residual gain demonstrates non-linearity or exogenous information.~~

Also prohibited, as variants of the same error:

- Calling `Δ_res` "non-linear skill", "non-linear component", or "information
  gain".
- Reading `Δ_res > 0` as evidence of physical predictability.
- Presenting `L_ARp` as a bound, ceiling, limit, or "linear predictability
  limit". It is conditional on stationarity, loss function, order `p`, available
  information, and the autocovariance estimator.
- Presenting `H*` as a theoretical predictability bound rather than an
  operational criterion of relative usefulness against a benchmark.

`Δ_res` is a **residual**: what a given model achieved beyond a given finite
linear reference, on a given support. Nothing more.

---

## C5. Prior-work attribution

The nearest antecedent identified is a rolling-origin PM10 study comparing
persistence, SARIMA and XGBoost, sharing terminology with this programme
(`H*`, rolling-origin, persistence-relative skill; arXiv 2603.20315).

**Requirement — resolve authorship before submission.** If it belongs to this
research programme, it must be cited as a **direct antecedent of the programme
itself**, not as generic external literature; self-citation must be transparent
and the relationship stated in the text. If it does not, it must be cited as the
closest external antecedent and motivation.

**Status flag:** `VERIFY_AUTHORSHIP` — open. This flag must be cleared before
any submission. Until cleared, the paper may not describe that work either as
"our previous work" or as "independent prior work".

**Not verified here:** the bibliographic details of that work and of the other
cited items were supplied by an external search tool and have not been checked
against the sources in this repository. Verification is required before they
enter `references.bib`.

---

## C6. Enforcement

Any manuscript text implementing a C2-class statement before its release
conditions are met, or any instance of a C4 formulation, is a **gate failure**
and blocks `P2_PAPER_GO` regardless of the state of the other conditions.
