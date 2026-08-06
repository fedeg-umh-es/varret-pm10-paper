# P2 Claim Contract

Version: 1.0  
Last updated: 2026-08-06  
Status: CANONICAL  
Parent canon: `P2_PROJECT_CANON.md` v2.0  
Decision ID: `2026-08-06-p2-finite-linear-memory-skill-decomposition`

---

## 1. Purpose

This contract controls what P2 may claim before and after the paired experiment. It separates methodological claims, bounded literature claims, empirical hypotheses and prohibited interpretations.

No manuscript wording overrides this file.

---

## 2. Permitted methodological claim

> We propose a paired rolling-origin evaluation that decomposes persistence-relative MSE reduction into a contribution associated with one-lag dependence, an additional contribution associated with finite linear memory, and a model-specific residual gain.

Conditions:

- losses are paired on identical cases;
- all reference parameters are estimated train-only;
- AR(1) and AR(p) are direct horizon-specific linear projections;
- attribution is performed in MSE;
- the additive identity is verified;
- results are reported separately by empirical model.

---

## 3. Permitted bounded literature claim

> In the traceable corpus of 20 works reviewed for P2, no evaluation was identified that jointly combined persistence, direct AR(1), direct finite-memory AR(p)/Yule–Walker, paired rolling-origin comparison and explicit preservation of an incomplete calendar to attribute PM10 forecast improvement between linear memory and model-specific residual gain.

Required qualifiers:

- this claim refers only to the documented corpus;
- it is not proof of absence from the complete literature;
- corpus search strings, databases, dates, inclusion decisions and included records must be retained;
- a systematic Scopus/Web of Science check is required before using an unbounded novelty formulation.

Forbidden replacement:

> This is the first method to decompose PM10 predictability.

---

## 4. Literature positioning that does not constitute the P2 novelty

The following are antecedents, not gaps:

1. Stronger naive references can reduce apparent forecast skill. Murphy (1992), *Weather and Forecasting*, formalised climatology, persistence and their optimal linear combination as alternative standards of reference.
2. AR(p)/Yule–Walker projection is standard time-series mathematics. P2 does not claim novelty for the equations.
3. Missing-observation time-series literature establishes that temporal spacing and the missingness treatment affect autocovariance and long-run variance estimation. This supports the calendar-preserving design but does not by itself implement P2.
4. García Crespí et al. (2026), arXiv:2603.20315, is a direct antecedent from the same research programme showing that rolling-origin validation can reverse PM10 model rankings against persistence. It must be cited as programme-internal prior work.

The novelty, if empirically supported, lies in the integrated diagnostic use of these elements for paired skill attribution in daily PM10 forecasting.

---

## 5. Empirical claim pending the gate

> The relative importance of one-lag dependence, additional finite linear memory and model-specific residual gain varies by station and forecast horizon.

Status:

```text
HYPOTHESIS — NOT YET AN EMPIRICAL RESULT
```

It becomes claimable only if:

- the result replicates across more than one station;
- paired support and train-only checks pass;
- the conclusion is not an artefact of lag order, missingness handling, oracle selection or bootstrap configuration;
- a non-trivial interpretation remains after uncertainty is considered.

---

## 6. Permitted consequence claim, with calibrated novelty

> Beating persistence does not by itself identify the source of predictive improvement, because part of the gain may be reproduced by a stronger finite-memory linear reference.

This is an established verification principle applied to the P2 design, not a wholly new epistemological discovery. Novelty language must concern the PM10 implementation and empirical attribution, not the general principle.

---

## 7. Prohibited claims

P2 must not claim that:

- residual gain proves nonlinearity;
- residual gain proves exogenous information;
- residual gain identifies a causal mechanism;
- AR(p) is a universal predictability ceiling;
- finite linear memory is the only source of persistence-relative skill;
- a model is generally superior because one station–horizon cell has positive residual gain;
- normalised component fractions must lie in `[0,1]`;
- negative components should be truncated;
- a test-set best-model envelope is a legitimate primary result;
- three stations establish universal PM10 generality;
- a web search alone proves the gap.

---

## 8. Required qualifiers

Every substantive P2 claim must specify or inherit:

- station set;
- evaluation period;
- daily resolution;
- horizons;
- empirical model;
- persistence baseline;
- lag order;
- direct AR reference definition;
- paired support;
- rolling-origin protocol;
- train-only estimation;
- missingness treatment;
- uncertainty method;
- limits of generalisation.

Preferred formulation:

> Under the paired rolling-origin protocol, for the evaluated stations and horizons, the finite-memory linear reference reproduced part of the persistence-relative MSE reduction, while the remaining model-specific component depended on station, horizon and model.

This wording is permitted only after the gate passes and must be adapted to the actual results.

---

## 9. Manuscript title and editorial promise

Working title:

> A Paired Rolling-Origin Decomposition of Persistence-Relative Skill in Daily PM10 Forecasting

Editorial promise:

> Surpassing persistence does not by itself identify additional predictive information, because the observed improvement can be decomposed into one-lag dependence, additional finite linear memory and model-specific residual gain.

The promise is methodological before execution and empirical only after `P2_PAPER_GO`.
