# P3 Aurora Governance Amendment 01

Status: `NEW_GOVERNANCE_AMENDMENT`
Date: 2026-08-16
Applies to: `P3_AURORA_GOVERNANCE_CLOSURE.md`

## Explicit status

**THIS IS A NEW GOVERNANCE DECISION.**
**IT IS NOT A RECONSTRUCTION OF MISSING HISTORICAL GOVERNANCE.**

Historical E1, an exact AURORA-NEXT decision record, explicit station
independence, explicit CAMS cycle stability, and formal P0/P1/P2-to-Phase-A→D
mapping were not recovered in the bounded search. This amendment supplies only
the minimum conditions required before a future Aurora experiment may be
activated.

## 1. Scope and scientific boundary

The Aurora question is frozen as:

> Do the gridded advantages of atmospheric foundation models survive when PM10 and PM2.5 forecasts are evaluated at European monitoring stations against persistence, local models, amplitude and event diagnostics?

Current P3 remains a separate question about meteorological incremental value
under genuine forecast-origin availability. The relationship is
`COMPLEMENTARY`.

No new scientific hypothesis is introduced. Aurora use, station evaluation,
dynamic fidelity, event diagnostics, and ghost skill are not declared novel by
this amendment.

## 2. Activation sequence

The minimum activation sequence is `G0 → G1 → G2 → G3 → G4`.

G0–G3 are pre-execution admissibility gates. G4 describes the minimum future
scientific comparison and is not executed by this amendment.

## 3. G0 — Product identity

Before any data extraction:

- identify the product exactly as `Aurora Air Pollution`;
- record version, checkpoint, release or equivalent provenance when available;
- retain the earlier alias `Aurora-AQ` only as an alias;
- do not conflate Aurora Air Pollution with Aurora 1.5 or Aurora 1.5 Ensemble;
- do not treat the test-fixture value `0.25` as a model version;
- do not use the empty `aurora15_meteo.yaml` stub as product evidence.

G0 fails if the product identity cannot be tied to a reproducible artifact.

## 4. G1 — Data access and provenance

Before any forecast evaluation, record:

- target variables PM10 and/or PM2.5;
- source product and archive location;
- historical forecast provenance and deterministic extraction procedure;
- initialization/issue time, valid time, and lead time whenever operational
  interpretation is intended;
- source-file or release hashes;
- units and all conversions;
- version/checkpoint and download metadata.

Unknown availability must not be labelled operational. A missing historical
archive blocks an operational interpretation but does not authorize a
retrospective reconstruction to be called operational.

## 5. G2 — Station eligibility

Before inspecting forecast performance:

- define the European monitoring stations and station metadata;
- define observation compatibility and pollutant identity;
- freeze the grid-to-station spatial matching rule;
- define minimum valid station/origin support;
- record exclusions deterministically;
- preserve common station/origin/horizon support across compared systems.

Station eligibility is not station independence. No independence claim is
authorized unless a separate explicit design is later approved.

## 6. G3 — Temporal and cycle validity

Before interpretation:

- identify forecast cycles and their relation to forecast origins;
- retain common origin, horizon, station, and target support;
- prohibit opportunistic cycle selection;
- assess missing, duplicated, delayed, or mismatched cycles before computing
  scientific conclusions;
- document any availability/latency assumptions.

This gate is intentionally named temporal/cycle validity. It must not be called
`CAMS cycle stability` unless CAMS is part of the future experiment and an
explicit stability question is justified and preregistered.

## 7. G4 — Minimum scientific comparison

This is a future experiment specification, not an execution:

- persistence;
- frozen local model(s);
- Aurora Air Pollution;
- identical station/origin/horizon/`y_true` support.

Primary metrics:

- MAE;
- RMSE;
- bias;
- skill relative to persistence.

Secondary diagnostics may be used only if needed to interpret a result:

- amplitude or variance retention;
- event metrics;
- rank reversal.

Ghost skill is not a primary objective. No arbitrary post-hoc diagnostic cutoff
is authorized by this amendment.

## 8. Future activation rule

Aurora may move from deferred to candidate-for-activation only when:

1. this amendment is reviewed and accepted;
2. G0–G3 are evidenced and pass;
3. the minimum station-level comparison is feasible;
4. no product conflation remains;
5. a dedicated Aurora novelty kill test returns at least `MODERATE` gap strength.

Until then:

```text
AURORA_EXPERIMENT_AUTHORIZED = NO
AURORA_NOVELTY_VERIFIED = NO
AURORA_NEXT_STATUS = DEFERRED_UNVERIFIED_IN_REPOSITORY
```

No activation date is set.

## 9. Amendment decision

```text
AMENDMENT_ID = P3_AURORA_GOVERNANCE_AMENDMENT_01
AMENDMENT_TYPE = NEW_GOVERNANCE_AMENDMENT
HISTORICAL_RECOVERY_REPLACED = NO
NEW_SCIENTIFIC_HYPOTHESIS = NO
PRODUCT = Aurora Air Pollution
VERSION = NOT_FROZEN
CURRENT_P3_RELATION = COMPLEMENTARY
AURORA_EXPERIMENT_AUTHORIZED = NO
MANUSCRIPT_UPDATE_AUTHORIZED = NO
```
