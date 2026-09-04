# P3 Aurora Governance Closure

Status: `PARK`
Mode: `RECOVER-OR-AMEND`
Audit date: 2026-08-16

This document closes the bounded governance recovery. It does not authorize or
execute Aurora forecasts, P0/P1/P2, E1, metric calculation, data download,
model training, manuscript changes, or Overleaf changes.

## 1. Evidence classes

Two evidence classes are kept separate:

- `RECOVERED_HISTORICAL_GOVERNANCE`: claims supported by repository, history, or
  directly linked audit artifacts.
- `NEW_GOVERNANCE_AMENDMENT`: explicit requirements adopted now because the
  historical decision artifact could not be recovered.

The second class is not presented as historical recovery.

## 2. Final bounded recovery

| Target | Repositories/history searched | Result |
|---|---|---|
| E1 | Aurora repository, all local refs/history, `e2-met-validation`, `pm10-research-audits`, active governance folders | `NOT_FOUND` |
| Phase A→D | Aurora registered protocol/configuration and history | `PARTIAL`: sequence and gate text recovered; execution decisions B–D absent |
| AURORA-NEXT | Aurora repository/history and existing project governance | `NOT_FOUND` as a canonical artifact; prior deferred state remains unverified in repository |
| Station independence | Aurora repository/history and targeted governance corpus | No explicit independence control; station eligibility exists |
| CAMS cycle stability | Aurora repository/history and targeted governance corpus | No stability concept; shared-cycle alignment test exists |
| Pilot supersession | Aurora history and existing project governance | Descriptive supersession recovered; formal task identifiers not recovered |

No source was access-blocked in this bounded search. `NOT_FOUND` therefore means
not found in the inspected accessible scope, not proof that no artifact exists
elsewhere.

## 3. Recovered historical governance

The historical product is `Aurora Air Pollution`, also referred to as
`Aurora-AQ`. No authoritative version/checkpoint/release is frozen. The empty
`configs/aurora15_meteo.yaml` is a stub and is excluded. Aurora 1.5 and Aurora
1.5 Ensemble are not substitutes for this direct-PM product.

The recovered question is:

> Do the gridded advantages of atmospheric foundation models survive when PM10 and PM2.5 forecasts are evaluated at European monitoring stations against persistence, local models, amplitude and event diagnostics?

The recovered thesis is:

> A foundation model may perform well against gridded analysis while losing local amplitude, hotspots, exceedance events, station-level skill, or model ranking.

The remote registered protocol provides the following historical Phase A→D
sequence:

| Phase | Historical purpose | Historical gate |
|---|---|---|
| A | Registration | Protocol committed and reviewed |
| B | Single cycle | One cycle, four horizons, three stations; all nine models; leakage pass; complete provenance |
| C | Pilot | 30 days, 5–10 stations, all horizons; DM tests; no leakage violations; B pass or conditional pass |
| D | Full scale | A–C documented as pass or conditional pass |

The registered protocol itself records Phase A as `CONDITIONAL PASS`, pending
external review, and states that Phase B execution is not permitted until that
condition is resolved. The local 55-test report is infrastructure validation,
not a completed Aurora experiment.

`P0_P1_P2_TO_PHASE_A_D_MAPPING = NOT_VERIFIED`: the remote repository does not
explicitly map the labels P0/P1/P2 to the A–D phases. The current provisional
states remain `P0 = PARTIAL`, `P1 = PARTIAL`, and `P2 = PARTIAL`.

## 4. E1 recovery

`E1_RECOVERY = NOT_FOUND`.

No Aurora E1 purpose, inputs, outputs, pass/fail criteria, validity conditions,
result, or canonical source was recovered. An unrelated E1 package in the
active P4 repository is not used.

E1 is not recreated by this closure.

## 5. Station-independence control

`STATION_INDEPENDENCE_REQUIRED = NOT_VERIFIED`.

The historical Aurora repository contains a station-eligibility specification
and regional thresholds, but no explicit station-independence concept or test.
Eligibility and independence remain distinct. No requirement is upgraded from
eligibility to independence by inference.

## 6. CAMS cycle control

`CAMS_CYCLE_STABILITY_REQUIRED = NOT_VERIFIED`.

The historical P2 configuration and tests require Aurora and CAMS to share a
`cycle_time`. This is a shared-cycle alignment check. It is not evidence of a
longitudinal CAMS cycle-stability requirement or result.

## 7. AURORA-NEXT

`AURORA_NEXT_STATUS = DEFERRED_UNVERIFIED_IN_REPOSITORY`.

No canonical AURORA-NEXT artifact or exact activation contract was recovered.
The prior project state indicates deferred activation without a fixed date.
That conversational/project state is not relabelled as a repository decision.

## 8. Pilot supersession decisions

Two descriptive supersessions remain recoverable, but their formal task IDs were
not found:

| Superseded task | Replacement | Reactivation |
|---|---|---|
| Broad multi-gap Aurora review mixing product, station, availability, fidelity and novelty questions | Bounded station-level pilot and registered Phase A→D protocol | `NO` |
| Immediate full Aurora/fifth-line experiment before data and provenance closure | Gated progression plus deferred pilot specification | `NO` |

The `experiments/incremental-skill-v1` branch records earlier incremental-value
framing; it contains no executed Aurora evidence and is not reactivated.

## 9. Amendment decision

`MISSING_GOVERNANCE_RECOVERY = PARTIAL`.

The following historical elements remain missing: E1, an exact AURORA-NEXT
decision record, explicit station independence, explicit CAMS cycle stability,
and formal P0/P1/P2-to-A–D mapping. Therefore:

`NEW_GOVERNANCE_AMENDMENT_REQUIRED = YES`
`NEW_GOVERNANCE_AMENDMENT_CREATED = YES`

The new amendment is [P3_AURORA_GOVERNANCE_AMENDMENT_01.md](P3_AURORA_GOVERNANCE_AMENDMENT_01.md).
It defines only minimum activation conditions already implied by the recovered
scientific question and existing protocol; it does not add a new hypothesis.

## 10. Scientific boundary

Aurora's question remains separate from current P3:

- `AURORA QUESTION`: station-level survival of gridded Aurora Air Pollution
  advantages against persistence/local models and amplitude/event diagnostics.
- `CURRENT P3 QUESTION`: incremental PM10/PM2.5 predictive value of meteorology
  restricted to information genuinely available at each forecast origin.
- `RELATIONSHIP = COMPLEMENTARY`.

Aurora is not converted into the current P3 meteorological condition. Ghost
skill is not made Aurora's primary question.

## 11. Canonical closure state

```text
PROJECT = P3
AURORA_PRODUCT = Aurora Air Pollution
AURORA_VERSION = NOT_FROZEN
E1_RECOVERY = NOT_FOUND
PHASE_A_D_GOVERNANCE = PARTIAL
STATION_INDEPENDENCE_REQUIRED = NOT_VERIFIED
CAMS_CYCLE_STABILITY_REQUIRED = NOT_VERIFIED
AURORA_NEXT_STATUS = DEFERRED_UNVERIFIED_IN_REPOSITORY
MISSING_GOVERNANCE_RECOVERY = PARTIAL
NEW_GOVERNANCE_AMENDMENT_REQUIRED = YES
NEW_GOVERNANCE_AMENDMENT_CREATED = YES
AURORA_NOVELTY_VERIFIED = NO
CURRENT_P3_RELATION = COMPLEMENTARY
AURORA_EXPERIMENT_AUTHORIZED = NO
AURORA_STATUS = GOVERNANCE_READY_BUT_NOVELTY_UNVERIFIED
NEXT_MINIMUM_ACTION = Keep Aurora deferred; complete the dedicated novelty kill test only after the amended activation contract is reviewed.
MANUSCRIPT_UPDATE_AUTHORIZED = NO
```

`GOVERNANCE_READY_BUT_NOVELTY_UNVERIFIED` means the minimum activation contract
is now explicit. It does not mean that any activation gate has passed.

## 12. Next minimum action

Review and approve Amendment 01 as a governance artifact; do not execute Aurora
until G0–G3 are evidenced and the dedicated Aurora novelty kill test is at least
`MODERATE`.
