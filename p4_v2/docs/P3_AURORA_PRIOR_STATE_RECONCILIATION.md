# P3 Aurora Prior-State Reconciliation

Audit date: 2026-08-16  
Active repository: `P4_Ghost_Skill_Dynamic_Fidelity`  
Active branch: `codex/p4-documentary-closeout`  
Active HEAD: `2f1d2552c25bc6c1aab8b9cf72bf1cbe0ab29560`

This is a documentation/provenance reconciliation. No Aurora gate, forecast,
metric, manuscript, or Overleaf artifact was executed or modified.

The exact labels `AURORA-P0`, `AURORA-P1`, `AURORA-P2`, `AURORA-NEXT`, and
Aurora-specific `E1` were not found in the active repository, the searched sibling
project corpus, or the inspected Git histories. They are therefore reported as not
recovered, not reconstructed from memory.

## 1. Product identity

The previous recoverable Aurora work referred to:

| Field | Recovered value |
|---|---|
| `PRODUCT_NAME` | Aurora Air Pollution |
| `PRODUCT_VERSION` | Aurora 0.4 degree / 0.4° |
| `PRODUCT_CLASS` | Gridded atmospheric-composition and PM forecast model |
| Direct PM role | Primary object of the proposed station-level audit |

This identity is recovered from `P5_PROJECT_CANON.md`, `P5_AURORA_PILOT.md`, and the
strategic Aurora note. Those documents explicitly distinguish:

* Aurora Air Pollution: direct gridded PM and atmospheric-composition forecasts;
* Aurora 1.5: deterministic meteorological forecasting, proposed as a possible NWP
  source for P3, not as a direct PM model;
* Aurora 1.5 Ensemble: probabilistic meteorological forecasting, not a direct PM
  ensemble.

No prior frozen experiment using Aurora 1.5 as a P3 meteorological predictor was found.
The current task's Aurora 1.5 gate is therefore a new proposed gate, not a recovered
historical gate.

## 2. Previous dominant question

The last explicit canonical wording located is in the P5 canon:

> Do the gridded advantages of atmospheric foundation models survive when PM10 and
> PM2.5 forecasts are evaluated at European monitoring stations against persistence,
> local models, amplitude and event diagnostics?

The same question is restated in the P5 strategic note as:

> Do the gridded advantages of Aurora Air Pollution versus CAMS survive when PM10 and
> PM2.5 are evaluated at European stations, preserving amplitude, episodes and skill
> against persistence?

Chronology recovered from the available documents:

1. An earlier broad Aurora gap review was archived in the Obsidian Aurora folder and
   explicitly marked superseded.
2. The P5 canon and pilot replaced that broad review with an incubation pilot focused on
   station-level evaluation and representativeness.
3. No subsequent P3 Aurora question, formal P0/P1/P2 gate, or executed Aurora result
   set was found.

## 3. Previous thesis

The bounded thesis was:

> A foundation model may perform well against gridded analysis while losing local
> amplitude, urban hotspots, exceedance events, station-level skill, or model ranking;
> the contribution would be to test whether gridded conclusions survive station-level
> evaluation.

The previous primary novelty candidate was not “using Aurora.” It was a possible change
in conclusion between gridded and station-level evaluation, associated with local
representativeness, amplitude loss, station type, forecast horizon, or ranking.

The pilot was explicitly conditional: it would become a formal line only if a
reproducible non-trivial result appeared, such as grid/station rank reversal, positive
skill with variance collapse, systematic local exceedance loss, station-type dependence,
or a ranking change after grid-to-point correction.

This thesis is distinct from the current P3 question about the incremental value of
origin-available meteorological information. It must not be silently substituted for it.

## 4. AURORA-P0

`AURORA_P0 = NOT_FOUND`

No file, manifest, commit message, or decision record with the exact `AURORA-P0` label
was found. Consequently, its purpose, inputs, outputs, pass/fail criteria, result,
freeze status, and supersession status cannot be recovered.

The P5 canon contains a general incubation/GO framework, but it is not evidence that a
formal P0 gate was executed.

## 5. AURORA-P1

`AURORA_P1 = NOT_FOUND`

No exact `AURORA-P1` artifact or result was found. The P5 pilot contains a proposed
zero-shot station-level audit, but no executed P1 gate, input manifest, prediction
release, or frozen result can be identified.

## 6. AURORA-P2

`AURORA_P2 = NOT_FOUND`

No exact `AURORA-P2` artifact or result was found. The P5 material mentions operational
availability and Aurora 1.5 as a possible P3 NWP source, but this is a role allocation
and design note, not an executed P2 gate.

## 7. E1

`E1_STATUS = NOT_FOUND`

The active P4 repository contains an unrelated historical `E1`/`E1-RR` daily
post-evaluation package. It is not an Aurora gate and must not be used to reconstruct
Aurora E1.

No Aurora-specific E1 purpose, validity conditions, output, pass/fail result, or
manifest was located. Therefore the following cannot be claimed as recovered:

* station independence as an executed E1 validity result;
* CAMS cycle stability as an executed E1 control;
* an Aurora station-level result;
* an Aurora comparison against a frozen CAMS release.

## 8. CAMS role

The recovered previous role of CAMS was:

* comparator/reference system for the direct Aurora Air Pollution station-level audit;
* gridded air-quality forecast reference against which Aurora's gridded advantage and
  station-level transfer would be assessed.

CAMS was not recovered as a formal Aurora cycle-stability gate.

`CAMS_CYCLE_STABILITY_REQUIREMENT = NOT_FOUND`

The P5 material does state that cycle/version, issue-time, latency, and product identity
must be treated carefully. That is a provenance requirement, not evidence that a cycle
stability test was performed.

## 9. Station independence

`STATION_INDEPENDENCE_REQUIRED = YES`  
`STATION_INDEPENDENCE_STATUS = NOT_TESTED`

Operational meaning in the previous design:

* the unit of analysis was a monitoring station, with results retained by station;
* more than one station and more than one station type were required for promotion;
* a single-station result was explicitly insufficient for the formal P5 line;
* failure would leave the work as an anecdotal or non-generalizable pilot result.

This was a hard promotion/GO condition, not a passed experimental result. No station
metadata manifest, station-level prediction release, or station-independence analysis
was found.

## 10. AURORA-NEXT

`AURORA_NEXT_STATUS = DEFERRED`

No exact file named `AURORA-NEXT` was found. The recoverable equivalent is the P5 canon's
“Next minimum action” and current-status section:

* P5 remained an `INCUBATION PILOT`;
* no formal fifth research line was opened;
* no new repository or benchmark should be created before the pilot specification was
  complete and data access confirmed;
* execution was deferred until the GO/NO-GO conditions could be assessed.

`AURORA_NEXT_ACTIVATION_CONDITION`:

> Complete and approve a bounded pilot specification covering target countries/networks,
> station datasets, exact Aurora product and variables, CAMS comparator, horizons,
> issue-time provenance, baselines, grid-to-station mapping, metrics, compute estimate,
> and five explicit GO/NO-GO tests; confirm data access before execution.

No activation date, forecast run, or Aurora 1.5 historical meteorology archive was
found.

## 11. Superseded tasks

Two prior task *descriptions* can be identified, but their formal names and gate IDs
were not recovered:

| Recovered descriptive task | Why superseded | What replaced it | Evidence remaining |
|---|---|---|---|
| Broad multi-gap Aurora review | It mixed product identities, imported unsupported gap claims, and treated several known limitations as open gaps | Archived Obsidian review plus the bounded P5 station-level pilot | Conceptual background only; not an executable protocol |
| Immediate formal fifth-line/full Aurora experiment | P5 was not ready for a new line because data access, station alignment, provenance and GO criteria were not closed | P5 incubation status and one-page pilot-specification prerequisite | Planning rationale and stop criteria only |

These are descriptive reconciliations, not reconstructed canonical task names. No
superseded task is reactivated.

## 12. Existing evidence

| Artifact | Type | Scientific role | Status | Canonical | Superseded | Usable now |
|---|---|---|---|---|---|---|
| `../P5_Foundation_Model_Audit/P5_PROJECT_CANON.md` | provenance/governance | Defines P5 identity, product separation, question, GO/NO-GO and deferred status | incubation pilot; no results | YES for P5 planning | NO | YES for prior-state control |
| `../P5_Foundation_Model_Audit/P5_AURORA_PILOT.md` | planning/notes | Explains the station-level pilot, product distinctions and bounded novelty | proposed pilot; no results | Supporting P5 document | NO | YES for prior-state control |
| `../P5_Foundation_Model_Audit/memoria_estrategica_aurora_foundation_models_atmosfericos.md` | strategic note | Records product roles and portfolio decision | planning only | NO | NO | YES as historical rationale |
| `20_Concepts/Aurora/Indice_Brechas_Aurora.md` | notes only | Earlier gap inventory | explicitly archived/superseded | NO | YES | NO for decisions |
| `docs/audit/P4_PROJECT_CANON.md` | project-boundary document | States Aurora is not P4 evidence until station-level predictions exist | P4 boundary only | Local uncommitted audit material | NO | Only for P4 boundary |
| `p4_v2/docs/P3_MULTISTATION_PROTOCOL_FREEZE.md` | protocol | Freezes current P3 `lags_only` benchmark and excludes meteorology | active P3 protocol | YES for current P3 | NO | YES |
| Aurora row-level predictions | row-level predictions | Would support station/grid or meteorology analysis | not found | NO | NO | NO |
| Aurora station-level metrics | station-level metrics | Would support previous station audit | not found | NO | NO | NO |
| CAMS comparison outputs | CAMS comparison | Would support Aurora-versus-CAMS evidence | not found in P3/P5 | NO | NO | NO |
| CAMS cycle analysis | cycle analysis | Would support cycle-stability validity | not found | NO | NO | NO |
| Station-independence evidence | station-level provenance | Would support multi-station validity | not found | NO | NO | NO |
| `AURORA-P0/P1/P2` manifests | gate manifests | Would establish prior frozen gates | not found | NO | UNKNOWN | NO |

The existence of planning material is not evidence that Aurora experiments were run.

## 13. Previous stop rules

The recovered P5 stop rules are preserved here without weakening:

1. Do not promote the pilot if Aurora outputs cannot be aligned reliably with station
   observations.
2. Do not claim operational use if issue-time/availability provenance is unavailable.
3. Do not promote a benchmark that reduces to another MAE/RMSE comparison.
4. Do not promote a result that duplicates an existing station-level study without a
   distinct scientific consequence.
5. Do not promote a one-station result; the result must be reproducible and not an
   anecdote.
6. Do not promote if no baseline-relative or fidelity/event result changes the
   interpretation.
7. Do not proceed when compute or data access prevents reproducibility.
8. Do not treat Aurora 1.5 as a direct PM model, or Aurora 1.5 Ensemble as a direct PM
   ensemble.
9. Do not call reanalysis use “cheating”; audit product identity and availability
   instead.
10. Do not open a formal P5 line, new repository, or fine-tuning task before the pilot
    specification and data access are approved.

## 14. Relationship to current P3

`CURRENT_P3_RELATION = COMPLEMENTARY`

| Current P3 element | Relationship to recovered Aurora state |
|---|---|
| Current P3 question | Concerns incremental PM predictability from origin-available meteorological information; prior Aurora state concerns whether a gridded direct PM model transfers to stations. |
| Scientific object | P3 compares information conditions for a local PM forecast; prior Aurora work audits gridded-to-station representativeness and model ranking. |
| Product role | Aurora Air Pollution was the prior direct PM object; Aurora 1.5 was only proposed as a possible NWP source for P3. |
| Evidence | Current P3 has a frozen lags-only benchmark; neither current P3 nor prior Aurora state has a provenance-complete Aurora 1.5 arm. |
| Governance | The prior Aurora work must not be used to claim that the current P3 operational meteorology condition has been executed. |

The relationship is not `SAME_QUESTION`: no recovered Aurora artifact compares
retrospective versus origin-available meteorology. It is not `CONFLICTING`: the two
questions can coexist if product roles and evidence are kept separate.

## 15. New-gate redundancy assessment

`NEW_AURORA_GATE = PARTIALLY_REDUNDANT`

The proposed `P3_AURORA04_AIR_POLLUTION_EVIDENCE_GATE` was not found as a prior formal
artifact and must not be treated as already executed. Its substantive controls are,
however, largely present in the prior P5 governance: exact product identity, station
alignment, multi-station validity, issue-time provenance, common valid pairs, baselines,
fidelity/event diagnostics, and reproducibility.

It is therefore redundant as a replacement for the prior GO/NO-GO logic, but could be a
future P3-specific documentary gate only after its product scope is corrected. In
particular, an `Aurora 0.4 Air Pollution` evidence gate cannot be silently applied to
`Aurora 1.5`, whose recovered role is meteorological NWP rather than direct PM prediction.
No new gate is executed in this reconciliation.

## 16. Recommended next action

Keep Aurora deferred and preserve the P5 incubation specification as the prior-state
reference. If Aurora work is resumed, first approve/recover the bounded pilot
specification and confirm product-level data access and provenance before designing any
Aurora 1.5 or Aurora Air Pollution experiment.

