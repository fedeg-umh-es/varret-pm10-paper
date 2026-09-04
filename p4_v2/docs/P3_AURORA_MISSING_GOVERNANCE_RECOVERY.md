# P3 Aurora Missing-Governance Recovery

Audit mode: read-only provenance and governance recovery. No Aurora forecast,
P1/P2/E1 execution, metric recomputation, manuscript edit, or Overleaf change
was performed.

## 1. Repositories inspected

| Repository | Ref/HEAD | Inspection purpose | Access |
|---|---|---|---|
| `fedeg-umh-es/aurora-pm-station-corrector` | `main` / `6f05ca63f1822fee210c455baa1072bfe66ccb82` | Aurora product, registered protocol, P1/P2 infrastructure, tests and outputs | READ_ONLY |
| `fedeg-umh-es/aurora-pm-station-corrector` | `experiments/incremental-skill-v1` / `350ec76078f5e476ac04a46040ba233e225ff6d8` | Earlier experiment wording and supersession history | READ_ONLY |
| `fedeg-umh-es/aurora-pm-station-corrector` | `claude/aurora-pm-station-corrector-2hg6vo` / `cadc95a2f94e57758edeccc66925a70b98cb90c3` | P0 validation history | READ_ONLY |
| `fedeg-umh-es/e2-met-validation` | `main` / `f83a0c19c636c08a470dccefef4dcc94bd4557b0` | Targeted search for linked E1/CAMS/station-governance artifacts | READ_ONLY |
| `fedeg-umh-es/pm10-research-audits` | `main` / `64a77138c11d3c3d6ff22a068d74015347ad0438` | Targeted cross-project governance and access records | READ_ONLY |
| `P4_Ghost_Skill_Dynamic_Fidelity` | `codex/p4-documentary-closeout` / `2f1d2552c25bc6c1aab8b9cf72bf1cbe0ab29560` | Active P3 boundary and output location | READ_ONLY |

The two targeted auxiliary repositories were accessible. No access failure was
recorded; absence claims below are limited to the inspected repository/history
and searched terms.

## 2. Access failures

`NONE`.

The active worktree contained pre-existing changes, including `p4_v2/`; they
were preserved. No checkout, reset, clean, stage, commit, or push was performed.

## 3. Product identity

The authoritative recovered product identity is:

- `PRODUCT_NAME = Aurora Air Pollution`
- earlier alias: `Aurora-AQ`
- `PRODUCT_CLASS = gridded atmospheric-composition / direct PM forecast model`

The repository registered protocol explicitly says Aurora was fine-tuned using
CAMS-derived products. The repository does not establish independence from
CAMS and prohibits such a claim.

The empty `configs/aurora15_meteo.yaml` is a stub and is not evidence for this
direct-PM line. Aurora 1.5 and Aurora 1.5 Ensemble remain distinct products and
were not substituted for Aurora Air Pollution.

## 4. Version identity

`AURORA_VERSION = NOT_FROZEN`.

No authoritative manifest, checkpoint identifier, release tag, or product
metadata in the inspected Aurora repository freezes a version for Aurora Air
Pollution. The only `0.25` occurrence located is the synthetic test fixture
`microsoft/aurora-0.25-pretrained`; it is not a product-version record. No
authoritative evidence supports `0.4` as the canonical version.

The earlier local `P3_AURORA_PRIOR_STATE_RECONCILIATION.md` is retained as
historical work and is not overwritten here; its inferred `0.4 degree` field is
superseded by this evidence-based correction.

## 5. Scientific question

Preserved recovered question:

> Do the gridded advantages of atmospheric foundation models survive when PM10 and PM2.5 forecasts are evaluated at European monitoring stations against persistence, local models, amplitude and event diagnostics?

## 6. P0/P1/P2

### AURORA-P0

`AURORA_P0 = PARTIAL`.

The recoverable P0-equivalent is Phase A registration in
`docs/registered_protocol.md` and `configs/experiment_registered.yaml`.
The protocol records `Phase A — CONDITIONAL PASS`, pending external review, and
prohibits Phase B until the condition is resolved. A validation artifact reports
55 passing local tests on the tested commit, with no failures, but this verifies
infrastructure; it is not a completed Aurora scientific run or a released
formal gate result.

### AURORA-P1

`AURORA_P1 = PARTIAL`.

P1 infrastructure is present in `configs/station_eligibility.yaml`,
`scripts/p1_01_compute_station_eligibility.py`, and station-eligibility tests.
The contract defines coverage, gap, duplicate, quality, and regional-decision
criteria. The committed station manifest is still a `TBD_001` pending fixture;
no real multi-station eligibility output or real-data execution artifact was
found. Station eligibility is therefore specified, not demonstrated.

### AURORA-P2

`AURORA_P2 = PARTIAL`.

Single-cycle hindcast infrastructure is present in
`configs/hindcast_single_cycle.yaml` and time/spatial-alignment tests. The
configuration specifies one cycle, three placeholder stations, PM10/PM2.5,
four lead times, shared Aurora/CAMS cycle time, spatial operators, and required
output fields. No real Aurora/CAMS input, real station output, or executed
single-cycle hindcast was found. The smoke-test outputs are persistence-only
synthetic scaffolding and do not establish P2 scientific execution.

## 7. E1

`E1_STATUS = NOT_FOUND`.

The exact targeted terms `E1`, `Aurora E1`, `eligibility E1`, and related station
independence/cycle-stability terms did not recover an Aurora E1 artifact in the
Aurora repository, its history, `e2-met-validation`, or the targeted
`pm10-research-audits` corpus. An unrelated E1 package in the active P4
repository is not used as Aurora evidence.

No Aurora E1 purpose, inputs, outputs, pass/fail criteria, validity conditions,
result, or canonical manifest can be claimed.

## 8. Station independence

`STATION_INDEPENDENCE_CONCEPT_FOUND = NO`.

`STATION_ELIGIBILITY_FOUND = YES`: the P1 infrastructure defines station
eligibility and regional promotion thresholds. It does not define a formal
station-independence test. The registered protocol requires station-level rows,
and the P1 configuration requires multi-year station data, but neither is an
independence result.

`STATION_INDEPENDENCE_REQUIRED = NOT_VERIFIED`.

`RELATION_BETWEEN_THEM = RELATED_BUT_DISTINCT`.

The consequence of failing the available station-eligibility requirements is
failure to proceed with the pilot dataset, not a demonstrated statistical
independence failure. No station-independence evidence was recovered.

## 9. CAMS cycle stability

`CAMS_ROLE = comparator/reference forecast and Aurora input/dependency context`.

`CAMS_CYCLE_STABILITY_CONCEPT_FOUND = NO`.

`SHARED_CYCLE_TEST_FOUND = YES`: the P2 configuration requires Aurora and CAMS
to share `cycle_time`, and `tests/test_time_alignment.py` contains pass/fail
tests for this pairwise alignment. This is a shared-cycle consistency test, not
a longitudinal CAMS cycle-stability analysis.

`RELATION_BETWEEN_THEM = SHARED_CYCLE_ALIGNMENT_ONLY`.

`CAMS_CYCLE_STABILITY_REQUIRED = NOT_VERIFIED`.

## 10. AURORA-NEXT

`AURORA_NEXT_STATUS = DEFERRED_UNVERIFIED_IN_REPOSITORY`.

No repository artifact named `AURORA-NEXT` or an equivalent decision record was
recovered. The prior project-level planning state available to this audit
describes Aurora as deferred pending a bounded pilot specification, confirmed
data access, exact product/checkpoint identity, station alignment, and complete
provenance. No rigid activation date was recovered.

`AURORA_NEXT_ACTIVATION_CONDITION`:

> Approve/recover a bounded Aurora pilot specification and confirm real-data and provenance readiness before any Aurora execution.

`AURORA_NEXT_DEPENDENCIES`:

- exact supported Aurora product/checkpoint identity;
- real station and CAMS inputs;
- station/grid and time alignment;
- issue/availability provenance;
- leakage-safe row-level outputs;
- unresolved P0/P1/P2 governance conditions.

`AURORA_NEXT_NEXT_ACTION`:

> Recover or explicitly approve the missing E1 and Phase A→D decision artifacts; keep Aurora execution disabled until then.

## 11. Phase A→D governance

`PHASE_A_D_GOVERNANCE = PARTIAL`.

The registered protocol provides the following sequence:

| Phase | Scope | Required gate |
|---|---|---|
| A | Registration | Protocol committed and reviewed; current decision remains conditional |
| B | Single cycle | One cycle, four horizons, three stations; all nine models, leakage pass, provenance complete |
| C | Pilot | 30 days, 5–10 stations, all horizons, DM tests, no leakage violations, B pass/conditional pass |
| D | Full scale | A–C documented as pass or conditional pass |

The sequence and dependencies are recovered, but no executed B/C/D evidence was
found. The `AURORA-P0/P1/P2` labels are a current governance mapping, not labels
present in the remote repository itself. Unresolved conditions include external
review of Phase A, real sources/stations, the Aurora checkpoint, CAMS dataset,
and the remaining `estimator_family: TBD` field before corrector training.

## 12. Superseded tasks

The exact formal names of the two previously described pilot tasks were not
recovered as repository gate identifiers. The following descriptive tasks are
retained as superseded and are not reactivated:

| Task | Original purpose | Superseded by | Remaining value | Reactivation |
|---|---|---|---|---|
| Broad multi-gap Aurora review | Mix Aurora product roles, station transfer, availability, fidelity, and novelty questions | Bounded station-level pilot / registered Phase A protocol | Historical rationale only | NO |
| Immediate full Aurora/fifth-line experiment | Start a formal large experiment before data, station, and provenance gates were closed | Phase A→D gated progression and deferred pilot specification | Stop criteria and planning context only | NO |

The earlier `experiments/incremental-skill-v1` branch is evidence of the earlier
incremental-value framing, not evidence of an executed experiment; it was
superseded by the registered protocol and P1/P2 infrastructure.

## 13. New-gate compatibility

Assessment of the previously drafted `P3_AURORA04_AIR_POLLUTION_EVIDENCE_GATE`:

| Component | Status | Reason |
|---|---|---|
| Product identity | ALREADY_GOVERNED | Registered protocol and data contract define Aurora Air Pollution and its CAMS dependency |
| Literal `04` version qualifier | CONFLICTING | No authoritative `0.4` version is frozen |
| Provenance | PARTIALLY_GOVERNED | Required checkpoint, dataset, cycle, latency and hashes are specified, but real values are TBD |
| Target compatibility | PARTIALLY_GOVERNED | Direct PM station evaluation is the registered Aurora object; no real aligned target exists |
| Temporal matching | ALREADY_GOVERNED | `cycle_time`, `valid_time`, lead and availability invariants are specified and tested synthetically |
| Spatial matching | ALREADY_GOVERNED | Bilinear/nearest operators, grid coordinates and consistency tests are specified |
| Row-level evidence | ALREADY_GOVERNED | Row schema and provenance requirements are explicit; no real release exists |
| Common support | PARTIALLY_GOVERNED | Comparable rows are required, but no real Aurora/CAMS common-support output exists |
| Station-level comparison | PARTIALLY_GOVERNED | Unit and eligibility infrastructure exist; real station panel is absent |
| Persistence comparison | ALREADY_GOVERNED | Persistence is defined and synthetic smoke output exists |
| Local-model comparison | ALREADY_GOVERNED | Local-only and local-plus-Aurora/CAMS variants are registered |
| Amplitude diagnostics | PARTIALLY_GOVERNED | Metrics are specified, but no Aurora result exists |
| Event diagnostics | PARTIALLY_GOVERNED | Train-only threshold and event metrics are specified, but no result exists |

`NEW_AURORA_GATE = PARTIALLY_REDUNDANT`.

The proposed gate is not a previously executed gate. Its substantive controls
largely duplicate the registered protocol and Phase A→D requirements, while the
missing real-data/provenance conditions remain unresolved. The `Aurora04` name
must not be used as a canonical product identity.

## 14. Current P3 relationship

`CURRENT_P3_RELATION = COMPLEMENTARY`.

- Current P3 primary question: origin-available meteorological incremental value.
- Aurora question: whether gridded Aurora Air Pollution advantages survive
  station-level PM evaluation against persistence/local models and amplitude/events.
- Aurora Air Pollution is a direct PM model, not the current P3 meteorological
  feature condition.
- Aurora 1.5 and Aurora 1.5 Ensemble are not substituted into this recovered line.
- Dynamic fidelity, amplitude, and event diagnostics may remain Aurora verification
  dimensions, but they are not the primary novelty of current P3.
- Ghost skill is not the primary Aurora question.

## 15. Canonical Aurora state

Only the following claims are frozen by this recovery:

```text
PROJECT = P3
AURORA_PRODUCT = Aurora Air Pollution
AURORA_VERSION = NOT_FROZEN
AURORA_PREVIOUS_DOMINANT_QUESTION = Do the gridded advantages of atmospheric foundation models survive when PM10 and PM2.5 forecasts are evaluated at European monitoring stations against persistence, local models, amplitude and event diagnostics?
AURORA_PREVIOUS_THESIS = A foundation model may perform well against gridded analysis while losing local amplitude, hotspots, exceedance events, station-level skill, or model ranking.
AURORA_P0 = PARTIAL
AURORA_P1 = PARTIAL
AURORA_P2 = PARTIAL
E1_STATUS = NOT_FOUND
STATION_INDEPENDENCE_REQUIRED = NOT_VERIFIED
CAMS_CYCLE_STABILITY_REQUIRED = NOT_VERIFIED
AURORA_NEXT_STATUS = DEFERRED_UNVERIFIED_IN_REPOSITORY
PHASE_A_D_GOVERNANCE = PARTIAL
CURRENT_P3_RELATION = COMPLEMENTARY
NEW_AURORA_GATE = PARTIALLY_REDUNDANT
PREVIOUS_AURORA_STATE_RECOVERED = PARTIAL
AURORA_EXPERIMENT_AUTHORIZED = NO
NEXT_MINIMUM_ACTION = Recover or explicitly approve the missing E1 and Phase A→D decision artifacts before any Aurora gate or experiment.
MANUSCRIPT_UPDATE_AUTHORIZED = NO
```

No novelty claim is made here about Aurora use, foundation models, station
evaluation, dynamic fidelity, or event diagnostics.

## 16. Next minimum action

Keep Aurora parked and recover or explicitly approve the missing E1 and Phase
A→D decision artifacts. Do not authorize Aurora execution, P1/P2/E1, or a new
Aurora gate before that governance gap is closed.
