# EMS scope audit

| Criterion | Result | Evidence or minimal action |
|---|---|---|
| Generic evaluation issue identified | YES | Error eligibility and dynamic fidelity answer distinct questions. |
| Environmental application clear | YES | Daily PM10 forecasting at Spanish monitoring stations. |
| Objective explicit | YES | Introduction states the model-eligibility question. |
| Rationale for approach explicit | YES | Error-based screening is followed by complementary fidelity checks. |
| Historical progress explained | YES | Introduction/background cover error, baseline comparison, verification, temporal validation, and event/variability diagnostics. |
| Added value over existing metrics explicit | YES | The paper quantifies the eligibility consequence; it does not claim new metrics. |
| Substantial testing documented | YES | 595 structured cells, five models, seven horizons, five expanding folds. |
| Alternative model families compared | YES | HGB, Ridge, SARIMA, seasonal naive, and STL+Ridge. |
| Temporal validation documented | YES | Expanding rolling-origin validation and train-only controls are stated. |
| Quality assurance documented | YES | Frozen-table provenance, integrity checks, deterministic generators, DM/HLN, BH, and traceability are documented. |
| Sensitivity assessment included | YES | Deterministic 3 x 3 Rule-B threshold neighbourhood; verdict ROBUST. |
| Transferable insight explicit | YES | Discussion states the evaluation-design insight separately from benchmark magnitudes. |
| Empirical generalisation bounded | YES | Limitations restrict claims to the pollutant, stations, models, horizons, and design. |
| Interest beyond PM10 explained | YES | The evaluation principle is framed for environmental model evaluation, without transferring the 97.1% rate. |

`SCOPE_VERDICT = PASS_WITH_DOCUMENTED_LIMITATIONS`

The SARIMA configuration conflict is documented and treated as a
submission-acceptable reproducibility limitation rather than inferred.
