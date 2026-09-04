# P3 Novelty Kill Test

## Executive verdict

GAP_STRENGTH = NOT_YET_VERIFIED

PRIMARY_NOVELTY_CANDIDATE = An availability-controlled comparison of pollutant-lag forecasts, retrospective meteorology, and genuinely origin-available meteorological forecasts on identical station-origin-horizon support.

PRIMARY_RISK = The audited corpus already contains lags-versus-meteorology comparisons and horizon/persistence analyses, while no executed P3 operational-meteorology arm with issue-time provenance was located.

GHOST_SKILL_AS_PRIMARY_NOVELTY = NO

DYNAMIC_FIDELITY_AS_PRIMARY_NOVELTY = NO

METEOROLOGICAL_AVAILABILITY_AS_PRIMARY_NOVELTY = NOT_YET_VERIFIED

The candidate gap survives provisionally as a question about information availability, not as a claim that existing studies universally misuse retrospective meteorology. The available corpus is insufficient to establish the state of the external literature. The current P3 benchmark contains only `lags_only`; the existing meteorology comparison is legacy/retrospective and not aligned to the frozen 323-station daily panel.

## 1. Candidate gap

Candidate statement:

> Existing PM10/PM2.5 forecasting studies may report gains from meteorological predictors, but it is unclear whether those gains remain when meteorological information is separated according to what was actually available at each forecast origin, and whether error-based gains correspond to preserved pollutant dynamics and event representation across forecast horizons.

| Component | Status | Basis in audited corpus |
| --- | --- | --- |
| Meteorological predictors are used in PM10/PM2.5 forecasting | VERIFIED | P1/P2 meteorology materials and bibliography entries. |
| Meteorology is separated by actual forecast-origin availability | PARTIALLY_VERIFIED | The P3-OP-01 contract specifies issue time, cycle, latency and spatial assignment; the executed legacy comparisons are explicitly retrospective. |
| Operational forecast meteorology is available with verifiable issue metadata | NOT_VERIFIED | No executed operational NWP/Aurora arm with complete origin-level metadata was found in the active P3 corpus. |
| Retrospective meteorology and lags-only are compared | PARTIALLY_VERIFIED | Madrid and Ireland legacy records document `lags_only` versus `lags_meteo`; the Ireland original predictions are not fully recoverable. |
| Comparisons use identical station-origin-horizon support | PARTIALLY_VERIFIED | The legacy protocol requires aligned support, but the current P3 daily panel has not yet been paired with a meteorological arm. |
| Error-based gains are checked against dynamic fidelity | PARTIALLY_VERIFIED | P4 and legacy meteorology interpretations contain fidelity diagnostics, but not under the current availability-controlled P3 design. |
| Error-based gains are checked against event representation | PARTIALLY_VERIFIED | P4/P3 diagnostic tables contain event metrics; the legacy meteorology comparison does not establish the full controlled comparison. |
| The complete literature lacks this distinction | NOT_VERIFIED | The repository corpus is not a complete literature search. |

The corpus therefore does not kill the availability-controlled question, but it also does not justify describing it as an established literature-wide gap.

## 2. Established components that are not novel

The following are established methodology or evaluation practice and can support rigor only:

- rolling-origin or expanding temporal validation;
- leakage-safe, training-only preprocessing;
- persistence as a forecast baseline;
- MAE, RMSE, bias and skill scores;
- horizon-wise comparisons;
- variance retention, standard-deviation ratios and correlation;
- event precision, recall/POD, FAR and CSI;
- multidimensional forecast verification;
- pairwise rank comparisons;
- KGE/Murphy-style decompositions and related fidelity summaries.

The combination `meteorology + H* + fidelity + events` is not, by itself, a sufficient contribution. The P1/P2 line already combines lag-only versus meteorological conditions, persistence, rolling-origin evaluation and an H*-type predictability summary. P4 already contains the error--fidelity/event disagreement diagnostic. P3 can only remain distinct if origin availability changes the scientific interpretation of meteorological value.

## 3. Literature coverage matrix

The detailed matrix is stored at:

`p4_v2/docs/P3_NOVELTY_GAP_COVERAGE_MATRIX.csv`

It contains 11 audited records. Project protocol/run records are separated from bibliography-only records. Missing methodological details are marked `UNKNOWN`; they are not treated as evidence of absence.

The strongest directly relevant corpus records are:

| Record | What is covered | What remains open |
| --- | --- | --- |
| P3 E2-MET Madrid | PM10, persistence, rolling-origin, horizon-wise lags versus meteorology, H* | No verified issue-time/cycle/latency metadata; legacy and single-site. |
| P3 E2-MET Ireland | Eight PM10 stations, persistence, rolling-origin, horizon-wise lags versus meteorology, H* | Retrospective concurrent meteorology; incomplete original row-level provenance. |
| P3-OP-01 contract | Formal separation of retrospective and operational arms and required availability metadata | Contract is not an executed operational comparison. |
| P3-v2 lags-only | Frozen 323-station daily common-support reference with dynamic/event diagnostics | No meteorological condition yet. |
| P4 dynamic-fidelity line | Error skill, variance retention, event behaviour and rank disagreement | No meteorological availability comparison. |

The bibliography-only records establish relevant subject matter but do not establish the P3 information-condition gap because their protocols were not available in the audited corpus.

## 4. Evidence for absence

| Question | Result | Interpretation |
| --- | --- | --- |
| Is there evidence that the literature fails to distinguish meteorological information by real forecast-origin availability? | PARTIALLY_SUPPORTED | The audited project records show the distinction is often absent from executed legacy comparisons, but the corpus cannot support an external literature-wide statement. |
| Is retrospective meteorology commonly interpreted as operational? | NOT_SUPPORTED | The relevant project documentation explicitly labels the legacy meteorology arm retrospective. No broad external claim is supported. |
| Is operational NWP meteorology absent from identical-support comparisons with lags-only and retrospective meteorology? | UNKNOWN | No such comparison was located in the corpus; external verification is required. |
| Are error-based meteorological gains jointly checked for dynamic/event fidelity? | PARTIALLY_SUPPORTED | The project corpus contains related diagnostic work, but not the complete availability-controlled P3 comparison. |

## 5. Scientific friction

| Friction | Status | Current evidence and limitation |
| --- | --- | --- |
| F1. Retrospective improvement weakens under origin-available meteorology | NOT_YET_TESTED | The legacy retrospective comparison cannot answer this. |
| F2. Meteorology extends error skill without equivalent fidelity/event improvement | SCIENTIFICALLY_CONSEQUENTIAL | Legacy interpretation reports accuracy improvement with mixed fidelity gains; the availability-controlled version remains untested. |
| F3. Meteorological value changes with horizon | SCIENTIFICALLY_CONSEQUENTIAL | Madrid/Ireland H* and horizon-wise records show different effects by lead; this is not yet an availability-controlled P3 result. |
| F4. Meteorological value differs by local persistence regime | SCIENTIFICALLY_CONSEQUENTIAL | The P1/P2 corpus explicitly contrasts high- and moderate-autocorrelation regimes; this is already substantial overlap. |
| F5. Accuracy and fidelity/event criteria change preference between information conditions | NOT_YET_TESTED | P4 shows preference disagreement within forecasting cells; the meteorological-condition comparison has not been executed on the frozen P3 support. |

These frictions make the question potentially consequential, but only F1 and the controlled form of F5 are clearly distinct from existing project work.

## 6. Existing-paper overlap

| Candidate contribution | Already covered by | Overlap level | What is truly distinct if any | Risk of salami slicing |
| --- | --- | --- | --- | --- |
| Error skill versus dynamic/event fidelity | P4 dynamic-fidelity/eligibility audit and current P3 lags-only diagnostics | HIGH | Only distinct if applied as a secondary check to meteorological information conditions. | HIGH |
| Meteorology versus persistence-relative skill | P1/P2 E2-MET Madrid and Ireland | HIGH | Origin-availability control and identical current P3 support. | HIGH |
| Horizon-dependent meteorological predictability | P1/P2 H* line | HIGH | None unless operational availability changes the horizon conclusion. | HIGH |
| Local persistence-regime heterogeneity | P1/P2 Madrid--Ireland comparison | DUPLICATIVE | Current 323-station daily panel could broaden the descriptive context, but not the core idea. | HIGH |
| Frozen 323-station lags-only reference | P3-v2 current benchmark | LOW | Provides the common reference needed for a later meteorology increment. | LOW |
| Origin-level operational availability contract | P3-OP-01 documentation | MEDIUM | An executed, provenance-complete operational arm would supply empirical content absent from the contract. | MEDIUM |
| H* or predictability-bound theory | Existing predictability-bound work | HIGH | P3 should use horizon-wise skill only as a secondary description, not create another H* story. | HIGH |
| Pairwise preference reversal | P4 rank-reversal/dynamic-fidelity line | HIGH | Meteorological-condition preference changes would be distinct only if tied to availability, not as a generic reversal count. | HIGH |

The known statement that error skill and dynamic/event fidelity can differ is already covered and cannot serve as P3's paper-level contribution alone.

## 7. Kill test by candidate claim

| Claim | Prior art found | Corpus support | What would kill it | Current status | Maximum defensible wording |
| --- | --- | --- | --- | --- | --- |
| N1. Controlling meteorological availability changes estimated incremental value | P1/P2 compares meteorology, but not a verified operational arm | PARTIAL | A complete audit shows prior studies already use the same availability-controlled design, or P3 conditions produce no interpretable difference. | SURVIVES_PROVISIONALLY | “The study tests whether the estimated incremental value of meteorology depends on the information condition defined at forecast origin.” |
| N2. Meteorological contribution is horizon-dependent | P1/P2 H* and horizon-wise results | VERIFIED | The claim is not distinct from the existing H* line, or no horizon variation is observed. | WEAK | “In the evaluated design, any meteorological increment is reported horizon by horizon.” |
| N3. Retrospective meteorology can overstate operational predictability | P3-OP-01 defines the distinction; no executed operational comparison located | PARTIAL | External literature or an existing project result provides a provenance-complete retrospective-versus-operational comparison with the same conclusion. | SURVIVES_PROVISIONALLY | “Retrospective meteorology is an upper-bound condition; its difference from origin-available forecast meteorology must be tested rather than assumed.” |
| N4. Error-based meteorological gains do not imply equivalent fidelity/event preservation | P4 dynamic-fidelity line and legacy meteorology interpretation | PARTIAL | Existing project work already tests this under the same conditions and support, or the controlled P3 results show no divergence. | DEAD | “Dynamic and event diagnostics are secondary checks on meteorological gains; they are not a standalone P3 novelty claim.” |
| N5. The effect differs by local persistence regime | P1/P2 Madrid--Ireland line explicitly advances this idea | VERIFIED | No regime interaction is found, or the result is indistinguishable from the existing comparison. | DEAD | “Prior work motivates, but does not by itself establish, why persistence regime should be considered when interpreting meteorological increments.” |

N1 and N3 remain conditional questions. N2, N4 and N5 are already substantially occupied by project papers or diagnostic modules.

## 8. Dominant question

All surviving work should answer one question:

> Does meteorological information extend operational PM10/PM2.5 predictability when forecast-origin availability is controlled?

The secondary condition is:

> If error-based predictability extends, is that extension also visible in dynamic and event representation?

Ghost skill must remain a diagnostic label inside this question. A separate ghost-skill storyline would duplicate P4 and split the contribution.

## 9. Negative-result value

| Outcome | Scientific information gained | Publishable without a model-win narrative? | Maximum claim |
| --- | --- | --- | --- |
| A. Operational meteorology materially extends predictability | Quantifies incremental information that survives origin-availability constraints and identifies its horizon range. | YES | “Under the evaluated provenance contract, the operational meteorological condition improved persistence-relative predictability over the specified horizons and support.” |
| B. Retrospective meteorology helps but operational meteorology does not | Demonstrates that an upper-bound retrospective result is not an operational forecast benefit. | YES | “Retrospective meteorology overstated the incremental value relative to the origin-available condition in this design.” |
| C. Neither condition materially extends predictability | Establishes a bounded null result and prevents attributing apparent gains to meteorology without sufficient origin information. | YES | “No material incremental value was detected under the evaluated conditions and support.” |
| D. Error-based predictability extends but fidelity/events do not | Shows why meteorological gains require complementary interpretation. | YES | “Error-based gains were not uniformly accompanied by the evaluated dynamic/event diagnostics.” |

Because B, C and D remain informative, the story does not depend on a model-win outcome. This supports preserving the question while withholding an OPEN decision until the availability evidence is verified.

## 10. Minimum evidence still required

Already present:

- frozen P3 `lags_only` condition;
- 323-station daily PM10 panel and common-support protocol;
- persistence baseline;
- horizon-wise evaluation design;
- Phase 3 error, fidelity and event source tables;
- a documented retrospective-versus-operational availability contract.

Still missing or unresolved:

- a provenance-complete `lags_meteo_retrospective` condition aligned to the frozen P3 panel;
- a provenance-complete `lags_meteo_operational` condition, if such forecasts exist, including source, issue time, cycle, lead, latency, version and station assignment;
- identical origins, horizons and common-support keys across conditions;
- an external literature verification of whether this exact availability-controlled comparison is already established;
- only those dynamic/event diagnostics that materially alter interpretation, without adding a new ghost-skill cutoff.

No further rank-reversal or ghost-skill adjudication should be treated as opening the P3 paper story before these conditions are resolved.

## 11. Claims currently defensible

- The audited corpus contains PM10/air-quality meteorology comparisons, but the legacy project comparisons are explicitly retrospective or insufficiently documented for operational availability.
- P3 has a frozen multistation lags-only reference against which a future meteorological increment could be evaluated.
- Origin availability is a scientifically consequential distinction because retrospective and operational information conditions need not estimate the same incremental value.
- Dynamic-fidelity and event diagnostics can be used as secondary interpretation checks on any meteorological gain.
- The current corpus does not establish a literature-wide absence claim.

## 12. Claims currently prohibited

- “No previous study distinguishes meteorological availability by forecast origin.”
- “Retrospective meteorology systematically overstates operational predictability.”
- “Operational meteorology improves PM10 predictability.”
- “Meteorology creates ghost skill” or that dynamic fidelity is the primary P3 novelty.
- “The horizon or persistence-regime effect is a P3 contribution independent of the existing P1/P2 work.”
- Any universal environmental-forecasting claim based on the current P3 lags-only results.

## 13. External verification still required

No external literature search was performed in this kill test. The project corpus is not sufficient to support an external-negative claim. Before OPEN, verify against current primary literature:

1. whether PM10/PM2.5 studies compare retrospective meteorology with genuinely origin-available forecast meteorology;
2. whether issue time, forecast cycle, latency and spatial assignment are reported;
3. whether such comparisons use identical origins and horizons;
4. whether dynamic fidelity or event representation is jointly assessed;
5. whether the surviving P3 contribution is already covered under another terminology.

## 14. Final portfolio decision

PARK

The availability-controlled question is not killed, but it is not ready to become the paper story. Existing work already covers meteorology-versus-persistence, horizon dependence, persistence-regime contrast, and error--fidelity disagreement in overlapping lines. P3 should remain parked until the external literature check and an origin-available meteorological provenance audit determine whether N1/N3 survive.

No Phase 3B execution was performed. No forecast, metric, manuscript, or Overleaf file was modified.
