# P3 External Novelty and Operational Meteorology Provenance Gate

Audit date: 2026-08-16  
Project: P3  
Repository: `P4_Ghost_Skill_Dynamic_Fidelity`  
Branch: `codex/p4-documentary-closeout`  
HEAD: `2f1d2552c25bc6c1aab8b9cf72bf1cbe0ab29560`

This is an adversarial gate. It does not authorize a new experiment, modify the
frozen lags-only benchmark, or update the manuscript.

## Executive decision

The exact P3 question remains plausible but is not externally verified as a
distinct literature gap. The corpus contains strong adjacent prior art: operational
air-quality systems using forecast meteorology, meteorological ablations, horizon-wise
comparisons, and accuracy-versus-event/variability diagnostics. The specific contrast
between retrospective meteorology and genuinely origin-available forecast meteorology
on identical station-origin-horizon support was not identified in the screened sources,
but that absence is not proof of novelty.

The current P3 repository cannot support `lags_meteo_operational`: it contains the
frozen 323-station lags-only PM10 benchmark and protocol documentation, but no retained
meteorological forecast files with issue time, initialization time, valid time, lead,
latency, version, and station mapping. The legacy Madrid/Ireland meteorology line is
explicitly retrospective or provenance-incomplete for operational availability.

Therefore the contribution is parked. No further forecasting, interpretation, or
manuscript work should proceed until an operational product and its historical
availability provenance are secured.

## 1. Research question and information conditions

The only paper-level question tested here is:

> Does meteorological information provide incremental predictive value for PM10/PM2.5
> when the meteorological inputs are restricted to information genuinely available at
> each forecast origin?

The permitted secondary question is whether any operational meteorological increment
also preserves dynamic fidelity or event representation. Ghost skill, variance
collapse, H* theory, generic verification, and model competition are not independent
P3 novelty claims.

| Condition | Meaning | Status in current P3 |
|---|---|---|
| `lags_only` | Pollutant lags, calendar, and origin-known information | Frozen and executed |
| `lags_meteo_retrospective` | Observed/reconstructed meteorology, potentially unavailable at origin; upper-bound condition | Legacy P2/P3 materials only; not current P3-v2 |
| `lags_meteo_operational` | Forecast meteorology with source, issue/init time, valid time, lead, latency, version and spatial mapping | Not available in current P3 |
| `oracle` | Future observed meteorology; non-operational upper bound | Not permitted as an operational condition |

Unknown availability is not operational availability.

## 2. External search scope and evidence rule

The bounded search covered peer-reviewed or primary technical sources from 2015--2026,
with emphasis on PM10/PM2.5 forecasting, meteorological predictors, CAMS/NAQFC/NWP
systems, multi-horizon validation, peak/event verification, and forecast availability.
Ten external studies or primary systems were screened in the accompanying coverage
matrix. Methodological details not reported by a source are `UNKNOWN`, not `NO`.

Primary sources reviewed include:

* Bertrand et al. (2023), CAMS European air-quality forecasting with machine-learning
  correction, *Atmospheric Chemistry and Physics*:
  <https://acp.copernicus.org/articles/23/5317/2023/index.html>
* Huang et al. (2017), NOAA NAQFC PM2.5 bias correction, *Weather and Forecasting*;
  NOAA repository record: <https://repository.library.noaa.gov/view/noaa/28100>
* FuXi-Air, *npj Clean Air* (2026):
  <https://www.nature.com/articles/s44407-026-00061-w>
* X2-AQFormer, *npj Clean Air* (2026):
  <https://www.nature.com/articles/s44407-026-00058-5>
* Hisar PM2.5/PM10 machine-learning study, *Scientific Reports* (2026):
  <https://www.nature.com/articles/s41598-026-60752-y>
* CAMS global forecast documentation, ECMWF:
  <https://www.ecmwf.int/en/forecasts/datasets/cams-global-atmospheric-composition-forecasts>
* CAMS Regional forecast documentation, ECMWF/Copernicus:
  <https://confluence.ecmwf.int/spaces/CKB/pages/202173092/CAMS%2BRegional%2BEuropean%2Bair%2Bquality%2Banalysis%2Band%2Bforecast%2Bdata%2Bdocumentation>
* ECMWF IFS Open Data and forecast-cycle documentation:
  <https://www.ecmwf.int/en/forecasts/datasets/open-data>
* Aurora model documentation, including Aurora 1.5 and Aurora Air Pollution:
  <https://microsoft.github.io/aurora/models.html>

The search does not justify a statement that no prior study has addressed the exact
question. It justifies the narrower statement that no exact matched comparison was
identified in the screened corpus.

## 3. Established components that are not novel by themselves

The following are rigor requirements or established verification practices, not a
stand-alone P3 contribution: rolling-origin validation; chronological splitting;
leakage-safe preprocessing; persistence baselines; MAE/RMSE and skill scores;
variance-retention or standard-deviation diagnostics; event metrics; multidimensional
verification; and rank comparisons.

The combination `meteorology + H* + fidelity + events` is also not sufficient as a
novelty claim. The legacy P2/P3 Madrid--Ireland line already contains lags-versus-
meteorology comparisons, persistence-relative horizon analysis and a persistence-regime
contrast. The P4 line already contains skill-versus-dynamic/event disagreement. P3 can
remain distinct only if controlling information availability changes the scientific
interpretation of the meteorological increment.

## 4. External coverage and absence test

The full row-level extraction is in:

`p4_v2/docs/P3_EXTERNAL_LITERATURE_COVERAGE_MATRIX.csv`

Summary of the adversarial test:

| Question | Result | Interpretation |
|---|---|---|
| Are meteorological predictors used in PM10/PM2.5 forecasting? | VERIFIED | Common in the screened studies and operational systems. |
| Do operational AQ systems use forecast meteorology? | VERIFIED | CAMS/ECMWF and NAQFC documentation/system papers establish this practice. |
| Is meteorological availability separated by actual forecast origin in the reported experiments? | PARTIALLY_VERIFIED | Operational system schedules exist, but predictor-level issue/latency provenance is often not reported in the experimental papers. |
| Is retrospective meteorology formally compared with operational forecast meteorology on identical support? | NOT_VERIFIED | No exact example was identified in the screened sources. This is not a literature-wide negative claim. |
| Are horizon effects studied? | VERIFIED | Multi-horizon forecasting and the existing P2/P3 H* line cover this. It is not an independent P3 novelty. |
| Are accuracy and event/variability behaviour jointly discussed? | PARTIALLY_VERIFIED | Huang et al. and FuXi-Air provide adjacent examples; P4 provides direct project overlap. |
| Is the exact P3 gap closed by the screened corpus? | NOT_VERIFIED | The exact matched availability-controlled comparison was not located, but the search is bounded. |

## 5. Good practice versus scientific gap

Restricting predictors to information available at forecast origin is first a necessary
leakage-control and deployment-validity requirement. It becomes a scientific gap only
if the availability distinction produces a different empirical conclusion about the
incremental value of meteorology.

That empirical consequence is not currently available. Thus:

`METEOROLOGICAL_AVAILABILITY_IS_ONLY_GOOD_PRACTICE = UNCERTAIN`

The correct present position is not that availability control is novel, and not that it
is merely hygiene. The distinction is a testable factor whose scientific status remains
unresolved until a provenance-complete comparison is performed.

## 6. Kill test by candidate claim

The detailed claim table is in:

`p4_v2/docs/P3_NOVELTY_CLAIM_KILL_TABLE.csv`

| Claim | Status | Kill-test conclusion |
|---|---|---|
| N1. Availability changes estimated meteorological value | `NOT_VERIFIED` | Plausible and consequential, but no matched external result or current P3 result establishes the change. |
| N2. Retrospective meteorology overstates operational value | `NOT_VERIFIED` | The conditions are conceptually distinct, but overstatement is not demonstrated. |
| N3. Meteorological value is horizon-dependent | `DEAD` as independent novelty | Already substantially covered by the Madrid--Ireland/P2 H* line and multi-horizon prior art. |
| N4. Operational meteorology extends predictability over lags-only | `NOT_VERIFIED` | Operational systems exist, but the P3 station-level comparison has not been executed with verified provenance. |
| N5. Skill gains need not preserve dynamics/events | `DEAD` as independent novelty | Covered by P4/P2 overlap and adjacent operational AQ literature; can remain a secondary diagnostic. |

## 7. Closest prior art and project overlap

The ranked comparison is in:

`p4_v2/docs/P3_CLOSEST_PRIOR_ART.csv`

The most important overlap is internal to the project: the Madrid and Ireland P2/P3
meteorology line already compares lags-only against concurrent meteorology, uses
persistence and rolling-origin/horizon-wise diagnostics, and interprets differences by
persistence regime. Its availability contract explicitly warns that the concurrent
meteorology is not certified operational without issue-time/latency evidence. P4 adds
the error--fidelity/event disagreement layer, but that is not a separate P3 novelty.

External prior art is also substantial. CAMS and NAQFC show that operational air-quality
forecasting with meteorological forcing is an established system practice. FuXi-Air
shows that meteorology ablation can alter accuracy and peak behaviour. These sources do
not close the exact matched availability comparison, but they make broad claims about
meteorological value, multi-horizon forecasting, or event consequences non-novel.

`EXISTING_PAPER_OVERLAP = HIGH`

## 8. Operational meteorology provenance audit

The candidate-source inventory is in:

`p4_v2/docs/P3_OPERATIONAL_METEOROLOGY_PROVENANCE.csv`

Official documentation shows that technically usable products exist. CAMS Regional
documentation specifies a daily 00 UTC run, 0--96 h forecasts, a 0.1-degree ensemble
grid, and published availability guarantees. CAMS Global documentation describes twice-
daily production and forecasts up to five days. ECMWF IFS documentation provides forecast
cycles and dissemination schedules. This establishes feasibility of documenting an
operational source, not possession of a P3-ready historical archive.

Aurora must remain split. Aurora 1.5 is a weather model using IFS HRES T0 data; Aurora
Air Pollution is an air-pollution model intended for CAMS analysis inputs. Aurora Air
Pollution is therefore not automatically a meteorological predictor source, and neither
product currently supplies the historical station-level issue-time archive required by
P3.

Current repository finding:

* `p4_v2` contains only the frozen lags-only MITECO PM10 benchmark and its provenance.
* The P2/P3 legacy meteorological files are absent or explicitly marked retrospective,
  regenerated, or provenance-incomplete for operational availability.
* No P3-tracked NWP/CAMS/Aurora files with complete origin-level metadata were found.

Therefore:

`OPERATIONAL_METEOROLOGY_PROVENANCE = FAIL`

The failure is a current data/provenance failure, not a claim that operational products
do not exist.

## 9. Madrid--Ireland overlap

`MADRID_IRELAND_OVERLAP = HIGH`

Already answered by the existing line:

* meteorology versus lags-only can change PM10 forecast skill;
* the effect varies by horizon and station/regime;
* Madrid and Ireland provide a persistence-regime contrast;
* retrospective/concurrent meteorology is not automatically operational.

Still unanswered:

* whether an origin-available forecast product gives a different incremental estimate
  from the retrospective condition on identical support;
* whether that difference survives the current 323-station daily panel and the frozen
  availability contract.

Adding operational meteorology could therefore be a genuine extension, but not a new
paper identity by itself. Dynamic/event diagnostics would add interpretation only if
they change how the availability-conditioned meteorological result is understood.

## 10. Negative-result test

| Outcome | Information gained | Publishable insight without a model-win narrative? | Dependence on positive result |
|---|---|---|---|
| A. Operational meteorology extends skill | Quantifies an origin-safe incremental value and its horizon range | Yes, if provenance and common support are credible | No |
| B. Retrospective helps but operational does not | Demonstrates that retrospective upper-bound gains do not transfer to operational use | Yes; potentially the strongest management-relevant result | No |
| C. Both conditions behave similarly | Shows that the availability distinction did not alter the estimate in this design | Yes, as a bounded equivalence/null result if uncertainty is handled | No |
| D. Neither improves over lags-only | Prevents attributing apparent gains to meteorology and bounds the usefulness of the predictor stream | Yes, as a bounded negative result | No |
| E. Skill improves but fidelity/events do not | Shows why meteorological gains require complementary interpretation | Yes as a secondary diagnostic, not as the primary P3 novelty | No |

`NEGATIVE_RESULT_INFORMATIVE = YES`.

The gap is not dependent on a positive model-win outcome. Its value depends instead on
the availability comparison being real and reproducible.

## 11. Desk-screen assessment

`DESK_SCREEN_STORY = WEAK`

The story currently fails to answer, from the existing evidence, what changes because
of origin-availability control. An editor can see the problem and the adjacent prior
art, but cannot yet see a provenance-complete operational condition or a distinct result.
The story could become plausible only after the source and availability contract are
secured; it should not be sold as a new metric, a first study, a large benchmark, or a
generic combination of known diagnostics.

## 12. Minimum evidence still required

One minimum action is required before any further P3 experiment or interpretation:

> Secure and version one historical operational meteorology product for the P3 period
> with source/product/variable, issue or initialization time, valid time, lead time,
> publication latency, version, and station/grid assignment for every origin.

Only after that provenance gate passes may the already-defined P3 design be considered
for an availability-conditioned comparison using `lags_only`, retrospective meteorology
and, if the archive supports it, operational meteorology on identical origin/horizon
support. No new metric, threshold, model family, or ghost-skill rule is authorized by
this gate.

## 13. Current defensible and prohibited claims

Currently defensible:

* Origin-availability control is necessary to interpret meteorological predictors as
  operationally usable.
* The project has not yet demonstrated whether that control changes estimated PM10
  predictability.
* Retrospective meteorology is an upper-bound condition, not evidence of operational
  availability.
* The existing P2/P4 diagnostic layers motivate checking dynamic and event behaviour as
  secondary interpretation of any future meteorological increment.

Currently prohibited:

* “No previous study distinguishes meteorological availability by forecast origin.”
* “Retrospective meteorology systematically overstates operational predictability.”
* “Operational meteorology improves PM10 predictability.”
* “Meteorology creates ghost skill.”
* “Dynamic fidelity” or “ghost skill” is the primary P3 novelty.
* Claims that Aurora Air Pollution is an operational meteorological predictor source.

## 14. Final portfolio decision

`PORTFOLIO_DECISION = PARK`

The question is not killed outright because the exact matched comparison was not
identified and negative outcomes remain informative. It is parked because prior-art
overlap is high and the current P3 repository fails the operational meteorology
provenance gate. No experiment, Phase 3B interpretation, manuscript update, or Overleaf
change is authorized.

