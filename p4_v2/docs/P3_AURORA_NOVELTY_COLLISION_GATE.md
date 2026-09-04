# P3 Aurora Prior-Art Collision / Novelty Kill Test

Audit date: 2026-08-16
Mode: `SCIENTIFIC GAP AUDIT`
No Aurora data were downloaded, no forecasts or metrics were executed, and no
manuscript or Overleaf artifact was modified.

## Executive verdict

`AURORA_OLD_PRIMARY_GAP = DEAD`.

The original grid-to-station/smoothing thesis is directly collided by recent
Aurora evaluations using monitoring observations, gridded references, PM10/PM2.5
and episode-level local-variability analysis. A 2026 study also reports a
large monitoring-station evaluation including Europe and a CAMS operational
comparison, although its full text was not independently retrieved in this
bounded search. The Puebla study separately establishes PM10/PM2.5 comparison
against persistence and dichotomous event verification.

The only residual question with plausible scientific content is narrower:
whether Aurora changes the preferred forecast relative to strong local
station-level models when all forecasts share identical station-origin-horizon
support. That is not the preserved old gap, and on current evidence it remains
weak/incremental rather than an authorization to open a new experiment.

## 1. Prior-art basis

### A — Li, Chen and Yao (2026), Frontiers in Environmental Science

The paper evaluates five-day retrospective Aurora forecasts over China for
PM2.5, PM10 and other pollutants, using hourly CNEMC monitoring observations
and gridded CHAP products. It harmonizes the sources, collocates gridded
fields to stations, and reports MAE, RMSE, bias, correlation and spatial
similarity. The paper explicitly finds stronger agreement with gridded
structures than with local variability, and reports smoothing of high
concentration episodes, abrupt maxima and localized hotspots. Its East China
episode comparison with WRF-Chem supplies a process-model comparator but not a
local statistical forecast baseline.

Source: [Frontiers full text](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1882322/full).

### B — Li, Chen, Huang and Xu (2026), Environmental Research Letters

The supplied citation and the accessible institutional publication record
identify a year-long retrospective global/regional Aurora PM2.5 evaluation with
3,540 monitoring stations, Europe included, and CAMS operational-forecast
comparison. The article full text and a stable article identifier were not
retrieved in this bounded web pass; those details are therefore marked
`PARTIAL` rather than treated as independently verified line-by-line.

Source: [institutional publication record](https://hg.iap.ac.cn/paper/211.html).

Even at this evidence level, the study directly collides with claims that
Aurora has not been evaluated at European monitoring stations or against an
operational atmospheric-composition system. Those claims cannot be preserved.

### C — Castillo-Miranda et al. (2025), Atmósfera

The study evaluates CAMS against persistence for PM10 and PM2.5 at five Puebla
monitoring stations. It uses contingency/dichotomous metrics including POD,
FAR and related scores, and reports persistence as generally better while CAMS
remains usable for some PM2.5 event comparisons.

Sources: [journal record](https://www.revistascca.unam.mx/atm/index.php/atm/article/view/53408), [full PDF](https://www.scielo.org.mx/pdf/atm/v39/0187-6236-atm-39-53408.pdf).

### D — Bodnar et al. (2025), Aurora foundation-model paper

The original Aurora paper already positions Aurora relative to operational
systems and explains its CAMS-based air-pollution fine-tuning context. It is
background prior art, not evidence for station-level local-model comparison.

Source: [Nature article](https://www.nature.com/articles/s41586-025-09005-y).

## 2. Claim-by-claim kill table

| Claim | Prior art | Collision strength | What is already known | What remains unanswered | Status | Maximum defensible position |
|---|---|---|---|---|---|---|
| A1 grid advantage may not survive station evaluation | A, B | STRONG | Aurora has already been evaluated against monitoring stations and compared with gridded references | Whether a new local-model comparison changes preference under stricter common support | DEAD | Prior studies show grid/station differences; no originality claim |
| A2 Aurora may smooth local peaks/hotspots | A | DIRECT | Peak flattening, damped maxima and localized-hotspot loss are reported | Whether the effect changes Aurora-vs-local-model preference in a new controlled design | DEAD | Treat smoothing as replicated limitation, not discovery |
| A3 insufficient evaluation against independent stations | A, B | STRONG | CNEMC station evaluation and the reported 3,540-station study directly address this | Exact independence/provenance details for B require full-text verification | DEAD | Do not claim absence of station evaluation |
| A4 no European station evaluation | B | DIRECT/PARTIAL | Accessible record plus supplied study description place Europe in a large station evaluation | Exact European subset and protocol need primary-text verification | DEAD | Europe cannot be sold as an untouched evaluation domain |
| A5 no operational atmospheric-composition comparison | B, D | STRONG | B is reported to compare CAMS operational forecasts; D establishes operational-system context | Exact support and lead alignment in B require full-text verification | DEAD | Do not claim no operational-system comparison |
| A6 persistence comparison may change utility | C | MODERATE | PM10/PM2.5 station utility can change against persistence; event metrics can disagree with continuous performance | Aurora-specific three-way comparison with persistence remains unverified | WEAK | A future study could test Aurora relative to persistence, but this is incremental |
| A7 local station models may change preference | A, C | PARTIAL | A uses WRF-Chem and station observations; C uses persistence, not local statistical/ML models | Aurora versus strong station-specific models under shared support is not closed | MODERATE | This is the strongest residual candidate, pending full prior-art verification |
| A8 error ranking may differ from event/amplitude ranking | A, C | PARTIAL | A reports amplitude/episode limitations; C reports dichotomous event verification; neither closes the exact Aurora/local ranking contrast | Whether preference changes between Aurora and local models on identical support | MODERATE | A bounded preference-discordance question may remain |
| A9 same-support comparison may change operational interpretation | A, B, C | PARTIAL | Existing studies align forecasts and observations to varying degrees, but the exact Aurora/persistence/local common-support decision is not established | Whether support-controlled comparison changes the preferred forecast | WEAK | Methodological rigor matters only if it changes preference; no claim before evidence |

## 3. No combination novelty

The package `Aurora + European stations + persistence + local models + variance
+ events` is not novel merely by accumulation. Components such as station
verification, persistence baselines, CAMS comparison, amplitude/episode
diagnostics and dichotomous metrics are already represented in the audited
prior art. A publishable contribution would require a non-obvious change in
the forecast preference or in the interpretation of Aurora's station utility,
not a larger table of familiar metrics.

## 4. Decision-friction test

The strongest remaining formulation is:

> Under identical station-origin-horizon support, does comparison against persistence and strong local models change the conclusion about whether Aurora is the preferable PM10/PM2.5 forecast?

The dependent secondary formulation is:

> If error-based preference favors Aurora, do event or amplitude diagnostics change that preference?

This is not the old grid-to-station question. It is a controlled model-preference
comparison. It could be scientifically consequential if Aurora wins against
persistence but loses to a local model, or if the preferred model changes under
event/amplitude criteria. However, the need for a common-support design is not
itself novelty, and no result currently demonstrates that such a change occurs.

## 5. Negative-result test

| Outcome | Scientific information gained | Prior-art coverage | Publishable value | Depends on Aurora winning? |
|---|---|---|---|---|
| A Aurora remains preferable to persistence/local models | Establishes station-level utility relative to stronger baselines | Partially covered in broad benchmark logic; not closed for Aurora/local models | Useful but incremental unless preference is robust and decision-relevant | YES, for a positive utility story |
| B Aurora beats persistence but not strong local models | Separates global/grid advantage from local forecast utility | Not directly closed | Potentially useful as a bounded model-adequacy result | NO |
| C Aurora loses to persistence at important supports | Shows the gridded foundation-model signal does not transfer to the tested station use case | Not directly closed | Potentially publishable if support/provenance are strong, but likely a validation result | NO |
| D Error favors Aurora while event/amplitude favors another forecast | Demonstrates a preference conflict that changes forecast interpretation | Components are known; exact Aurora/local consequence is not closed | The strongest residual scientific outcome | NO, but requires a clear preference change |
| E All criteria agree | Establishes no diagnostic conflict in the tested support | Compatible with prior art | Mostly incremental benchmarking | NO |

Negative results remain informative, but none automatically creates a new
method or a general Aurora theory. If only outcome D were considered valuable,
the gap would be weak and dependent on a particular diagnostic conflict.

## 6. European-specificity test

`EUROPE_ADDS_SCIENTIFIC_FRICTION = NO` for the preserved question.

Europe is not a novelty claim by geography. A European design could become
scientifically informative only with a pre-specified contrast involving local
versus regional behaviour, atmospheric regimes, or station-network structure.
The recovered Aurora question does not specify such a contrast, and the cited
2026 prior art already reports Europe within a broad station evaluation.

## 7. Local-model baseline test

`LOCAL_MODEL_COMPARISON_GAP = OPEN`.

The audited Aurora studies do not clearly establish a direct comparison against
station-specific SARIMA/tree/ML forecasts on identical support. This is the
strongest residual gap, but absence of that comparator is not novelty by itself.
Its scientific value would come only from changing the conclusion about which
forecast is preferable at a station and horizon, especially when persistence
and event/amplitude diagnostics are included.

## 8. Same-support test

`SAME_SUPPORT_GAP = PARTIAL`.

The China study documents temporal/spatial harmonization and station/grid
collocation; the Puebla study uses common station observations for CAMS and
persistence. The exact four-way key requirement — station, origin, horizon and
target — across Aurora, persistence and a local model is not established in the
audited prior art. This remains a reproducibility condition, not a contribution
unless it changes the substantive model preference.

## 9. Desk-reject test

`DESK_SCREEN_CLASS = WEAK`.

An editor would likely read the preserved story as an incremental station-level
Aurora validation or a new-geography benchmark because recent work already
covers station observations, gridded-vs-local behaviour, peak smoothing,
European coverage, CAMS comparison, persistence and event metrics. The
principal desk-reject risk is that the local-model/common-support contrast does
not produce a distinct scientific consequence beyond a more careful benchmark.

## 10. Final portfolio decision

`PORTFOLIO_DECISION = PARK`.

The original Aurora question is killed as a novelty basis. A much narrower
local-model/common-support preference question remains technically open, but it
does not currently satisfy the threshold for opening Aurora as an independent
scientific line. It must not be rescued by adding metrics, enlarging geography,
or presenting the same-support protocol as novelty.

`AURORA_EXPERIMENT_AUTHORIZED = NO`.

## 11. Next minimum action

Keep Aurora parked and do not authorize an experiment; any future revival must
first define a narrower local-model preference question and pass a dedicated
full prior-art verification of that question.
