# P4 — Ghost Skill & Dynamic Fidelity

Version: 1.3  
Last updated: 2026-08-01  
Status: P4_DOCUMENTARY_CLOSEOUT_VALIDATED  
Scientific disposition: CLOSED_NO_GO  
Canonical file: P4_PROJECT_CANON.md
Numerical-invariance decision: [[P4_Numerical_Invariance_Hash_Contract_Decision]]

---

## 1. Source of truth

This file is the only persistent source of truth for P4.

No claim about:

- ghost skill;
- variance collapse;
- 17 stations;
- rank reversal;
- events;
- extremes;

is canonical unless supported by committed row-level evidence or explicitly classified as provisional.

---

## 2. Canonical identity

### Name

P4 — Ghost Skill & Dynamic Fidelity

### Central research question

> Can a model improve average error relative to persistence while losing amplitude, variability, extremes or operational usefulness?

### Scientific object

The object is the decoupling between:

- error-based skill;
- dynamic fidelity;
- variance retention;
- event representation;
- operational utility.

### Core thesis

Positive RMSE skill is insufficient evidence of a useful forecast.

A model may reduce average error by producing over-smoothed predictions that fail to preserve the variability and events that matter.

This phenomenon is called:

```text
ghost skill
```

---

## 3. Canonical definition

Ghost skill is present when:

1. a model has positive error-based skill relative to a baseline;
2. the prediction exhibits materially degraded dynamic fidelity;
3. the degradation is operationally or scientifically relevant.

Possible manifestations:

- variance collapse;
- reduced amplitude;
- underestimation of maxima;
- loss of temporal correlation;
- poor exceedance detection;
- rank reversal after adding fidelity diagnostics.

Ghost skill is an empirical diagnostic concept, not a universal physical law.

---

## 4. Mandatory diagnostics

Continuous-error metrics:

- MAE;
- RMSE;
- bias;
- skill relative to persistence.

Dynamic-fidelity metrics:

- variance retention;
- standard-deviation ratio;
- alpha-KGE;
- correlation;
- amplitude ratio;
- temporal variability;
- peak retention.

Event metrics:

- hit rate;
- false alarm rate;
- precision;
- recall;
- CSI;
- event bias;
- exceedance intensity error.

Structural diagnostics:

- rank reversal;
- horizon dependence;
- station heterogeneity;
- model-family comparison.

---

## 5. Skill_VP

`Skill_VP` may be used only as an auxiliary composite diagnostic.

It must not be presented as:

- a universal forecast metric;
- a replacement for RMSE;
- a regulatory score;
- a standalone proof of usefulness.

Any composite score must be accompanied by its components.

The primary evidence remains:

- error skill;
- variance/amplitude fidelity;
- event performance;
- horizon-wise behaviour.

---

## 6. Evidence hierarchy

### Grade A — Complete evidence

- row-level predictions;
- observed values;
- model and baseline;
- station;
- origin;
- horizon;
- fold;
- reproducible metrics;
- source tables;
- scripts;
- manifests.

### Grade B — Partial evidence

- aggregate metrics;
- some station-level outputs;
- scripts but incomplete predictions;
- figures with partly reproducible source tables.

### Grade C — Manuscript ahead of evidence

- multi-station claims without row-level outputs;
- figures without source tables;
- results embedded only in prose or scripts;
- local-only files;
- unrecoverable predictions.

Strong claims require Grade A.

---

## 7. Canonical 17-station rule

No claim about 17 stations is defensible without:

- all-station predictions;
- station metadata;
- variance-retention outputs;
- event metrics;
- source tables for figures;
- reproducible aggregation.

If only three series have complete raw evidence, the paper must not imply that all 17 are equally reproducible.

Canonical classification:

```text
17-station package:
Grade C for strong claims
Grade B for pipeline intent
```

This classification may change only after a remote evidence audit.

---

## 8. Novelty threshold

Weak framing:

> Machine-learning forecasts underestimate variability.

Potentially strong framing:

> Error-based skill and dynamic fidelity can rank models differently, and this rank reversal changes conclusions about operational usefulness across horizons and stations.

The paper must show that ghost skill:

- is systematic;
- changes model ranking or interpretation;
- appears across relevant stations, horizons or model families;
- affects extremes or decisions;
- is not a single anecdotal plot.

---

## 9. Boundaries with other lines

P4 may use P1 rolling-origin evaluation and H*.

P4 may use P2 to determine whether skill is explainable through linear memory.

P4 may analyse P3 meteorological models for dynamic fidelity.

P4 must not absorb:

- the primary H* methodological contribution;
- operational-availability design;
- Aurora as a standalone benchmark line;
- unverified multi-station claims.

---

## 10. Aurora relationship

Aurora Air Pollution is a high-value P4 case because gridded forecasting may show:

- good average error;
- smooth regional structure;
- damped local maxima;
- loss of station-level variability;
- missed exceedances;
- grid-to-station rank reversal.

The Aurora pilot must test whether:

- gridded skill survives at stations;
- variance is retained;
- hotspots are preserved;
- event skill remains useful;
- the ranking changes when fidelity metrics are introduced.

Aurora is not evidence for P4 until prediction-level station comparisons exist.

---

## 11. Current scientific state

Conceptual strength:

```text
HIGH
```

Evidence status:

```text
CONDITIONAL GO
```

Main blocker:

Multi-station claims exceed the committed row-level evidence.

Current priority:

- preserve reproducible subsets;
- recover or regenerate all-station outputs;
- align claims with evidence grade;
- avoid manuscript-first expansion.

---

## 12. Required canonical outputs

At minimum:

```text
predictions_row_level
metrics_by_station_horizon
variance_retention_by_station_horizon
event_metrics_by_station_horizon
rank_reversal_table
station_metadata
run_manifest
figure_source_tables
```

Every figure must have a committed source table.

---

## 13. Role allocation

### ChatGPT / Claude

- assess whether the phenomenon is scientifically consequential;
- calibrate novelty;
- review claim strength;
- design figure narrative;
- distinguish diagnosis from physical explanation.

### Codex

- compute fidelity metrics;
- generate event metrics;
- detect rank reversal;
- create source tables;
- execute reproducible analysis.

### Claude Code

- audit evidence completeness;
- verify all-station claims;
- confirm files are committed;
- identify manuscript–repository divergence.

### Overleaf

- write only from validated tables;
- avoid inflating station count;
- distinguish empirical result from interpretation.

---

## 14. Promotion criteria

P4 becomes full GO when:

- row-level evidence exists for the claimed station set;
- figures are regenerated from committed tables;
- ghost skill changes a consequential conclusion;
- event consequences are quantified;
- station heterogeneity is reported;
- claims are bounded to the evaluated corpus and protocol.

---

## 15. Current priority

P4 documentary closeout is validated. The active front is the P2 closure
audit. P3 remains deferred until P2 is closed or explicitly released.

---

## 16. Next minimum action

Audit whether P2 is closed or explicitly released before resuming P3. This
does not authorize any new P4 experiment or manuscript expansion.

---

## 17. Update log

### 2026-08-01 (Version 1.3)

- **Documentary status changed to P4_DOCUMENTARY_CLOSEOUT_VALIDATED**:
  decision [[P4_Numerical_Invariance_Hash_Contract_Decision]] retires the
  undocumented historical `d8a4cc…` control and adopts the fully specified
  replacement contract yielding `cfd238…` for 187 identical numeric tokens.
- **Scientific disposition remains CLOSED_NO_GO**: the documentary contract
  correction changes no empirical result and does not reopen experiments,
  manuscript expansion or claim escalation.
- **Project sequence**: audit or close P2 before resuming P3.

### 2026-07-30 (Version 1.2)

- **Status changed to CLOSED_NO_GO**: The ghost-skill hypothesis remains scientifically plausible and descriptively supported in limited cases, but its multi-station decisional benefit has not been demonstrated with Grade A prospective evidence.
- **Core decision**: No further experiments, manuscript expansion, or claim escalation are authorized. P4 may only be reopened if a genuinely prospective evaluation block and all external decision inputs become available before outcome inspection.

### 2026-07-28 (Version 1.1)

- **Status changed to HOLD**: Paused submission to *Environmental Modelling & Software* due to a traceability discrepancy (missing `variance_retention_all_stations.csv`, full predictions table for the 17 stations, and non-regenerable Figure 5).
- **Core decision**: Pause submission until one of the five recovery/resolution paths is completed.
- **Priority**: Resolving Paper A's evidence trace is the absolute priority of the program; must be completed before starting Paper B or resuming P3.

---

## 18. Numerical-invariance documentary contract

Decision `2026-08-01-p4-numerical-invariance-hash-contract` is accepted and
recorded in [[P4_Numerical_Invariance_Hash_Contract_Decision]].

The historical expected hash
`d8a4cc5dcc24ebc7fdb942a597bbd9378e3a551c5d436544415a74e002b56e1f`
is retained only as the provenance of an obsolete documentary control. Its
tokenizer, text range, normalization, serialization, encoding and source commit
were not documented, so the original contract cannot be reproduced.

The replacement contract selects `paper_a.tex` from the literal
`\begin{abstract}` marker through, but excluding, the literal
`\section*{Data and Code Availability}` marker; extracts numeric tokens with
the Python regular expression `[+-]?(?:\d+\.?\d*|\.\d+)`; preserves textual
order; joins tokens with ASCII commas; encodes the serialization as UTF-8; and
computes SHA-256.

For base commit `f57f076078760af8a88bd87815fdf94ab0064fa3`, documentary
closeout commit `390685f1f1312954ee67513f3e0db11b2670e7f9` and validation
correction commit `11f8c88e5e3860eee75a333c83382127c360a930`, the contract yields
187 identical tokens and digest
`cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3`.

This is a documentary contract correction. It changes no empirical result,
manuscript number, data, table, figure, scientific code or protected result
artefact. It does not recover the unknown historical contract.
