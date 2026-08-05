# P4 Exceedance & Rank-Reversal Diagnostic Module

- **Project**: P4 — Ghost Skill & Dynamic Fidelity
- **Role**: auxiliary exceedance and rank-reversal diagnostic
- **Evidence status**: REAL_DATA_UNVERIFIED until the producer pipeline is audited

This module does **not** belong to P1, P2, or P3. It is not, by itself,
evidence of ghost skill. P4's central object is whether a model keeps
positive skill relative to persistence while losing dynamic fidelity,
extreme-event representation, or operational usefulness. This module
supplies one input to that question — exceedances, hit rate, false alarm
rate, precision, recall, CSI, event bias, exceedance intensity error, and
rank changes between continuous and event metrics — and nothing here
should be cited as a self-sufficient ghost-skill proof.

This module is intentionally kept out of the existing P33/E1-RR pipeline
scope described in `docs/e1_rr_post_evaluation_contract.md`. It reads
row-level prediction tables and does not modify, extend, or depend on the
E1-RR variance-retention code path.

## Traceability note

This code was **not** ported from an existing file. The repository owner
described three external source artifacts for this task:

- `evaluacion_excedencias_pm10.py` (original module)
- `Modulo_Python_Evaluacion_Excedencias_y_Rankings_PM10.md` (original documentation)
- a prior technical audit (`audit_report.md`, verdict `CONDITIONALLY_COMPLIANT`,
  34 tests passing, blockers B1/B2, secondary issues S4/S5)

All three live outside this repository, on the owner's local machine, and
were not reachable from the environment this module was written in (a
sandboxed remote container with no access to the owner's filesystem). The
owner explicitly chose, when asked, to have this module **written fresh
from the detailed functional specification** they provided (schema
adapter behaviour, `ranking_comparison` contract, duplicate/alignment
checks, classification labels, threshold-sweep labeling, bootstrap
requirements, manifest fields, and the 17 required test scenarios) rather
than block on transferring the original files.

Consequences of this:

- This is **not** a line-for-line port. Function names, internal
  structure, and any implementation details of the original module are
  unknown here.
- The "34 tests, all pass" figure from the prior audit describes the
  *original* module, not this one. This module ships its own test suite
  (see `tests/p4_exceedance/`), built from the same 17 required scenarios,
  and its pass count is reported independently in
  `audit_followup_report.md` — the two numbers are not comparable and
  should not be conflated.
- B1 (schema mismatch) and B2 (`ranking_comparison` crash on
  no-comparison input) are fixed here because the specification described
  the exact failure mode; there was no need to see the original buggy
  code to fix behavior it never had.

If the original files become available in the repository or are pasted
into a session, they should be diffed against this implementation and any
behavioral divergence should be reconciled explicitly and documented, not
silently merged.

## Layout

```
src/evaluation/p4_exceedance/
    schema_adapter.py       # Fase 3 / B1: date -> target_date adapter, resolution checks
    contract.py              # Fase 9: row-level input contract, evidence classification
    integrity_checks.py      # Fase 5 / S4 / S5: duplicate detection, case alignment
    ranking_comparison.py    # Fase 4 / B2: continuous-vs-event ranking comparison
    classification.py        # Fase 6: YES/NO/TRADE_OFF_ONLY/... classifier
    metrics.py                # exceedance contingency metrics (reuses compute_event_metrics)
    threshold_sweep.py       # Fase 7: regulatory / diagnostic / calibrated thresholds
    bootstrap.py              # Fase 8: contiguous block bootstrap
    manifest.py                # Fase 11: evaluation_manifest builder
    evaluation_manifest.template.json

tests/p4_exceedance/         # Fase 10 test suite (17 required scenarios + more)

scripts/p4_run_exceedance_diagnostic.py   # CLI: synthetic demo + real-data probe

outputs/p4_exceedance/
    demo_synthetic/           # DEMO_SYNTHETIC run outputs
    real_probe_elche_equivalent/  # REAL_DATA_UNVERIFIED probe outputs (if run)
```

## Schema adapter (B1)

Real prediction tables in this repository
(`outputs/metrics/predictions*.csv`) use `date` for the forecast target
and `origin_date` for the issue time — not `target_date`. `adapt_schema`
performs an explicit, logged adaptation:

1. Uses `target_date` if present.
2. Otherwise uses `target_time` if present, or interprets `date` as the
   target time and records that interpretation in the adaptation report.
3. Converts `origin_date`/`origin_time` to datetime.
4. Validates `target_date > origin_date` for every row and that `horizon`
   is a strictly positive integer for every row.
5. Infers temporal resolution (`daily` vs `hourly`) by checking, across
   **every** row, whether `target_date == origin_date + horizon * 1 day`
   or `target_date == origin_date + horizon * 1 hour`. If neither holds
   uniformly (or the caller declares a resolution that turns out
   incoherent with the data), it raises `SchemaAdapterError` — it never
   guesses.
6. Never mutates the input DataFrame or touches the source CSV.

## `ranking_comparison` (B2)

The original bug was `pd.DataFrame([]).sort_values("horizon")` raising
`KeyError` whenever there was nothing to compare (0 or 1 model, or no
comparable horizon). This implementation always builds its output with an
explicit column list
(`station, horizon, metric_continuous, metric_event, kendall_tau,
kendall_pvalue, n_models, n_pairs, n_reversals, evaluation_status`), so the
degenerate case returns a 0-row, correctly-typed DataFrame instead of
throwing. `evaluation_status` is one of `EVALUATED`,
`NOT_EVALUABLE_SINGLE_MODEL`, `NOT_EVALUABLE_NO_COMMON_CASES`,
`NOT_EVALUABLE_INSUFFICIENT_MODELS`, `NOT_EVALUABLE_INCOMPLETE_RANKING`.
Ties are handled with Kendall tau-b (`scipy.stats.kendalltau(..., variant="b")`).

## Duplicates and case alignment (S4, S5)

`integrity_checks.detect_duplicates` flags exact duplicate rows by key
(`station, model, origin_date/origin_time, target_date, horizon[,
fold_id]`) and reports counts, affected keys, models, and stations — it
never drops rows silently.

`integrity_checks.check_common_support` checks, per `station × horizon [×
fold_id]`, that every model was evaluated on exactly the same set of
cases. Misaligned groups are reported with a per-model missing-case table
and must **not** be resolved by silently intersecting case sets.
`ranking_comparison` is told which groups are misaligned and reports
`NOT_EVALUABLE_NO_COMMON_CASES` for them without computing Kendall tau. A
separate, explicitly-named `COMMON_SUPPORT_SENSITIVITY` mode can compute
the intersection as a clearly labeled sensitivity view — it is never the
default and never silent.

## Classification (Fase 6)

`classification.classify_reversal` returns one of `YES`, `NO`,
`TRADE_OFF_ONLY`, `NOT_TESTED`, `NOT_EVALUABLE`, `UNCLEAR`. A conflict
between event sub-metrics (e.g. better POD but worse FAR, with no
predefined single operational ranking) is detected separately via
`detect_event_submetric_conflict` and takes precedence over a
continuous-vs-event reversal verdict — it is reported as
`TRADE_OFF_ONLY`, never auto-promoted to `YES`.

## Threshold policy (Fase 7)

Three call paths are kept structurally distinct and cannot collapse into
each other:

- `regulatory_threshold_result`: a fixed threshold (`threshold_mode="FIXED"`),
  usable as a primary estimate.
- `diagnostic_sweep`: scans thresholds against the evaluation data itself;
  **always** labeled `threshold_mode="POST_HOC_DIAGNOSTIC"` and
  `usable_as_primary_estimate=False`, regardless of which threshold looks
  best.
- `calibrated_threshold_result`: selects a threshold from calibration data
  only. It raises if `calibration_period` does not end at or before
  `evaluation_period` begins, so a test-window selection can never be
  mislabeled as a valid calibration.

## Bootstrap (Fase 8)

`bootstrap.block_bootstrap` builds moving blocks over a sorted,
contiguity-checked `origin_date` axis (`require_contiguous=True` by
default), with a configurable `block_length` (default **14, provisional**)
and a reproducible `random_seed`. The default `block_length_justification`
is the literal string `PROVISIONAL_DEFAULT_NOT_JUSTIFIED_BY_ACF_OR_EPISODE_DURATION`,
and the result carries a non-empty `warning` in that case. Confidence
intervals produced with the unjustified default must not be presented as
scientifically validated.

## Row-level contract and evidence status (Fase 9)

`contract.validate_contract` checks a table against the canonical
row-level schema (`station, model, horizon, y_true, y_pred`, plus one
origin-timestamp column and one target-timestamp column; `fold_id`,
`baseline`, `producer_repository`, `producer_commit` optional) and lists
what is missing without inventing it.

`contract.classify_evidence_status` returns `DEMO_SYNTHETIC` (caller
declares synthetic data), `REAL_DATA_UNVERIFIED` (default for real data),
or `REAL_DATA_AUDITED` — the last one only when a complete
`ProducerEvidence` record is supplied (rolling-origin protocol,
train-only preprocessing, explicit baseline, producer repository,
producer commit, dataset, station, period, fold). No probe run in this
repository currently has that evidence, so every real-data run stays
`REAL_DATA_UNVERIFIED`.

## Manifest (Fase 11)

`manifest.build_manifest` fixes `project` and `analysis_role` and fills
all other fields with `None` or `"PENDING_VERIFICATION"` unless the
caller supplies a value. See
`src/evaluation/p4_exceedance/evaluation_manifest.template.json`.
