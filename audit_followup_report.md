# Audit follow-up report — P4 exceedance & rank-reversal diagnostic module

- **Project**: P4 — Ghost Skill & Dynamic Fidelity (not P1, P2, or P3)
- **Role**: auxiliary exceedance and rank-reversal diagnostic
- **Date**: 2026-08-05

## 1. Repo auditado

`fedeg-umh-es/varret-pm10-paper` (this repository). Not `eduintellect/varret-pm10-paper`
— that owner was mentioned in a later instruction but this session's GitHub
access is scoped to `fedeg-umh-es/varret-pm10-paper`, which was already
attached and on the correct branch; the user confirmed this repo when asked.

## 2. Rama y commit inicial

- Branch: `claude/p4-exceedance-module-lb14em` (pre-existing, already checked out).
- Commit at start of work: `2398565652227dedf0e6eaf1e0765242dc37545d`
  ("docs: refresh tracked EMS manuscript PDF", 2026-07-30). The branch had
  no diff against `main` at that point.

## 3. Archivos inspeccionados

Full repository listing (`find . -maxdepth 3 -type f`), plus targeted reads of:
`README.md`, `docs/e1_rr_post_evaluation_contract.md`, `docs/protocol.md`,
`pyproject.toml`, `requirements.txt`, `src/evaluation/compute_event_metrics.py`,
`scripts/39_rank_comparison_kge_vs_phi.py`, `audit/trazabilidad_tres_estaciones.md`,
and the schemas of `outputs/metrics/predictions.csv`,
`outputs/metrics/predictions_valencia_vivers.csv`,
`outputs/metrics/predictions_zarra_emep.csv`.

**The three external source files named in the task instructions —
`evaluacion_excedencias_pm10.py`, `Modulo_Python_Evaluacion_Excedencias_y_Rankings_PM10.md`,
and the prior `audit_report.md` — were never inspected.** They live on the
repository owner's local machine
(`/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/...`), and this
session runs in a sandboxed remote container with no access to that
filesystem. This was confirmed twice (path does not exist; `add_repo`
targets a different GitHub owner than this session's scope), and the owner
explicitly chose, when asked, to have this module **written fresh from
their detailed functional specification** rather than block on transferring
the files. See `docs/p4_exceedance_module.md` § Traceability note for the
full explanation. This is the single most important caveat in this report.

## 4. Veredicto

**CONDITIONALLY_COMPLIANT.**

Everything in scope for this task — B1, B2, S4, S5, the schema/contract/
manifest infrastructure, the classification and threshold/bootstrap
modules, the test suite, the synthetic demo, and the real-data probe — is
implemented, tested, and runs cleanly. The conditions are: (1) this is a
new implementation against a specification, not a verified port of the
audited original module, so the prior `CONDITIONALLY_COMPLIANT` verdict
and its "34 tests" do not transfer to this code; (2) the producer
pipeline's leakage / train-only-preprocessing audit is still pending, by
design (out of scope, per explicit constraints) — real-data evidence
stays `REAL_DATA_UNVERIFIED` until that happens.

## 5. B1 — schema mismatch (target_date vs. date)

**Fixed and tested.** `src/evaluation/p4_exceedance/schema_adapter.py::adapt_schema`:
accepts `target_date` directly if present; otherwise interprets `date` as
the forecast target time (or `target_time`) and logs the interpretation;
converts `origin_date`/`origin_time`; validates `target_date > origin_date`
row-wise, `horizon > 0` and integral row-wise; infers or validates daily
vs. hourly resolution against *every* row and raises `SchemaAdapterError`
(never guesses) when neither hypothesis holds uniformly. Never mutates the
input DataFrame. Verified against this repository's real files
(`outputs/metrics/predictions*.csv`, which use exactly `date` +
`origin_date`) — see § 11.
Tests: `tests/p4_exceedance/test_schema_adapter.py` (18 tests).

## 6. B2 — `ranking_comparison` crash on no-comparison input

**Fixed and tested.** `src/evaluation/p4_exceedance/ranking_comparison.py::ranking_comparison`
never calls `.sort_values` on a schema-less frame: an empty input (0 rows,
possibly 0 columns — the literal `pd.DataFrame([])` case) returns
immediately with the stable column set
(`station, horizon, metric_continuous, metric_event, kendall_tau,
kendall_pvalue, n_models, n_pairs, n_reversals, evaluation_status`); a
single model or zero models per group is reported via
`NOT_EVALUABLE_SINGLE_MODEL` / `NOT_EVALUABLE_INSUFFICIENT_MODELS` instead
of raising. Kendall tau-b (`scipy.stats.kendalltau(..., variant="b")`)
handles ties. `evaluation_status` also covers
`NOT_EVALUABLE_NO_COMMON_CASES` (fed from the S5 alignment check) and
`NOT_EVALUABLE_INCOMPLETE_RANKING` (a registered model missing a metric
value). This exact code path was exercised live: the real-data probe on
Valencia-Vivers and Zarra (6 models, one — `sarima` — with far fewer rows)
produced `NOT_EVALUABLE_NO_COMMON_CASES` for all 7 horizons instead of
crashing or silently mis-ranking. See § 13.
Tests: `tests/p4_exceedance/test_ranking_comparison.py` (10 tests), including
one that reproduces the literal historical crash
(`test_does_not_raise_keyerror_on_sort_values_horizon`).

## 7. S4 — duplicate detection

**Implemented and tested.** `integrity_checks.py::detect_duplicates` flags
exact duplicate rows by key (`station, model, origin_date/origin_time,
target_date, horizon[, fold_id]`), reports the count, the affected keys as
a DataFrame, and affected models/stations. It never drops rows. In the
CLI pipeline, any duplicate found forces the corresponding
`(station, horizon)` group(s) to `NOT_EVALUABLE_NO_COMMON_CASES` in
`ranking_comparison` rather than silently averaging over duplicated rows.
Tests: `tests/p4_exceedance/test_integrity_checks.py` (part of 11 tests).

## 8. S5 — case alignment across models

**Implemented and tested, including a bug found and fixed during live
execution.** `integrity_checks.py::check_common_support` checks, per
`station × horizon`, that every model was evaluated on exactly the same
case set (`origin_date/origin_time, target_date, horizon, fold_id` — fold
is part of the *case* key, not the group key, see below), and exports a
per-model missing-case table. A separate, explicitly-named
`COMMON_SUPPORT_SENSITIVITY` mode computes the intersection as a clearly
labeled sensitivity view; it is never silent and never the default.

**Bug found while running the real-data probe**: the first implementation
put `fold_id` in the *group* columns. Since `fold_id == origin_date` in
this repository's real data (one fold = one forecast origin), that made
every group trivially contain at most one case, so a model that was
**entirely** missing from a fold simply did not appear in that group
instead of being flagged as missing — a silent miss of exactly the
condition S5 exists to catch. Fixed by moving `fold_id` into the case-key
columns and rejecting `fold_id` as a group column
(`ValueError` with an explicit message). Regression test added:
`test_model_entirely_missing_from_one_fold_is_detected`. This is now
verified on real data: `sarima` in the Valencia-Vivers/Zarra probes has
~14% of the row count of the other five models, and is correctly
detected as misaligned for every horizon (see § 13).

## 9. Tests

| | count |
|---|---|
| Pre-existing repo tests collectible in this environment (excl. 2 files needing `lightgbm`, not installed — see § 18) | 15 |
| New tests (`tests/p4_exceedance/`) | 76 |
| **Total collected** | **91** |
| Pass | 91 |
| Fail | 0 |

Command: `python -m pytest tests/ -q --ignore=tests/test_empirical_protocol.py --ignore=tests/test_reproduction_artifacts.py` → `91 passed in 1.72s`.
The two ignored files fail to *collect* (not to run) because this
conservative-memory `.venv` only installs `pandas, numpy, scipy, pytest,
pyyaml` — not the full `requirements.txt` (`lightgbm`, `statsmodels`,
`scikit-learn`, `lightgbm`, `pyarrow`, `matplotlib`). This is unrelated to
this task's changes; it reproduces on a clean venv with the same minimal
install, before or after this branch's commits.

The 76 new tests cover all 17 Fase-10 scenarios (schema adaptation from
`date`; rejection of non-later `target_date`; daily coherence; hourly
coherence; ambiguous resolution; empty-schema single-model;
zero-comparable-models; duplicates; misaligned cases; no-common-support;
Kendall tau-b with ties; YES; NO; TRADE_OFF_ONLY; POST_HOC_DIAGNOSTIC
threshold sweep; reproducible bootstrap; contiguous ordered blocks) plus
additional coverage for the contract, manifest, and full event-metrics
modules.

## 10. Demo sintética

Ran: `python scripts/p4_run_exceedance_diagnostic.py demo`.
Two runs, both `DEMO_SYNTHETIC`:

- **`demo_synthetic_main`**: two synthetic stations, 3 models (incl.
  `persistence`), 360 rows. `north` shows no reversal (`model_a` wins both
  skill and CSI); `south` is engineered so `model_a` has higher skill but
  `model_b` has higher CSI — classified `YES`. Confirms the classifier
  distinguishes `YES` from `NO` end-to-end. Runtime 0.04 s.
- **`demo_synthetic_integrity_issues`**: 10 rows with one injected exact
  duplicate and one injected missing-case misalignment. Correctly detected
  (`n_duplicates=2`, `is_aligned=false`); `ranking_comparison` degrades to
  a 0-row, well-formed table (no persistence baseline in this toy set, so
  no metric could be computed) instead of crashing — this exercises the
  same code path as the historical B2 bug. Runtime 0.01 s.

Outputs: `outputs/p4_exceedance/demo_synthetic/` (and `.../integrity_issue_showcase/`).

## 11. Probe real

Ran: `python scripts/p4_run_exceedance_diagnostic.py probe`, against
`outputs/metrics/predictions.csv`, `predictions_valencia_vivers.csv`,
`predictions_zarra_emep.csv` — this repository's own real row-level
prediction tables. These are the in-repo equivalent of
`predictions_elche.csv`: `audit/trazabilidad_tres_estaciones.md` (a
prior, read-only, already-committed audit in this repo) identifies
`outputs/metrics/predictions.csv` as the Elche predictions table and
documents that `station` is not a literal column — `dataset` encodes it —
which is exactly the mapping used here. SHA-256 of `predictions.csv`
(`915a821c...58ee`) matches that prior audit's recorded hash.

All three runs completed with `evidence_status = REAL_DATA_UNVERIFIED`
(the script asserts this and would abort otherwise). Total runtime for all
three ≈ 1.1 s; peak RSS ≈ 177 MiB.

**This probe is not scientific evidence.** No leakage or train-only
preprocessing audit of the producer pipeline was performed. The threshold
used is a P75-of-evaluation-data diagnostic value, explicitly labeled
`POST_HOC_DIAGNOSTIC`, not a calibrated operational threshold. Elche's
"6 of 7 horizons classified YES" is a raw diagnostic output shown for
traceability, not a claim that rank reversal exists in Elche.

Outputs: `outputs/p4_exceedance/real_probe_elche_equivalent/{elche,valencia_vivers,zarra}/`.

## 12. Duplicados encontrados

Zero exact duplicate keys in any of the three real prediction tables
(`station, model, origin_date, target_date, horizon`) — confirmed both by
the CLI's `detect_duplicates` and by an independent pandas check during
inspection. `duplicate_report.csv` is present (empty, header-only) for
each real-probe run for traceability. The synthetic integrity-issue demo
confirms the detector fires correctly when duplicates *are* present
(2 flagged rows from 1 injected duplicate key).

## 13. Alineación de casos

- **Elche** (3 models: `hgb_direct`, `persistence`, `ridge_direct`, equal
  row counts): fully aligned, all 7 horizons `EVALUATED`.
- **Valencia-Vivers** and **Zarra** (6 models, incl. `sarima` with ~14% of
  the row count of the other five — e.g. Zarra: 1196 vs. 10188 rows):
  misaligned at every horizon. `ranking_comparison` correctly reports
  `NOT_EVALUABLE_NO_COMMON_CASES` for all 7 horizons in both stations
  instead of computing a Kendall tau on a mismatched case set.
  `case_alignment_table.csv` lists every missing case per model
  (11,362 rows for Valencia-Vivers, dominated by `sarima`'s much smaller
  case set relative to the other five models).

## 14. Estado del threshold sweep

Three modes implemented and unit-tested
(`src/evaluation/p4_exceedance/threshold_sweep.py`):
`regulatory_threshold_result` (`FIXED`), `diagnostic_sweep` (always
`POST_HOC_DIAGNOSTIC`, `usable_as_primary_estimate=False`), and
`calibrated_threshold_result` (`CALIBRATED`, raises if
`calibration_period` does not end at or before `evaluation_period`
begins). **Not wired into the CLI's demo/probe pipeline** — the CLI uses a
fixed P75-of-evaluation-data threshold directly, explicitly labeled
`POST_HOC_DIAGNOSTIC` in the manifest, rather than routing it through
`diagnostic_sweep`'s threshold-selection logic (there is nothing to
"select" for a single fixed percentile). The three functions are exercised
only by their own unit tests (9 tests), not by an end-to-end CLI run.

## 15. Estado del bootstrap

Implemented and unit-tested (`src/evaluation/p4_exceedance/bootstrap.py`,
9 tests): contiguous moving-block bootstrap over a sorted, contiguity-
checked `origin_date` axis, configurable `block_length`
(default **14, explicitly provisional**), reproducible `random_seed`.
`block_length_justification` defaults to the literal string
`PROVISIONAL_DEFAULT_NOT_JUSTIFIED_BY_ACF_OR_EPISODE_DURATION` and the
result carries a non-empty `warning` field in that case. **Not run** as
part of the demo or real probe (no ACF / episode-duration analysis exists
in this repository to justify a block length, so there is nothing
meaningful to bootstrap yet without producing a number that looks more
validated than it is). Exercised only by its own unit tests.

## 16. Etiqueta final de evidencia

- Demo: `DEMO_SYNTHETIC`.
- Real probe (all three stations): `REAL_DATA_UNVERIFIED` — asserted
  programmatically in the CLI, not just documented. `REAL_DATA_AUDITED`
  requires a complete `ProducerEvidence` record (rolling-origin protocol,
  train-only preprocessing, explicit baseline, repository, commit,
  dataset, station, period, fold); `rolling_origin_protocol` and
  `preprocessing_train_only` are deliberately left unset for the real
  probe, so `classify_evidence_status` cannot return `REAL_DATA_AUDITED`
  even though `producer_repository`/`producer_commit` are known.

## 17. Archivos modificados

New files only; nothing pre-existing was edited.

```
 .gitignore                                              |   6 +
 audit_followup_report.md                                | (new, this file)
 docs/p4_exceedance_module.md                             | (new)
 scripts/p4_run_exceedance_diagnostic.py                  | (new)
 src/evaluation/p4_exceedance/__init__.py                 | (new)
 src/evaluation/p4_exceedance/bootstrap.py                 | (new)
 src/evaluation/p4_exceedance/classification.py            | (new)
 src/evaluation/p4_exceedance/contract.py                  | (new)
 src/evaluation/p4_exceedance/evaluation_manifest.template.json | (new)
 src/evaluation/p4_exceedance/integrity_checks.py          | (new)
 src/evaluation/p4_exceedance/manifest.py                  | (new)
 src/evaluation/p4_exceedance/metrics.py                   | (new)
 src/evaluation/p4_exceedance/ranking_comparison.py        | (new)
 src/evaluation/p4_exceedance/schema_adapter.py             | (new)
 src/evaluation/p4_exceedance/threshold_sweep.py            | (new)
 tests/p4_exceedance/*.py (9 files)                        | (new)
 outputs/p4_exceedance/**  (demo + probe run artifacts)     | (new, generated)
```

`docs/e1_rr_post_evaluation_contract.md`, `P4_PROJECT_CANON.md` (does not
exist in this repo), Overleaf sources, and `paper2H` were not touched, per
the explicit constraints.

## 18. Elementos no verificables

- The original `evaluacion_excedencias_pm10.py`, its documentation, and
  its prior technical audit (34 tests, `CONDITIONALLY_COMPLIANT` verdict)
  — never accessed; see § 3.
- Absence of leakage in the producer pipeline for Elche /
  Valencia-Vivers / Zarra predictions — explicitly not audited or claimed.
- Whether real rank reversal exists in Elche or any station — the probe's
  `YES` outputs are raw diagnostic results on unaudited data, not a
  verified finding.
- `block_length=14` is not justified by any ACF / episode-duration
  analysis in this repository.
- Two pre-existing test files (`test_empirical_protocol.py`,
  `test_reproduction_artifacts.py`) could not be collected in this
  minimal `.venv` (missing `lightgbm`); this is a pre-existing environment
  characteristic, not something introduced by this work, and was not
  investigated further per the conservative-memory instruction.

## 19. Siguiente paso único

Diff and reconcile this implementation against the original
`evaluacion_excedencias_pm10.py` / its documentation / its prior audit
once those files can actually be transferred into a session with access
to this repository (paste into chat, or push to a repo this session can
read) — resolving any behavioral divergence explicitly rather than
merging silently, per `docs/p4_exceedance_module.md` § Traceability note.
