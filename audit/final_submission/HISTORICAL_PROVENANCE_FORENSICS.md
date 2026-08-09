# P4 — Historical Provenance Forensics

Audit date: 2026-08-10

## Scope and preservation rule

This was a read-only forensic search for historical artifacts that could
close the current Paper-A SARIMA provenance gap. No model was fitted, no
forecast was regenerated, no metric was recomputed for publication, and no
manuscript, figure, publication table, or canonical aggregate was modified.

The canonical local integrity check passed 16/16 before and after the
search. The frozen contract remains 595 rows, five models, 17 stations,
seven horizons, zero LightGBM rows, Rule A = 277, Rule B = 8, changes = 269,
97.1%, 101 formal discordant cases, and rho(alpha, skill) = -0.863.

The governing specification was
`docs/p4_legacy_closeout/P4_PROJECT_CANON.md`, version 1.4 dated 2026-08-08.
Its Grade-A definition requires row-level predictions, observed values,
model and baseline, station, origin, horizon, fold, reproducible metrics,
source tables, scripts, and manifests. A strong 17-station claim also
requires all-station prediction evidence and a reproducible aggregation
chain.

## Search scope

The following locations were inspected non-destructively:

- current repository working tree, all reachable refs, deleted paths, and
  unreachable Git objects;
- public `github` refs and the previously audited `paper-a-row-level-evidence-v1`
  release;
- sibling `P4_Ghost_Skill_Dynamic_Fidelity` and the provenance clone of the
  current repository;
- the sibling `pm10-valencia-experiments` repository, including its legacy
  artifacts and benchmark checkpoint directories;
- `pm10-forecasting-validation`, P1/P3 PM10 repositories, and their archived
  SARIMA/prediction outputs;
- archived and provenance directories within the scoped research root
  `03_Investigacion/repos/`.

No unrelated home-directory search was performed.

## Candidate inventory

| Candidate | Location | Historical provenance | Commit/date | Rows | Stations | Horizons | `y_true` | `y_pred` | Persistence | Origin | Fold | Protocol identity | SARIMA exact match | Status |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|
| Public all-station release | GitHub release `paper-a-row-level-evidence-v1/predictions_all_stations.csv` | Public immutable release; SHA-256 documented as `2551b3a2acf549e94cd92e39386b96aca2358cca41fed7f85c933d35c77ef823` | tag `4e3a97a`, release previously audited | 895,737 data rows | 17 | 7 | yes | yes | partial for SARIMA support | yes | yes, date-valued | incomplete current manifest; combined persistence starts later than some SARIMA origins | no; 5,253 pre-2020 SARIMA rows lack the observed origin support needed to derive persistence | **PARTIAL_CANDIDATE** |
| Valencia-Vivers legacy predictions | `outputs/metrics/predictions_valencia_vivers.csv`; copied byte-identically to `pm10-valencia-experiments/legacy/predictions/valencia_vivers_predictions.csv` | Historical file in the current repository; sibling provenance manifest records byte identity to source commit `30cd774...` | source history `3cafd4c` (2026-05-23), revised in `e2120f8` (2026-05-25); blob `d8feaa...` at `e2120f8` | 52,028 | 1 | 7 | yes | yes | yes, as separate persistence rows | yes | date-valued origin field named `fold`; sibling report describes five folds but raw schema has no `fold_0`–`fold_4` identifiers | historical E1-RR output, not a complete 17-station package | no; maximum SARIMA skill difference 0.030246 and alpha difference 0.032923 against the canonical station subset | **PARTIAL_CANDIDATE / CONFLICTING** |
| Zarra-EMEP legacy predictions | `outputs/metrics/predictions_zarra_emep.csv`; copied byte-identically to `pm10-valencia-experiments/legacy/predictions/zarra_emep_predictions.csv` | Historical file in the current repository; sibling provenance manifest records byte identity to source commit `30cd774...` | source history `3cafd4c` (2026-05-23), revised in `e2120f8` (2026-05-25); blob `cd1eaf...` at `e2120f8` | 52,136 | 1 | 7 | yes | yes | yes, as separate persistence rows | yes | date-valued origin field named `fold`; sibling report describes five folds but raw schema has no `fold_0`–`fold_4` identifiers | historical E1-RR output, not a complete 17-station package | no; maximum SARIMA skill difference 0.194947 and alpha difference 0.067358 against the canonical station subset | **PARTIAL_CANDIDATE / CONFLICTING** |
| Elche legacy predictions | `outputs/metrics/predictions.csv`; copied to `pm10-valencia-experiments/legacy/predictions/elche_predictions.csv` | Historical file in the current repository and byte-identical sibling copy | source history `e7e40fd` (2026-05-22) and later provenance copy | 26,001 | 1 | 7 | yes | yes | yes | yes | no explicit fold identifier | historical E1-RR output | not applicable; no SARIMA rows | **PARTIAL_CANDIDATE** for non-SARIMA only |
| E0 benchmark checkpoints | Untracked `pm10-valencia-experiments/outputs/benchmark/checkpoints/` | Current sibling benchmark output, not a preserved source export for Paper A; README and code explicitly identify it as an independent E0 re-execution | sibling working tree at `a1de587` (2026-08-09); checkpoint files are untracked | about 11–12k per station/model file | 3 | 7 | yes | yes | yes, separate files | yes | explicit fold-support files exist, but they describe the E0 execution | independent fixed-ex-ante E0 protocol; legacy comparison notes state that SARIMA origin support differs from the historical output | not tested as a current-Paper-A candidate because protocol identity is explicitly different | **WRONG_PROTOCOL** |
| P4 provenance clone / Ghost Skill sibling | `_provenance_clones/varret-pm10-paper/` and `P4_Ghost_Skill_Dynamic_Fidelity/` | Duplicates of the current repository's historical artifacts and scripts | same historical source lineage | same as current local files | at most 3 raw stations | 7 | partial | partial | partial | yes where present | partial | no additional source lineage | no new SARIMA evidence | **DUPLICATE / PARTIAL** |
| Other PM10 SARIMA projects | `pm10-forecasting-validation`, P1/P3, and related clones | Independently traceable projects with their own protocols | historical project commits | varied | different station sets | varied | often yes | often yes | often yes | yes | project-specific | monthly/H* or hourly/Madrid-Ireland protocols, not current Spanish 17-station Paper A | not applicable to current benchmark | **WRONG_PROJECT / WRONG_PROTOCOL** |

## Strong-candidate validation

The two historical station files containing SARIMA were validated without
refitting. Their separate persistence rows provide direct row-level pairing,
and their `origin_date`, target `date`, horizon, and `y_true`/`y_pred` fields
are present. The exact pooled persistence-relative skill and alpha values were
then compared with the corresponding canonical station/model/horizon values.

The comparison failed exact reconciliation for both stations:

- Valencia-Vivers: maximum absolute SARIMA skill difference
  `0.03024600996500519`; maximum absolute alpha difference
  `0.03292254905419098`.
- Zarra-EMEP: maximum absolute SARIMA skill difference
  `0.19494657951909722`; maximum absolute alpha difference
  `0.06735823120932102`.

These discrepancies are provenance conflicts, not permission to select a
different support, refit SARIMA, infer an order, or tune a calculation. The
files therefore remain useful historical partial evidence but cannot serve as
the missing canonical source for the 119 SARIMA station-horizon cells.

The public all-station release was also checked for the minimum support needed
to derive persistence from the observed value at the SARIMA origin. It
contains 19,312 SARIMA rows, of which 13,315 have an observed row at the same
origin date and 5,997 do not. All 5,253 SARIMA rows with origins before 2020
lack that support in the release. This confirms the previously documented
release limitation; it does not close it.

## Git and historical-object findings

Reachable history contains the three historical station prediction files and
their scripts, but no `skill_sarima_{station}.csv` exports or 17-station
SARIMA-aligned persistence files. Deleted historical paths found in the
current repository were manuscript metric tables, not missing SARIMA
prediction sources. Unreachable commits and trees were inspected by path;
they retain the same older outputs and conflicting hourly/LightGBM-era
artifacts, not a complete current-protocol SARIMA source.

The public tag and release likewise contain no `skill_sarima_{station}.csv`
files. The release README explicitly identifies those unreleased files as the
source that would be needed for exact SARIMA skill recovery.

## Recovery assessment

| Required current-Paper-A component | Recovered from historical evidence | Assessment |
|---|---|---|
| Exact SARIMA predictions for all 17 stations | No; SARIMA row-level files are available only for two historical stations in the relevant legacy set | **OPEN** |
| Exact SARIMA-aligned persistence for all 119 cells | No; public release lacks pre-2020 origin support, and the two local station files conflict with canonical aggregates | **OPEN** |
| Exact five-fold/origin mapping for all 17 stations | Partial for the three-station legacy set; not established for the current 17-station benchmark | **OPEN** |
| Current-protocol manifest | Current audit manifest documents the conflict, but no historical manifest reconciles the complete 17-station chain | **PARTIAL** |
| End-to-end historical chain | Deterministic partial reproduction exists for 42 legacy summary cells in the sibling repository; it does not cover current 17-station SARIMA evidence | **OPEN** |

No candidate meets the definition of an exact historical source for the
current 17-station Paper-A SARIMA benchmark. No replacement pipeline or
synthetic provenance was created.

## Verdict

`HISTORICAL_EVIDENCE_PARTIALLY_FOUND`

Authentic historical row-level evidence was found and materially narrows the
provenance gap for three stations, including SARIMA and persistence pairing for
two stations. It does not close the required 17-station SARIMA baseline,
origin/fold, and exact canonical-reconciliation gap.

Therefore:

```text
CURRENT_17_STATION_5_MODEL_PAPER_CANNOT_REACH_GRADE_A_FROM_RECOVERED_HISTORICAL_EVIDENCE
```

The next governance option is:

```text
SCIENTIFIC_REVIEW_OF_REMAINING_GAP
```

This audit did not perform the later maximum-reproducible-subset audit, did
not alter the paper, and did not publish or replace any evidence package.
