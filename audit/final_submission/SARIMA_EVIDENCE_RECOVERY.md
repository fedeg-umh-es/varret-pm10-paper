# SARIMA evidence recovery audit

Audit date: 2026-08-10

## Recovery rule

This audit searched the repository working tree, Git history and public
GitHub refs/releases. No SARIMA model was refit, no historical order was
selected, and no missing prediction or baseline value was inferred from an
expected aggregate number.

The governing persistence implementation is the last causally available
observed PM10 value at the forecast origin. The historical SARIMA generator
implements this as the last non-missing training value and writes a matched
persistence row alongside each SARIMA row. That generator is code, not a
preserved prediction artifact.

## Candidate-source inventory

| Artifact | Location | Remote/local | Version/commit | Rows | Columns | Stations | Horizons | Origin | Fold | `y_true` | `y_pred` | Persistence | Exact matching demonstrated? | Status |
|---|---|---|---|---:|---|---:|---:|---|---|---|---|---|---|---|
| Combined row-level release | GitHub release asset `paper-a-row-level-evidence-v1/predictions_all_stations.csv` | Remote | tag `4e3a97a`; SHA-256 `2551b3a2acf549e94cd92e39386b96aca2358cca41fed7f85c933d35c77ef823` | 895,737 | `dataset,model,fold,origin_date,horizon,date,y_true,y_pred` | 17 | 7 | YES | YES | YES | YES | PARTIAL | NO for all SARIMA cells | PARTIAL: persistence rows cover the combined 2020–2024 base protocol, while SARIMA includes earlier origins. |
| Valencia-Viveros historical predictions | `outputs/metrics/predictions_valencia_vivers.csv` | Local and present in public Git history | `e2120f8`; predecessor `3cafd4c` | 52,028 | Eight row-level columns | 1 | 7 | YES | YES | YES | YES | YES | NO against canonical SARIMA aggregate | CONFLICTING: maximum SARIMA skill difference 0.03025 and alpha difference 0.03292. |
| Zarra-EMEP historical predictions | `outputs/metrics/predictions_zarra_emep.csv` | Local and present in public Git history | `e2120f8`; predecessor `3cafd4c` | 52,136 | Eight row-level columns | 1 | 7 | YES | YES | YES | YES | YES | NO against canonical SARIMA aggregate | CONFLICTING: maximum SARIMA skill difference 0.19495 and alpha difference 0.06736. |
| Daily local prediction table | `outputs/metrics/predictions.csv` | Local/public Git | `e7e40fd` | 26,001 | Eight row-level columns | 1 | 7 | YES | YES | YES | YES | YES | Not applicable | NO SARIMA rows; only HGB, Ridge and persistence. |
| Legacy long prediction table | `docs/p4_legacy_closeout/predictions_long_pm10.csv` | Local | historical closeout material | 26,001 | `target_date,model,horizon,y_true,y_pred,origin_date` | 1 | 7 | YES | NO explicit fold | YES | YES | YES | Not applicable | NO SARIMA rows and no explicit fold field. |
| Local raw PM10 data | `data/raw/pm10_daily.csv`, `pm10_valencia_vivers.csv`, `pm10_zarra_emep.csv` | Local/public Git | current repository | raw series | date/value files | 3 at most | n/a | Dates present | n/a | observed only | n/a | Derivable for these files only | NO for 17 stations | Insufficient station coverage; cannot reconstruct the missing 14 station histories. |
| SARIMA generator | `scripts/02_generate_sarima_predictions.py` | Local/public Git | current repository | code only | code | parameterised | 7 | YES | origin-valued fold | would generate | would generate | YES in code | NO without rerunning/refitting | Code-only source; execution is prohibited in this recovery. |
| SARIMA skill exports | `skill_sarima_{station}.csv` | Searched local tree, Git history and public refs | not found | — | — | 0 | — | — | — | — | — | — | NO | MISSING. The public release README identifies these as the source of exact SARIMA skill, but they are not released. |

## Persistence alignment test

The release asset contains 19,312 SARIMA rows. For each SARIMA row, the
asset was checked for an observed row whose target `date` equals the
SARIMA `origin_date`, which is the minimum information needed to recover
the persistence value from the release itself. Results:

- SARIMA rows: 19,312;
- rows with an observed target at the same origin date: 13,315;
- rows without such an observed target: 5,997;
- SARIMA rows with origins before 2020: 5,253;
- pre-2020 SARIMA rows lacking the required observed origin value: 5,253.

The public asset therefore does not contain the observed origin values
needed to derive the missing SARIMA persistence rows. The repository has no
17-station raw PM10 archive from which those values could be recovered.

The two local historical station files do contain persistence rows, but
their SARIMA-derived metrics do not exactly reconcile to the canonical
aggregate. They are historical candidates, not valid substitutes for the
17-station canonical source.

## SARIMA gate

`SARIMA_BASELINE_EXACTLY_RECOVERABLE = NO`

`SARIMA_BASELINE_NOT_EXACTLY_RECOVERABLE = YES`

The only way to close this gate without refitting is to recover the
historical SARIMA-matched persistence/prediction outputs used to create the
canonical 119 SARIMA cells, or equivalent per-station `skill_sarima` and
matched row-level source files with verifiable provenance.
