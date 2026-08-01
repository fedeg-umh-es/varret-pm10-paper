# P4 documentary closeout report

## 1. Commit base and scope

- Repository: `/Users/fede/repos/varret-pm10-paper`
- Documentary branch: `codex/p4-documentary-closeout`
- Canonical producer/base commit: `f57f076078760af8a88bd87815fdf94ab0064fa3`
- Base subject: `Reproduce and finish Paper A without temporal leakage (#5)`
- Closeout date: 2026-08-01 (Europe/Madrid)
- Empirical execution: none
- P1 reactivation: no

The branch was created directly from the locally available canonical commit. No
pull, fetch, reset, rebase, merge, commit, or push was performed.

## 2. Files modified or created

| Path | Action | Scope |
|---|---|---|
| `paper_a.tex` | Modified | Wording guard, persistence missing-origin clarification, SARIMA selection-provenance disclosure, Data and Code Availability |
| `README.md` | Modified | Canonical P4 entry point and explicit legacy/non-canonical boundary |
| `config/config.yaml` | Modified | Legacy/non-canonical banner only; configuration values unchanged |
| `docs/empirical_reproducibility_audit.md` | Modified | Upstream-evidence boundary and correction of stale legacy-preprocessor description |
| `docs/p4_canonical_provenance_manifest.json` | Created | Canonical commit, commands, protocol, configurations, hashes, legacy boundary, and P1 independence |
| `docs/p4_documentary_closeout_report.md` | Created | This closeout record |

No prediction, metric, event table, result table, scientific figure, producer
script, dependency lock, raw datum, or processed datum was modified.

## 3. Manuscript wording changes

The edits preserve all empirical values and narrow only the interpretation:

- Abstract wording now describes loss of amplitude/episode information rather
  than declaring forecasts operationally uninformative (`paper_a.tex:27-30`).
- Small positive RMSE skill is no longer called operational value; it is framed
  as insufficient evidence of preserved amplitude or event-detection fidelity
  (`paper_a.tex:44-48`).
- “Useful at medium leads” is replaced by the directly supported statement
  “more accurate than persistence” (`paper_a.tex:80-88`).
- Event metrics are explicitly a case-study event-detection complement, not a
  utility analysis (`paper_a.tex:116-120,430-447`).
- Persistence now records the producer's actual missing-origin behavior: the
  most recent causally available validated observation is used and no future
  value enters (`paper_a.tex:255-267`).
- SARIMA is a fixed diagnostic-case specification, not a tuned or selected
  winner (`paper_a.tex:280-290`).
- Fold-training thresholds are described as avoiding threshold fitting on
  verification outcomes; the text explicitly declines to guarantee stable base
  rates (`paper_a.tex:479-485`).
- Data and Code Availability now pins the producer commit, authoritative script,
  provenance manifest, row-level predictions, and reproducible dependencies
  (`paper_a.tex:523-538`).

The terms “ghost skill” and Skill$_{VP}$ remain bounded exactly as before: the
former is informal shorthand, and the latter is an auxiliary diagnostic rather
than a universal replacement. The manuscript does not identify the variance
ratio as `alpha_KGE` and does not add inferential superiority language.

## 4. SARIMA provenance decision

The canonical producer fixes `(1,0,1)(1,0,0)[24]`, trend `c`, no stationarity or
invertibility enforcement, fit per fold, and sequential state updates without
within-fold parameter refitting (`scripts/run_paper_a_empirical.py`).

The historical audit names private repository
`fedeg-umh-es/P1_PM10_Meteorology_Hstar` at commit
`54569072ae525974838a84950cb0e31cf83cd5a2`, but:

- that repository is not present under the bounded local repository roots;
- that commit is not an object in either the P4 or local P1 repository;
- the documented upstream prediction artifact differs in model family,
  horizon grid, and protocol and is explicitly not the source of P4 results;
- no inspectable upstream configuration or pre-results selection contract is
  locally available.

Accordingly, the manuscript no longer claims that the order was recovered from
an upstream configuration. Its pre-result selection is recorded as
**not verifiable**, while the executed fixed specification remains fully
reproducible inside P4. P1 supplies no P4 prediction, metric, table, or figure.

## 5. Hashes verified

All hashes below were recalculated from the working tree after branching and
before documentary edits. They are recorded in
`docs/p4_canonical_provenance_manifest.json`; an automated post-edit cross-check
validated 22 manifest entries with zero size or SHA-256 mismatches.

| Artifact | SHA-256 |
|---|---|
| `data/raw/madrid_air_hourly_2023.zip` | `b3ee481e0a787239dd07b33e93b2da97e31e6b5123d3c659f49e14549fb62b2e` |
| `data/processed/casa_de_campo_pm10_2023.csv` | `c27fcef79c33c04d3c1af8d7e5994a97f9f8f970e185641c844db33205eda738` |
| `scripts/run_paper_a_empirical.py` | `918e0f7768e272881fbca30120ce68937b7a429853bdcb4c7c410efe6186c12b` |
| `outputs/reproduction/predictions_rolling_origin.parquet` | `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4` |
| `outputs/reproduction/metrics_rolling_origin.csv` | `5d7022b08d762f8f5c61cb70e6c00e85eee95580684a407f62152ae5d27e0583` |
| `outputs/reproduction/events_p75_rolling_origin.csv` | `78847a7bf2895af58b9515097b7cc235dce40adfab3fbacb5896520722ce9013` |
| `outputs/reproduction/predictions_holdout.parquet` | `e8a62e2550ba2b91e25e8bad04a283cfb19c424ef90608d8d6f2042a2d015454` |
| `outputs/reproduction/metrics_holdout.csv` | `73342c638575cd6e14028fd72c23c86f981380f71999016083c333b1a1cc5a3f` |
| `outputs/reproduction/events_p75_holdout.csv` | `e7dfbd1054cc62a773b8869d88ca6599edb46c6d9e90e67cc7c80cc289ecfb8e` |
| `outputs/tables/paper_a_rolling_results.tex` | `f5f6499301b011cc045b91bab7f5391916ae3f88eda47b8d43863beec1f4e010` |
| `outputs/figures/figure1_skill_variance.png` | `7d136802725f052a6de340301f8db026f8b37e9a978a6358dfdc060ba893bbd8` |
| `outputs/figures/figure2_skillvp_events.png` | `527a6cf6b16674cb47bda6af110c9c9631e28fa92ea04741b43ba48af6944c1d` |
| `requirements-reproduction.txt` | `82462f75445db7e5af65b06cdb26cb278367438db46ce0b902f897187b3cbd06` |

The SHA-256 of the manuscript's scientific numeric-token sequence from the
abstract through the conclusion is unchanged at
`d8a4cc5dcc24ebc7fdb942a597bbd9378e3a551c5d436544415a74e002b56e1f`.

## 6. Legacy elements marked

`README.md` now identifies the daily P33/E1-RR question, scope, data contracts,
pipeline, and execution instructions as legacy/non-canonical for P4. It also
identifies the canonical hourly producer and corrects the stale statement that
row-level hourly predictions were not distributed.

`config/config.yaml` now carries an explicit legacy banner and warns that its
SARIMA configuration differs from the canonical hourly specification. Values
were intentionally retained unchanged.

The canonical manifest additionally classifies historical daily P33,
meteorology, KGE, H*, multi-station, and model-family pipelines/outputs as
non-canonical unless expressly named by the manifest. No legacy file was
deleted.

## 7. Absence of recomputation or experimental execution

The closeout used only Git inspection, text search, file reads, SHA-256/size
checks, JSON syntax validation, and documentary file edits. It did not run:

- `scripts/recover_madrid_pm10.py`;
- `scripts/run_paper_a_empirical.py`;
- `scripts/render_paper_a_results.py`;
- `src/plotting/plot_master_figure.py`;
- `make data`, `make reproduce`, `make figures`, `make paper`, or tests;
- any P1 or P3 command.

Neither `latexmk` nor `pdflatex` was available in `PATH` or the standard MacTeX
binary path. No dependency was installed. Consequently `paper_a.pdf` was left
unchanged as required and does not yet contain the documentary source edits;
packaging in an existing LaTeX environment may compile `paper_a.tex` directly,
without invoking the Makefile target that regenerates figures.

## 8. Diff summary and repository state

The working tree contains only the six authorized documentary paths listed in
Section 2. `git diff --check` reports no whitespace errors. Restricted diffs
confirm zero changes under:

- `outputs/reproduction/`;
- `outputs/tables/`;
- `outputs/figures/`;
- the canonical producer/plotting scripts;
- `requirements-reproduction.txt`.

The local P1 repository remains at
`245b68388ab9cfa34f9c253611a5318aa3d344f5` with a clean working tree. No commit
or publication action has been taken in P4.

## 9. Decision

`P4: DOCUMENTARILY CLOSED`

The central empirical contribution is traceable to a self-contained P4 package,
its claims are bounded to the available evidence, P1 is not a result dependency,
and all remaining differences are documentary packaging rather than scientific
execution. The uncompiled PDF is explicitly recorded as a packaging note, not
an unresolved empirical or methodological dependency.
