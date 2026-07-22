# Paper A submission-readiness checklist

## Completed

- [x] `origin/main` verified at `f57f076`.
- [x] Canonical rolling-origin CSV and Parquet artifacts checked.
- [x] Abstract, Results, Discussion, and Conclusion aligned numerically.
- [x] LightGBM negative skill signs preserved.
- [x] SARIMA 48 h precision represented as undefined, not zero.
- [x] Candidate models and persistence compared on identical valid rows.
- [x] P75 thresholds documented as training-fold estimates.
- [x] Dataset attributed to Madrid Open Data, not EEA.
- [x] Old repository name absent from the manuscript.
- [x] Generated LaTeX table repaired and guarded by an automated check.
- [x] Both figures regenerated from canonical rolling-origin results.
- [x] All cited references verified; no undefined or orphan citations.
- [x] Generic manuscript language and claims reviewed line by line.
- [x] Data and Code Availability updated.
- [x] README aligned with the hourly Paper A reproduction.
- [x] Final clean PDF build and page-by-page visual inspection (8/8 pages).
- [x] Full test suite (22 passed), consistency check, and `git diff --check`.
- [x] Branch pushed and draft PR opened: [PR #6](https://github.com/fedeg-umh-es/varret-pm10-paper/pull/6).

## Pending author decision

- [ ] Confirm author spelling, affiliation, corresponding email, and ORCID.
- [ ] Approve title and bounded contribution statement.
- [ ] Decide whether to retain “ghost skill” as informal shorthand.
- [ ] Decide whether E2 blocked-holdout sensitivity stays in the main paper.
- [ ] Decide whether uncertainty estimates or formal loss tests are required
  before submission; they were not added because new experiments were out of
  scope.

## Pending *Atmospheric Research* preparation

- [x] Target identified in `codex_experiment_plan_atm_research.txt` as
  *Atmospheric Research* (Elsevier); its official Guide for Authors was checked
  on 2026-07-22.
- [x] Journal-fit risk documented: PM10 air-pollution forecasting is in scope,
  but the contribution threshold remains a human editorial judgment.
- [ ] Apply the current Elsevier LaTeX template and final reference style.
- [ ] Add the required title-page metadata after authorship is confirmed.
- [ ] Add required funding, conflict-of-interest, author-contribution, ethics,
  AI-use, and data declarations as applicable.
- [ ] Prepare highlights and graphical abstract if requested by the submission
  system; prepare a cover letter and any supplementary-material captions.

## Real blockers

No scientific or technical blocker is known for author review. Submission to
*Atmospheric Research* remains pending author metadata, declarations, and
template-specific packaging.
