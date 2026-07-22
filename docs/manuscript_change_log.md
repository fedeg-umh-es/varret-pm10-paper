# Manuscript change log

## 2026-07-22: complete editorial preparation

### Scientific and methodological clarification

- Kept rolling-origin evaluation as the canonical evidence and relabelled the
  80/20 holdout as protocol sensitivity.
- Added exact fold construction, causal missing-input handling, no-scaling
  policy, and same-row persistence comparison.
- Added LightGBM feature/hyperparameter details and SARIMA specification.
- Added a methods subsection defining train-fold P75 events, reported metrics,
  and undefined precision when no positive flags occur.
- Expanded limitations to cover single-site/year external validity, fixed model
  configurations, lack of uncertainty testing, non-regulatory P75 thresholds,
  and unvalidated Skill_VP behavior outside this case.

### Editorial changes

- Reworked abstract opening and conclusion to avoid equating error skill with
  operational value.
- Removed promotional or universal language around models and Skill_VP.
- Replaced “exceedance” with “P75 event” where regulatory confusion was
  possible.
- Distinguished 8,760 hourly grid points from 8,465 valid observations.
- Clarified that SARIMA precision at 48 h is undefined because flag rate is
  zero.
- Updated captions and Data and Code Availability.

### Reproducibility and build changes

- Fixed `scripts/render_paper_a_results.py` to emit valid LaTeX row terminators.
- Added `scripts/check_paper_a_consistency.py` and a pytest wrapper covering
  canonical cells, generated table equality, matched row-level verifications,
  required/forbidden manuscript claims, undefined precision, and figure files.
- Removed unused LaTeX packages and added section float barriers.
- Made `make test` use `python3 -m pytest` and gave matplotlib a writable cache.
- Rewrote the README around the hourly Paper A source of truth while explicitly
  isolating retained daily legacy modules.

### Bibliography

- Verified all nine cited entries against authoritative metadata.
- Removed three uncited entries (`murphy1988`, `harvey1997`, `taylor2001`).
- Updated `docs/reference_audit.md` with evidence and citation-integrity counts.

### Empirical results

No canonical empirical result, prediction, metric CSV, or Parquet artifact was
changed.  Figures and the LaTeX table were regenerated from committed canonical
outputs.
