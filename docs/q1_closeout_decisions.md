# Q1 Closeout Decisions

Date: 2026-06-01

## Decisions

- Manuscript source of truth: Overleaf.
- Target level: Q1.
- Paper identity: post-evaluation variance-retention diagnostic for PM10
  forecasting.
- Do not merge this paper with the H* cross-domain manuscript.
- Final empirical scope: all-stations PM10 diagnostic over 17 stations and 7
  daily horizons, using the canonical all-stations outputs under `outputs/`.
- Three-station material is explanatory only; it is not the closing evidence
  package.

## Core claim

Positive persistence-relative skill can coexist with substantial collapse of
forecast-trajectory variance. Therefore, horizon-wise skill should be reported
together with variance retention when operational interpretation depends on
dynamic variability.

## Must stay in the final paper

- Rolling-origin evaluation.
- Train-only preprocessing.
- Persistence baseline.
- Horizon-wise `Skill(h)`.
- Horizon-wise variance-retention ratio `alpha(h)`.
- `Skill_VP(h)` only as an auxiliary diagnostic.
- Explicit diagnostic thresholds and their non-universal interpretation.
- Operational relevance through exceedance/event behavior if supported by the
  existing artifacts.

## Do not add

- New model families for leaderboard expansion.
- Probabilistic forecasting or calibration claims.
- Cross-domain H* claims.
- Strong claims that variance retention alone validates or invalidates a model.
- Claims that abstract-level SLR findings prove full-text absence of good
  evaluation practice.

## Cleanup policy

- Use `outputs/figures/`, `outputs/tables/`, and `outputs/metrics/` as canonical
  artifact locations.
- Treat root-level figures and PDFs as Overleaf convenience copies.
- Remove absolute paths from scripts before submission.
- Keep `tmp/`, local caches, and downloaded inventory caches out of version
  control unless a specific file is part of the reproducibility package.
- Treat `scripts/p34_utils/`, `results/kge_*`, and `docs/companion_kge/` as
  companion or legacy material outside the main variance-retention closeout.
