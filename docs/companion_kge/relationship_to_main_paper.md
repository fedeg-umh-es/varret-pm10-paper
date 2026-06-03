# Relationship To The Main Paper

## What The Companion Note Inherits

- The rolling-origin evaluation design.
- The 17-station PM10 benchmark.
- The five-model comparison surface.
- The seven forecast horizons.
- The canonical predictions table and master diagnostic table.
- The main paper's variance-retention framing and concern with operational interpretability.

## What The Companion Note Adds

- A horizon-wise KGE decomposition into `r_h`, `alpha_h`, and `beta_h`.
- An explicit empirical check that `alpha_h == phi_h`.
- Rank-comparison diagnostics between Skill, `phi_h`, `r_h`, KGE, and KGE skill.
- Robustness checks for ranking tie conventions, horizon breakdown, and station-type breakdown.
- A compact evidence package for deciding whether KGE deserves a separate companion-note treatment.

## What It Does Not Duplicate

- It does not regenerate forecasts.
- It does not rerun model training.
- It does not alter the main paper's variance-retention outputs.
- It does not replace the main paper's Skill, Skill_VP, exceedance, or station-level diagnostics.

## Cross-Reference Guidance

The main paper should be cited or mentioned as the source of the benchmark, rolling-origin protocol, variance-retention motivation, and canonical outputs. The companion note should cite the main paper for the evaluation setting, then focus on the additional KGE ranking evidence.

