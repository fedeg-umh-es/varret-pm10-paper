# Paper A — canonical evidence package

**Project:** P4 — Ghost Skill & Dynamic Fidelity · **Paper:** A
**Repository:** `fedeg-umh-es/varret-pm10-paper`
**Source branch:** `claude/auditoria-hstar-censura-cota-4sqaa5` (PR #7; carries the
merged α fix from PR #10).

## Purpose
Freeze a traceable evidence package for Paper A: every canonical claim, number,
table, and figure maps to a **versioned** source file, a regeneration script,
and a commit. No number here originates in the manuscript; no figure asserts a
value absent from its source table.

## Dominant question
Can a daily PM$_{10}$ point forecast achieve positive persistence-relative RMSE
skill while failing to retain the observed day-to-day variance (variance
collapse / dynamic-fidelity loss)?

The α Var/SD correction (PR #10) is a **precision fix to the manuscript text**,
not the scientific contribution, and changed **no reported number**.

## Contents
| file | role |
|---|---|
| `canonical_claims.csv` | claims with type, evidence, status, limitation |
| `canonical_numbers.csv` | every cited number → source file + regeneration command |
| `evidence_map.csv` | claim → source table → prediction artifact → script → commit → manuscript |
| `canonical_tables.md` | main/supporting tables and how to regenerate them |
| `canonical_figures.md` | figures → source tables/scripts |
| `manuscript_change_map.csv` | the two α-definition edits (numbers unchanged) |
| `reproducibility_manifest.json` | commit, dimensions, regeneration commands, invariants |
| `paper_outline.md` | title, thesis, contributions, main table/figure, prohibited claims |

## Key numbers (all traced in `canonical_numbers.csv`)
- 17 stations × 5 models × 7 horizons = 595 cells (119 per model).
- Collapse (α<0.5): HGB 118/119, Ridge 118/119, SARIMA 110/119; seasonal naive 0/119; STL+Ridge 0/119.
- Median skill: HGB 0.205, Ridge 0.219, SARIMA 0.208, seasonal naive −0.026, STL+Ridge −1.107.
- α is a **variance ratio**: 14/14 audited cells match `Var(ŷ)/Var(y)`, 0/14 the SD ratio.

## Invariants
Predictions not regenerated · models not retrained · bootstrap not executed ·
no numbers changed by the α fix. Paths are repo-relative; no absolute local paths.
