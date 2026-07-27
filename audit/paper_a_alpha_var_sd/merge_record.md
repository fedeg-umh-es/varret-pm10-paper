# Paper A — PR #10 merge record

- **Repository:** `fedeg-umh-es/varret-pm10-paper`
- **PR:** #10 — *fix: align alpha definition across code and Paper A (variance ratio, not SD)*
- **Base branch:** `claude/auditoria-hstar-censura-cota-4sqaa5` (the Paper A / PR #7 branch — **not** `main`)
- **Head branch:** `audit/paper-a-alpha-var-sd`
- **Source commit:** `62cc0b88a857ce388a6e270468bcdb7a39e958a9`
- **Merge SHA (squash):** `a0894ecdcfd15b0e1609dfcf34da6a9634d4ffb5`
- **Files changed:** `paper_a.tex` (α definition SD→Var on lines 100/166 + estimator clause) and `audit/paper_a_alpha_var_sd/` (report, inventories, impact map, before/after, manifest, guard test).

## Verification at merge
- Diff scope in-bounds: only `paper_a.tex` + `audit/` — no code/output/data/prediction changes.
- Scientific: 14/14 cells `alpha == Var(ŷ)/Var(y)`; 0/14 `== SD ratio`; production `_compute_alpha` returns the variance ratio.
- Tests: `test_alpha_var_vs_sd.py` 10/10 pass. `git diff --check` clean.
- LaTeX: no `pdflatex` in environment; structural check passed (balanced environments/braces, `amsmath` present, no residual SD-ratio definitions).
- **Numerical outputs changed: false.** Predictions regenerated: false. Bootstrap executed: false. Collapse result 118/119: unchanged.

## Merge-order note
PR #10 was merged into the Paper A branch that PR #7 still targets `main` with. The α fix therefore travels with PR #7 when the authors later merge PR #7 → `main`. PR #7 was **not** merged to `main` by this action.
