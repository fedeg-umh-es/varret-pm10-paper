# Paper A — Audit of the α (variance retention) definition: Var vs SD

**Project:** P4 — Ghost Skill & Dynamic Fidelity · **Paper A**
**Repository:** `fedeg-umh-es/varret-pm10-paper`
**Remote:** `github.com/fedeg-umh-es/varret-pm10-paper` (session proxy)
**Manuscript audited:** `paper_a.tex` on the PR #7 branch (`claude/auditoria-hstar-censura-cota-4sqaa5`), the reframed 17-station version headed to JAMC.
**Audited commit (initial):** `25f124d8c1ebbb06133b39d0ecd369059de15957`
**Working tree at start:** clean.
**Mode:** read-only over data/predictions/outputs; only manuscript text + `audit/` files touched. No predictions, splits, rolling-origin, persistence baseline, H\*, RMSE skill, models, hyperparameters, raw data, or other metric definitions modified. No bootstrap, retraining, or prediction regeneration.

---

## 1. Verdict

**`MANUSCRIPT_ERROR`** (Phase-6 Case D).

The code, all output tables, the metric name ("variance retention"), the thresholds, `Skill_VP`, and the *original* manuscript all use the **variance ratio**. The reframed PR #7 manuscript alone mis-declares α as a **standard-deviation ratio** in two places (its inline definition and its displayed equation), contradicting its own reported numbers. The correction is to the manuscript text only; **no number changes**, no code change, no recomputation.

---

## 2. Formulas

```
Implemented before:     alpha = Var(y_pred) / Var(y_true), ddof=0        (src/diagnostics/variance.py:_compute_alpha)
Declared in manuscript: alpha(h) = SD(y_hat_h) / SD(y_h)                 (paper_a.tex lines 100 and 166, PR#7)  <-- WRONG
Scientifically intended: Var(y_pred) / Var(y_true)                       (name "variance retention"; % variance; original main manuscript defines "a variance ratio"; thresholds; Skill_VP)
Implemented after:      unchanged (Var ratio) — code was already correct
Manuscript after fix:   alpha(h) = sigma^2_yhat / sigma^2_y = Var(y_hat)/Var(y)  (paper_a.tex, this branch)
```

---

## 3. Evidence

- **Exact function:** `src/diagnostics/variance.py:51-55` `_compute_alpha`:
  `np.var(y_pred, ddof=0) / np.var(y_true, ddof=0)` — a variance ratio. Guard for zero observed variance returns `0.0`. Bootstrap CI (`_bootstrap_alpha_ci`, L58-76) uses the same variance ratio.
- **`src/kge_diagnostics.py:52-57`** computes `SD(pred)/SD(true)` and calls it `alpha_h`/`phi_h`, but that is the **KGE** α component (a separate diagnostic, KGE/Paper C), not Paper A's variance-retention α. This is the likely origin of the transcription slip: the KGE SD-ratio convention leaked into the variance-retention definition during reframing.
- **Empirical proof (decisive):** recomputing α from `outputs/metrics/predictions.csv` and matching against `outputs/tables/variance_retention_summary.csv`:
  **14/14 cells `alpha == Var(pred)/Var(true)`; 0/14 `alpha == SD(pred)/SD(true)`.**
- **Reported medians** in `model_family_diagnostic_summary.csv` (HGB 0.1506, Ridge 0.0874, SARIMA 0.0951, seasonal 0.9998, STL+Ridge 1.3987) equal the manuscript's stated medians and are variance ratios. If α were the SD ratio, HGB's median would be √0.1506 ≈ 0.388, not 0.151 — the manuscript's own numbers refute its stated formula.
- **Original manuscript on `main`** explicitly says "a variance ratio" and writes `σ²_ŷ/σ²_y` — the reframed version regressed this.
- **Affected outputs/tables/figures:** `variance_retention_summary.csv` (`alpha`, `skill_vp`, flags, CIs); `model_family_diagnostic_summary.csv` (`median_alpha`); `figure4_alpha_profiles.pdf`, `figure5_scatter_skill_alpha.pdf`, `figure6_threshold_sensitivity.pdf`. Full chain in `impact_map.csv`.
- **Affected paragraphs:** Metrics (L100), Variance-retention definition (L164-169), and every collapse/retention claim that cites α values (abstract L34, results, sensitivity §). All are **numerically correct**; only the L100/L166 definition was wrong.

---

## 4. Numerical impact

- **Rows affected:** 0 (no value recomputed). `before_after.csv` records `alpha_before == alpha_after` for all five model families.
- **Stations / horizons / models:** none change value.
- **Rankings:** unchanged (√ is monotone; median-α order identical before/after).
- **Thresholds:** unchanged — the 0.5/0.8/1.2/1.5 conventions were always applied to the variance ratio.
- **Conclusions / signs:** unchanged. Variance-collapse result (118/119 HGB & Ridge; 110/119 SARIMA) stands verbatim.

---

## 5. Changes made

- **Modified:** `paper_a.tex` — lines 100 and 166 changed from `SD(ŷ)/SD(y)` to `Var(ŷ)/Var(y)` = `σ²_ŷ/σ²_y`, plus one clarifying clause pinning the estimator (population variance, ddof=0, matched finite pairs per station–model–horizon cell) and stating explicitly that this is a variance ratio, not an SD ratio.
- **Created:** `audit/paper_a_alpha_var_sd/` — `report.md`, `inventory.csv`, `manuscript_inventory.csv`, `impact_map.csv`, `before_after.csv`, `manifest.json`, `test_alpha_var_vs_sd.py`.
- **Outputs regenerated:** none (not required; numbers unchanged).
- **Commands:** grep inventory; recompute α (var vs sd) vs `variance_retention_summary.csv`; run `test_alpha_var_vs_sd.py`.

---

## 6. Verification

- New test `test_alpha_var_vs_sd.py`: **10/10 pass** (`PYTHONPATH=. python3 …`), incl. canonical `y_true=[0,1,2,3], y_pred=[0,2,4,6] ⇒ sd_ratio=2, var_ratio=4` and the production `_compute_alpha` returning 4 (variance ratio).
- Existing `tests/test_variance.py` / `tests/test_variance_retention_schema.py` unaffected (no code changed; `pytest` not installed in this environment, so not executed here).
- **Invariants held:** raw data, row-level predictions, splits, baselines, RMSE skill, H\*, other metrics — all unchanged (no code path modified). Bootstrap not executed; models not retrained; predictions not regenerated.
- `git diff --check`: clean. Working tree: only `paper_a.tex` modified + new `audit/` files.
- LaTeX: no `pdflatex` in this environment; the equation edit is a local, self-contained change (amsmath `\operatorname{Var}` / `\sigma^2` — no new packages).

---

## 7. Git

- Base branch: `claude/auditoria-hstar-censura-cota-4sqaa5` (PR #7 — the branch carrying the reframed manuscript).
- Audit branch: `audit/paper-a-alpha-var-sd`.
- Initial commit: `25f124d8c1ebbb06133b39d0ecd369059de15957`.
- Final commit: recorded in `manifest.json` after committing.
- **Push performed: NO** — awaiting explicit authorization (per task instruction). Report + proposed diff delivered first.

---

## 8. Permitted claims (after correction)

- "The variance-retention ratio α(h) = Var(ŷ_h)/Var(y_h) is the fraction of observed variance retained by the forecast at horizon h (population variance, ddof=0, matched pairs per station–model–horizon cell)."
- "HGB direct and Ridge direct exhibit near-universal variance collapse (α < 0.5) in 118/119 cells each; SARIMA in 110/119." (numbers verified as variance ratios)
- "α = 0.5 corresponds to half the observed variance retained." (correct for a variance ratio)
- "Skill_VP(h) = Skill(h)·α(h), with α the variance ratio."

## 9. Prohibited claims

- Do **not** describe α as a standard-deviation / amplitude ratio, or state "α = 0.5 means half the amplitude/SD retained" (that would be √0.5 ≈ 0.71 of the variance).
- Do **not** equate this α with the KGE α component (`src/kge_diagnostics.py`), which is an SD ratio and belongs to the separate KGE diagnostic.
- Do **not** claim any number changed as a result of this correction.

---

## Limitations

- The per-cell 17-station α table was verified against the `e1_rr_daily` predictions available on disk (14/14 cells); the aggregate medians for all 17 stations were taken from the versioned `model_family_diagnostic_summary.csv`. The definitional finding (code = variance ratio; manuscript said SD ratio) is independent of station coverage.
- Fix is textual; if a future edit intends α to be an SD ratio, that is a **scientific** redefinition (Case C) requiring recomputation of all α-derived numbers and is explicitly out of scope here.
