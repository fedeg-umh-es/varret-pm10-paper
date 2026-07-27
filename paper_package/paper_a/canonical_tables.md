# Paper A — canonical tables

Every table maps to a versioned source file and a regeneration script. No number
originates in the manuscript alone.

## Main table — five-model diagnostic summary
- **Source:** `outputs/tables/model_family_diagnostic_summary.csv`
- **Regenerate:** `python3 scripts/13_build_five_model_diagnostic_summary.py`
- **Rendered in manuscript:** `model_family_diagnostic_summary.tex` (\input into `paper_a.tex`)
- **Content:** per model (5 rows): `n_cells`, `median_skill`, `median_alpha`,
  `collapsed_cells`, `collapse_rate_pct`, `near_ideal_cells`,
  `positive_skill_cells`, `median_skill_vp`, DM significance counts, exceedance
  (recall/CSI/FAR), Murphy components.
- **Key rows:** HGB (skill 0.205, α 0.151, collapse 118/119), Ridge (0.219,
  0.087, 118/119), SARIMA (0.208, 0.095, 110/119), seasonal naive (−0.026,
  1.000, 0/119), STL+Ridge (−1.107, 1.399, 0/119).

## Supporting table — per-cell variance retention
- **Source:** `outputs/tables/variance_retention_summary.csv`
- **Columns:** `dataset, model, horizon, n, skill, mae_skill, alpha,
  alpha_ci_low, alpha_ci_high, skill_vp, collapse_flag, inflation_flag,
  near_ideal_flag, low_sample_flag`
- **α definition:** `alpha = Var(y_pred)/Var(y_true)`, ddof=0 (see
  `src/diagnostics/variance.py::_compute_alpha`).

## Supplementary — threshold sensitivity
- **Regenerate:** `python3 scripts/06_threshold_sensitivity.py`
- Collapse rates at α<0.4/0.5/0.6 (robustness of the collapse convention).

## Audit table — α Var/SD (precision correction, not a result)
- **Source:** `audit/paper_a_alpha_var_sd/before_after.csv` (α unchanged; the
  SD-ratio the erroneous label implied is recorded for context only).
