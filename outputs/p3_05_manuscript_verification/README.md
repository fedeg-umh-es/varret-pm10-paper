# Outputs — Verification Audit of Canonical Paper A (P3-05)

This directory contains the audit artifacts for task **P3-05: compilar y verificar el manuscrito canónico de Paper A**.

## Inventory of Audit Artifacts

- `verification_summary.json` — Structured JSON summary of the complete audit.
- `manuscript_inventory.csv` — File manifest of `paper_a.tex`, inputs, bib, and figures.
- `latex_dependency_audit.csv` — Audit of `\input`, `\includegraphics`, labels, refs, and citations.
- `manuscript_claim_audit.csv` — Verification of manuscript claims against `canonical_claims.csv`.
- `manuscript_number_audit.csv` — Verification of reported numbers against `canonical_numbers.csv`.
- `alpha_definition_audit.csv` — Specific audit of $\alpha(h) = \operatorname{Var}(\hat{y})/\operatorname{Var}(y)$ definition.
- `figure_audit.csv` — Verification of all 8 PDF figure assets.
- `table_audit.csv` — Verification of included LaTeX tables.
- `cross_paper_contamination_audit.csv` — Audit for Paper B terms ($H^*$, oracle envelope, etc.).
- `latex_log_audit.csv` — Compilation toolchain log audit.
- `pdf_visual_audit.md` — Detailed analysis of PDF asset status.
- `desk_reject_surface.md` — Desk-reject risk evaluation (Title, Abstract, Significance, Contributions).
- `repair_plan.md` — Itemized repair plan for build assets.
- `final_report.md` — Full executive audit report and global verdict.
