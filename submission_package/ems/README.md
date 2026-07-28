# Submission Package for Environmental Modelling & Software

This folder contains the complete, ready-to-upload submission package for **Paper A** adapted for *Environmental Modelling & Software* (Elsevier).

## Package Contents

1. **Manuscript Source & PDF:**
   - `paper_a_ems.tex`: Canonical LaTeX manuscript source using `elsarticle.cls`.
   - `paper_a_ems.pdf`: Cleanly compiled 17-page manuscript PDF.
   - `references.bib`: BibTeX bibliography database (18 entries).

2. **Mandatory Elsevier Submission Items:**
   - `highlights.txt`: 5 bullet points (68–82 characters each).
   - `graphical_abstract.pdf` / `graphical_abstract.png`: Vector and raster graphical abstract.
   - `cover_letter_draft.md`: Customized cover letter for *Environmental Modelling & Software*.

3. **Mandatory Declarations & Statements:**
   - `credit_author_statement.txt`: CRediT roles for all co-authors.
   - `declaration_of_competing_interest.txt`: Conflict of interest declaration.
   - `data_availability_statement.txt`: Public data repositories (MITECO) statement.
   - `code_availability_statement.txt`: GitHub code repository statement.
   - `funding_statement.txt`: Grant placeholder text.

4. **Supplementary Materials:**
   - `supplementary_material_ems.tex`: Supplementary material LaTeX source.
   - `supplementary_material_ems.pdf`: Compiled 2-page supplementary material PDF.

5. **Figure Files:**
   - All 11 PDF figure files (`figure1_*.pdf` through `station_map_*.pdf`).

## P3-12R consolidation note (2026-07-28)

- `figure5_scatter_skill_alpha.pdf` and `figure_skill_alpha_five_models.pdf`
  were regenerated from the validated, recovered 17-station evidence
  (`evidence/paper_a/aggregates/`); the only visual change is the x-axis
  label, corrected from an SD/SD notation to the canonical
  `alpha = Var(y_hat)/Var(y)`. See `outputs/p3_12r_artifact_consolidation/figure5_regeneration_audit.csv`.
- `model_family_diagnostic_summary.tex` was added to this folder (it is
  `\input` by `paper_a_ems.tex` but was missing from the package).
- The `\input{supplementary_material.tex}` reference in `paper_a_ems.tex` was
  corrected to `\input{supplementary_material_ems.tex}` to match the actual
  filename present in this folder (mechanical fix, no content change).
- `data_availability_statement.txt` and `code_availability_statement.txt`
  (and the matching section in `paper_a_ems.tex`) were updated to precisely
  distinguish raw data, processed data, aggregate diagnostics, row-level
  predictions, and code, per `outputs/p3_12r_artifact_consolidation/data_availability_audit.csv`.
- `file_manifest.csv` was regenerated to cover all 34 files actually present
  in this folder (the previous manifest only listed 15).
- Row-level predictions (`predictions_all_stations.csv`, 895,737 rows) are
  **not** included in this editorial package; see the Data Availability
  Statement for their location.
