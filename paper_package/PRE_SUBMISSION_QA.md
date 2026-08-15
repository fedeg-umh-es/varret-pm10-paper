# P4 Paper A Pre-Submission QA

## 1. Repository state

- HEAD: `a8604fd4a1fca681e3e526b8e1827c2d3ce15281`
- Branch: `codex/p4-lightgbm-ems-gap-audit`
- Worktree: DIRTY; `git diff --check` PASS. The tracked diff is four repaired section files plus the pre-existing modified ZIP. The claim-repair source diff is 60 insertions / 50 deletions; the ZIP is a binary pre-existing change.
- Pre-existing modifications: `paper_package_overleaf.zip`; `P4_DYNAMIC_FIDELITY_MODEL_ELIGIBILITY_AUDIT.md`; `docs/TRACEABILITY.md`; `docs/p4_legacy_closeout/`; `manuscript.pdf`; `manuscript.tex`; `multirow.sty`; `physics.sty`; `scripts/45_run_rq2_diagnostic_pipeline.py`; `scripts/audit_jem_selection_consequences.py`; `scripts/reproduce.py`; `siunitx.sty`; `tests/test_rq2_post_eval_diagnostics.py`; `uv.lock`. These were preserved and not treated as QA repairs.
- Claim-repair modifications: `paper_package/sections/abstract.tex`, `paper_package/sections/introduction.tex`, `paper_package/sections/discussion.tex`, `paper_package/sections/conclusions.tex`, and `paper_package/CLAIM_REPAIR_MATRIX.md`.
- Unexpected modifications: NONE. `paper_package/PRE_SUBMISSION_QA.md` is the sole artifact created by this QA run.

## 2. Framing integrity

MODEL_ELIGIBILITY_DEFINITION_PRESERVED = YES

DYNAMIC_FIDELITY_SCOPE_PRESERVED = YES

The controlling audit and claim-repair matrix were read in full. The active `paper_package/main.tex` includes the repaired Abstract, Introduction, Methods, Results, Discussion, Conclusions, and Supplementary Material. The manuscript-level definition limits eligibility to heuristic post-evaluation screening of station--model--horizon cells; no active passage converts it into ranking, top-1 selection, acceptance, certification, or deployment.

## 3. Section gates

ABSTRACT_GATE = PASS

INTRODUCTION_GATE = PASS

METHODS_RESULTS_COMPATIBILITY = WARNING

DISCUSSION_GATE = PASS

CONCLUSION_GATE = PASS

The Methods and Results remain compatible because they define Rule A/Rule B as post-evaluation filters, explicitly state that they do not select a final winner, and the Results explicitly exclude population rankings and final-winner claims. Two residual uses of “operational” are lexically ambiguous but are framed as analysis-internal criteria and are surrounded by non-universal caveats; they should receive author confirmation before submission.

## 4. Claim consistency

| File | Claim/term | Safe interpretation | Status | Action |
|---|---|---|---|---|
| `paper_package/sections/methods.tex:99-103` | “The alpha and P75-recall thresholds are operational criteria for this analysis” | Internal criteria used to implement the stated post-evaluation cell screen; not acceptance or deployment criteria, as constrained by the Introduction and surrounding caveat | WARNING | Author review only; confirm that “operational” is intended in the internal analytical sense. No source edit made in this QA. |
| `paper_package/sections/results.tex:63-78` | “The joint evidence changes the decision set” and “one operational threshold pair” | Rule A versus Rule B changes pass/fail status of benchmark cells under the displayed heuristic pair; not a model-selection or deployment decision | WARNING | Author review only; preserve the existing Results and do not interpret “decision set” or “operational” as top-1 selection or acceptance. |
| `paper_package/sections/framework.tex:33` (auxiliary file not included by current `main.tex`) | “The alpha and P75-recall thresholds are operational criteria for this audit” | Same internal-audit meaning, but the file remains a reusable auxiliary source with ambiguous wording | WARNING | If this auxiliary section is ever reactivated, apply the manuscript-level boundary before use. It has no effect on the current build. |
| `paper_package/tables/table3_decision_rule.tex:3-8` and `paper_package/tables/rule_threshold_sensitivity.tex:3` | “eligible cells” / “retained Rule-B eligible cells” | Cell-level passage of the explicitly defined Rule A/Rule B screens | PASS in context | No action. Captions also state the benchmark scope and non-independence; no ranking or deployment interpretation is asserted. |

No unsafe use of `admissible`, `inadmissible`, `best model`, certification, or universal selection language was found in the active manuscript. `dynamic fidelity` remains an umbrella for existing variance-retention and event-behaviour diagnostics. `Skill_VP` is absent from the active new narrative and is not promoted to a replacement metric.

## 5. Numerical and evidence integrity

NUMERICAL_EVIDENCE_CHANGED = NO

SCIENTIFIC_EVIDENCE_MODIFIED = NO

NEW_EXPERIMENTS_DETECTED = NO

SCIENTIFIC_CODE_MODIFIED = NO

The repaired source preserves the empirical counts, percentages, thresholds, horizons, model names, station count, statistical-test claims, table content, figure content, and evidence references. The only newly introduced numeric token is the semantic qualifier “top-1”; it is not an empirical result. The tracked diff contains no tables, figures, scripts, source code, output tables, manifests, or evidence artifacts. The pre-existing modified ZIP was not treated as the canonical target and was not changed during this QA.

## 6. LaTeX build

BUILD_STATUS = PASS

UNDEFINED_REFERENCES = NO

MISSING_CITATIONS = NO

MISSING_ASSETS = NO

MATERIAL_LAYOUT_PROBLEMS = WARNING

The documented `pdflatex → bibtex → pdflatex → pdflatex` sequence passed in an isolated temporary copy of `paper_package` and produced an 11-page PDF. A direct build using the user TeX tree first failed because the local `expl3.sty` was newer than the installed LaTeX format; isolating `TEXMFHOME`, `TEXMFVAR`, and `TEXMFCONFIG` resolved the environment mismatch without changing repository files. The successful build had no undefined references, missing citations, missing figures/tables, fatal errors, or overfull boxes. Benign hyperref PDF-string warnings and underfull boxes remain.

Complete visual inspection of all 11 rendered pages found correct title/author block, abstract, equations, tables, figures, captions, references, URLs, Data and Software Availability, AI disclosure, and supplementary content. The only layout warning is that S2 and S3 supplementary headings appear at the bottom of page 10 while their floating tables begin on page 11; content is present and readable, but pagination should be checked during author review.

## 7. Data/software availability

DATA_SOFTWARE_AVAILABILITY_GATE = WARNING

Blocking issue, if any:

The statement is accurate for the current state: it identifies the repository, the canonical aggregate table and provenance records, states that no external DOI is asserted, and explicitly retains `FINAL_REPOSITORY_SYNC_REQUIRED = YES`. The current dirty worktree and uncommitted claim-repair source mean repository synchronisation remains required before an actual EMS submission or final packaging. This does not block author review.

## 8. AI disclosure

AI_DISCLOSURE_GATE = PASS

The disclosure accurately limits use to language editing, code/document assistance, and LaTeX formatting, and states that no data, forecasts, empirical results, scientific conclusions, or submitted artwork were generated or altered. This QA introduced no scientific computation or evidence changes.

## 9. Submission blockers

- Repository synchronisation and final submission-commit packaging remain required before EMS submission; the package itself explicitly declares this requirement.

## 10. Non-blocking warnings

- Residual “operational criteria” / “operational threshold pair” wording in unchanged Methods/Results is compatible with the cell-screen definition but should be confirmed by the authors to avoid any acceptance/deployment reading.
- Supplementary S2/S3 headings are separated from their floating tables by a page break in the generated PDF.
- The first build attempt exposed a local TeX support-file mismatch; the isolated documented build passed.

## 11. Final verdict

READY_FOR_AUTHOR_REVIEW

## 12. Next permitted action

Author review of this QA report and its non-blocking warnings, followed by repository synchronisation only after author approval.
