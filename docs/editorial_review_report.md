# Editorial review report: Paper A

Date: 2026-07-22  
Reviewed source: `origin/main` at `f57f076`  
Review branch: `agent/editorial-review-paper-a`

## Outcome

The generic manuscript is internally coherent with the committed
rolling-origin reproduction.  Its central claim is now bounded to one hourly
PM10 station-year: modest positive SARIMA skill at 24-48 h coexists with severe
variance attenuation and loss of P75-event sensitivity.  LightGBM is not
presented as consistently skilled, and Skill_VP remains an auxiliary
diagnostic.

The repository names *Atmospheric Research* (Elsevier) as the target journal.
Its official Guide for Authors was checked on 2026-07-22. The journal's scope
includes air pollution and atmospheric aerosols, so the PM10 application is in
scope; the remaining fit risk is the contribution threshold for a single
station-year diagnostic study. Template-specific submission items remain
pending ORCID and funding information, applicable declarations, and a final
article-type decision.

## Problems found and corrections made

| Area | Problem found | Correction |
|---|---|---|
| Claims | The abstract treated RMSE-relative skill as a standard yardstick and moved too quickly from diagnostic results to operational value. | Recast RMSE skill as commonly used; replaced operational claims with bounded statements about practical interpretation and indirect diagnostics. |
| Protocol | Fold sizes, same-row baseline comparison, scaling policy, P75 calibration, and undefined precision were not fully specified. | Added exact grid-based fold sizes, train-only P75 definition, no-scaling statement, causal filling rule, matched verification rows, and missing-precision convention. |
| Models | LightGBM and SARIMA configuration was incomplete and LightGBM was called a strong predictor. | Added fixed lag/rolling/calendar features and hyperparameters, clarified inactive row subsampling, stated SARIMA order/constant/state update, and removed performance-promotional language. |
| Results | The blocked holdout was labelled protocol robustness. | Renamed it protocol sensitivity and made rolling-origin the canonical analysis. |
| Events | P75 events could be mistaken for regulatory exceedances; SARIMA precision at 48 h could be read as zero. | Defined P75 as a training-calibrated diagnostic threshold, explicitly not regulatory; stated that precision is undefined because no flags are issued. |
| Discussion | Skill_VP and event results were linked too directly to operational consequences. | Limited Skill_VP to compact description and treated event metrics as convergent diagnostics, not direct utility evidence. |
| Limitations | Statistical uncertainty and the non-regulatory event threshold were missing. | Added absence of confidence intervals/formal loss tests, decision-context limits, single-site/year scope, and untested regimes. |
| Reproducibility | The generated LaTeX rows ended with one backslash and failed a clean compilation. | Fixed the generator to emit `\\`, kept `\bottomrule` in the generated fragment, and added a divergence test. |
| Repository orientation | The README foregrounded an older daily P33 module and contained obsolete artifact paths. | Reoriented the README to the hourly Paper A source of truth and isolated retained legacy material. |
| Bibliography | Three uncited entries remained; the audit covered only eight repaired entries. | Removed the three orphans and verified all nine cited works. |

## Claims softened or removed

- Removed the implication that positive RMSE skill establishes operational
  value or usefulness.
- Replaced a strong operational-consequence claim with evidence of convergent
  variance and event diagnostics.
- Removed the characterization of LightGBM as a strong predictor in this run.
- Replaced protocol robustness with protocol sensitivity.
- Avoided treating P75 events as regulatory exceedances.
- Preserved the statement that LightGBM does not consistently beat persistence.
- Preserved SARIMA's positive but modest rolling-origin skill and its 24-48 h
  variance collapse as the bounded central result.

## Empirical consistency checks

- Abstract skill ranges and reported SARIMA variance-retention values match
  `metrics_rolling_origin.csv`.
- Every E1 table cell is generated from `metrics_rolling_origin.csv`.
- E2 values match `metrics_holdout.csv` and are labelled secondary sensitivity.
- E3 recall, precision, and flag-rate statements match
  `events_p75_rolling_origin.csv`.
- SARIMA 48 h precision remains missing while flag rate is zero.
- Row-level comparison confirms LightGBM and SARIMA use identical fold, origin,
  target, horizon, observation, persistence, and P75-threshold rows.
- No EEA attribution, old repository URL, or legacy 0.92-0.96 LightGBM claim
  remains in the manuscript.
- Both figure builders read the canonical rolling-origin CSV files directly.

## Reference audit

- Cited keys: 9; bibliography entries: 9.
- Undefined citations: 0; uncited entries: 0.
- Authors, titles, years, venues, volume/issue/pages, and DOI or ISBN were
  checked against DOI, publisher, official proceedings, or authoritative
  metadata records.
- Detailed evidence is recorded in `docs/reference_audit.md`.

## Structural diagnosis

### Part 0: Desk Reject Surface

#### A) Affirmative contribution check

**Verdict: Pass.** The title and abstract state the positive contribution:
horizon-wise joint interpretation of persistence-relative skill, variance
retention, and P75-event diagnostics in a leakage-free PM10 case study.

#### B) Familiar framing check

**Verdict: Familiar for environmental/time-series forecasting.** Internal
labels are not required to understand the title or abstract, and Skill_VP is
introduced only after established quantities.

#### C) Two-minute editor test

- Contribution: empirical evidence that modest positive error skill can coexist
  with severe variance attenuation at longer horizons.
- Relevance: it qualifies horizon-wise point-forecast evaluation in PM10.
- Article type: bounded empirical-diagnostic methods application.
- Understandability: clear without reading the full manuscript.

**Verdict: Clear.**

**Desk reject surface verdict: Medium surface risk.** The main residual risk is
not presentation but contribution threshold: a single station-year and an
unvalidated auxiliary diagnostic may be insufficient for some journals.

Top surface-level fixes completed: affirmative abstract contribution,
non-promotional Skill_VP framing, and explicit empirical scope.

The abstract is framed primarily as a positive reproducible forecasting-
evaluation contribution.  It is not framed as a general theory, H* result, or
critique of prior literature.

### Part 1: Journal Fit

The repository's `codex_experiment_plan_atm_research.txt` names *Atmospheric
Research* (Elsevier) as the target. Its official Guide for Authors, checked on
2026-07-22, identifies air pollution and atmospheric aerosols among the
journal's special emphases. The paper is therefore topically plausible, but its
single-station, evaluation-diagnostic contribution has a material desk-reject
risk relative to papers centred on atmospheric processes. The guide requires
editable source files; it limits abstracts to 250 words, requests 1--7 English
keywords, and requires title-page, competing-interests, funding, and CRediT
information. It also lists highlights and a graphical abstract for submission
preparation. These items are recorded in the checklist.

### Part 2: Structural diagnosis (C-C-C and PIER)

#### A) Skip Test

Introduction topic sentences:

1. **Intro P1:** Short-term air-quality forecasting is used to anticipate
   changes at successive lead times.
2. **Intro P2:** Squared-error criteria create a practical complication.
3. **Intro P3:** The paper addresses an empirical, protocol-specific gap.
4. **Intro P4:** The study provides a bounded empirical characterization.

Discussion topic sentences:

1. **Discussion P1:** The evidence supports a horizon-wise diagnostic reading.
2. **Discussion P2:** Skill_VP compactly describes, but does not establish the
   utility of, the divergence.
3. **Discussion P3:** Event-oriented evaluation requires caution.

**Result:** Pass.  Topic sentences reconstruct context, problem, contribution,
interpretation, and limits without a major logical jump.

#### B) PIER anatomy

- **Methods/Rolling-origin P2 — formerly overloaded:** protocol sequence,
  missingness, scaling, and matched-row evaluation were implicit or scattered.
  Mechanical fix completed: ordered these elements from fold construction to
  sequential updating and verification.
- **Results/E2 P1 — role leakage:** the term robustness implied a conclusion
  stronger than a single holdout permits.  Mechanical fix completed: relabelled
  as sensitivity and subordinated it to rolling-origin.
- **Discussion P2 — unsupported link:** event diagnostics were treated as an
  operational consequence.  Mechanical fix completed: separated empirical
  co-occurrence from decision-specific usefulness.

#### C) C-C-C alignment

- Last Introduction block: states bounded objective and contribution.
- Discussion: interprets and limits results rather than merely repeating them.
- Abstract: follows context, method, evidence, and bounded conclusion.

#### D) Section-purpose audit

| Section | Expected question | Current function | Verdict |
|---|---|---|---|
| Introduction | Why does this matter? | Establishes the error/fidelity tension and bounded PM10 gap. | Aligned |
| Related work | What is known and what remains unresolved? | Positions baseline-aware evaluation and variance diagnostics without claiming universal novelty. | Aligned |
| Methodology | What was done? | Gives auditable data, folds, models, metrics, and threshold policy. | Aligned |
| Results | What evidence was produced? | Separates canonical E1, sensitivity E2, and event E3. | Aligned |
| Discussion | What does it mean? | Interprets model-dependent divergence and limits utility claims. | Aligned |
| Conclusion | Why does it matter now? | States the bounded diagnostic implication without adding evidence. | Aligned |

### Part 3: Action plan

- **ACTION 1:** Keep rolling-origin as the canonical protocol — **Results E1/E2** — to prevent holdout sensitivity from being read as equivalent evidence. Completed.
- **ACTION 2:** Define event calibration before reporting event results — **Methods, P75 diagnostics** — to make leakage control and undefined precision auditable. Completed.
- **ACTION 3:** Separate diagnostic co-occurrence from operational utility — **Abstract, Discussion P2-P3, Conclusion** — because no decision loss was evaluated. Completed.
- **ACTION 4:** Preserve single-case boundaries — **Related Work, Limitations** — because external validity and Skill_VP validation remain open. Completed.

## AUDITORÍA — Manuscrito final

**Veredicto global:** OK

El texto es consistente con el ancla editorial del proyecto.

## Academic-language naturalness audit

| Fragment changed | Problem detected | Stylistic risk | Applied change |
|---|---|---|---|
| “standard yardstick” | Promotional generalization | Formulaic opening | Replaced with “commonly used”. |
| “strong, widely used nonlinear predictor” | Unnecessary evaluation adjective | Inflated model status | Retained only “widely used” and stated the non-benchmark role. |
| “confirms the operational consequence” | Over-neat evidence-to-impact transition | Causal/operational overclaim | Replaced with “supplies a convergent diagnostic”. |
| Repeated “operational usefulness” | Same register and certainty across sections | Homogeneous, overstated voice | Varied language by section and restricted usefulness claims to decision-specific evidence. |

Three priority changes completed: reduced promotional adjectives, separated
results from interpretation, and varied the degree of caution across abstract,
methods, discussion, and conclusion.

Global rewriting rule applied: state the observed quantity first, then the
bounded interpretation, then the untested implication.

Methodological warning: no stylistic edit changed rolling-origin evaluation,
train-only processing, persistence comparisons, horizon-wise metrics, or the
canonical numerical outputs.

## Visual and PDF checks

- Clean `latexmk` build completed with BibTeX.
- Final PDF: 9 letter-size pages after restoring confirmed authorship and
  declarations.
- Final log: no undefined citations, undefined references, overfull boxes, or
  underfull boxes.
- All nine pages rendered to PNG and inspected individually.
- Both manuscript figures opened separately at original resolution.
- Table rules, negative signs, equations, captions, cross-references, line work,
  markers, legends, and page transitions are legible.
- SARIMA precision at 48 h is absent from the plotted precision curve and its
  undefined status is explicit in the caption and manuscript.
- No clipping, overlap, broken glyph, or materially misplaced float was found.

## Skills and tooling audit

Skills used because they materially affected the review:

- `github:github`: repository and branch orientation against `origin/main`.
- `github:yeet`: scoped commit, push, and draft-PR workflow; `gh` was
  authenticated and used where connector coverage was not required.
- `pdf`: clean compilation, page rendering, PDF metadata checks, and visual
  inspection of every page and both source figures.

Applicable capabilities used alongside those skills were the real LaTeX
toolchain, repository tests, automated manuscript/output consistency checks,
and primary-source web verification for cited references.

Skills considered but not used because they were not applicable:

- `documents`: its artifact workflow targets Word/Google Docs rather than the
  repository's LaTeX source.
- `control-browser`: public search and direct official-source retrieval were
  sufficient; no interactive browser task was needed.
- `github:gh-address-comments` and `github:gh-fix-ci`: there were no existing PR
  review threads or failing GitHub Actions checks in scope.
- Documents, presentations, spreadsheets, image generation, visualization,
  email, calendar, messaging, website, Box, Drive, Notion, Wix, plugin, and
  skill-creation skills: none contributes to this local LaTeX manuscript audit,
  reproducibility validation, or GitHub delivery.

## Risks requiring human decision

1. Confirm *Atmospheric Research* as the target and decide whether its
   contribution threshold is compatible with a single-station, single-year
   diagnostic study.
2. Supply and verify both authors' ORCID identifiers and the funding statement;
   author names, affiliations, emails, corresponding author, CRediT roles, and
   competing-interest declaration have been confirmed from the earlier
   manuscript.
3. Decide whether the blocked-holdout sensitivity belongs in the main text or
   supplementary material for the selected journal.
4. Accept or revise the lack of interval estimates/formal loss tests; adding
   experiments was outside this review's authorized scope.
5. Approve final use of the informal “ghost skill” term, which currently appears
   only as explicitly non-formal shorthand.
