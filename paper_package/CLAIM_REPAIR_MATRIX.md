# Paper A claim repair matrix

Scope: minimal claim repair for the `Dynamic Fidelity and Model Eligibility` framing. No forecasts, metrics, thresholds, tables, figures, model code or evidence artefacts were changed.

| Section | Original claim | Diagnosis | Final claim | Evidence preserved |
|---------|----------------|-----------|-------------|--------------------|
| Abstract | “making that distinction explicit changes forecast eligibility” | OVERCLAIM by ambiguity: could imply model selection | “making that distinction explicit changes the set of station--model--horizon cells passing a post-evaluation screening audit” | Same 595-row aggregate table and Rule A/B audit |
| Abstract | “Rule B added operational thresholds” | OVERCLAIM | “Rule B added study-specific heuristic thresholds” | Same alpha/P75 thresholds and 3 x 3 sensitivity artifact |
| Abstract | “different eligibility decisions” | SUPPORTED_WITH_HEDGE | “different cell-screening outcomes under these stated rules”; eligibility is explicitly not model ranking or operational acceptance | Same 277, 8, 269 and 97.1% values |
| Introduction | “the set of forecasts judged acceptable” | OVERCLAIM | “the set of station--model--horizon cells passing a stated error screen” | No change to Rule A or Rule B |
| Introduction / Contributions | “the quantified decision consequence” | SUPPORTED_WITH_HEDGE | “the quantified change in this cell-level screening set”; explicitly not a model-selection method or universal framework | Same deterministic post-evaluation transformation |
| Framework (auxiliary, not included by current `main.tex`) | “The filters ... audit cell eligibility” | Underspecified | Current manuscript definition in Introduction constrains eligibility to passage of a stated post-evaluation heuristic cell screen; auxiliary file left unchanged | Same Rule A/Rule B formulas |
| Methods | “thresholds are operational criteria” | SUPPORTED_WITH_HEDGE | Wording preserved as an analysis-internal description; manuscript-wide definition excludes operational acceptance, universal standards and deployment rules | Same thresholds; no threshold values changed |
| Results | “Fidelity requirements materially change model eligibility” | SUPPORTED_WITH_HEDGE | Wording preserved; under the Introduction definition this reports a change in cell-screen status, not model selection | Same table and counts |
| Results | “one operational threshold pair” | SUPPORTED_WITH_HEDGE | Wording preserved as the pair used in the audit; surrounding framing excludes operational acceptance or universal use | Same sensitivity grid and percentages |
| Discussion | “different eligibility decisions” | SUPPORTED_WITH_HEDGE | “different cell-level screening outcomes under the stated rules” | Same 101 discordant cells and 277-to-8 transition |
| Discussion | “decision consequence ... across a structured multi-station ... benchmark” | SUPPORTED_WITH_HEDGE | “cell-level screening consequence ... across” the benchmark, with provenance limits retained elsewhere | Same aggregate provenance and limitations |
| Discussion | “The transferable insight ... environmental model-evaluation conclusions may change” | SUPPORTED_WITH_HEDGE | “The bounded methodological implication from this PM10 case is that evaluation conclusions in analogous studies may change” | No cross-pollutant or cross-country evidence added |
| Conclusions | “conventional error-based eligibility and dynamic fidelity identified different sets” | SUPPORTED_WITH_HEDGE | “conventional error-based screening and dynamic fidelity identified different sets”; eligibility is defined as heuristic cell-screen passage, not ranking or acceptance | Same numerical conclusions |
| Conclusions | “bounded but transferable ... should make clear” | SUPPORTED_WITH_HEDGE | “analogous forecast evaluations may need to report whether...” and explicit exclusion of universal rate, acceptance rule and operational benefit | No new recommendation evidence added |

## Repair invariant

The repaired manuscript preserves the empirical contribution as a deterministic comparison of Rule A and Rule B statuses in the existing aggregate benchmark. It does not claim:

* a different top-1 model selected by RMSE versus eligibility;
* operational deployment utility or acceptability;
* a universal threshold or eligibility standard;
* a new metric replacing RMSE, alpha, event metrics or Skill_VP;
* causal explanation of variance collapse.

`SCIENTIFIC_RESULTS_CHANGED = NO`

`NEW_EXPERIMENTS_RUN = NO`

`CLAIM_REPAIR_SCOPE = MINIMAL`
