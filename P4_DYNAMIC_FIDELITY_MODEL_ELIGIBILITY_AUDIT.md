# P4 Dynamic Fidelity / Model Eligibility Audit

## 1. Repository state

* Repo: `varret-pm10-paper` at `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`
* Branch: `codex/p4-lightgbm-ems-gap-audit`
* HEAD: `a8604fd4a1fca681e3e526b8e1827c2d3ce15281` (`docs: update SERRA cover letter and highlights`)
* Working tree: `DIRTY`. The tracked manuscript source `paper_package/` is clean at HEAD; `paper_package_overleaf.zip` is modified and several audit/build artefacts are untracked. The modified ZIP contains a different title (`Dynamic Fidelity and Model Selection...`) and is not the exact target version audited here.
* Canonical manuscript: `paper_package/main.tex` at `3d248a2c4578bbf8812ff0e50a8dbdf6d4e416d5` (parent of the decision-audit rewrite), titled `Variance Retention: A Forecast-Verification Diagnostic for Skilful-but-Smoothed Point Forecasts of Daily PM$_{10}$`; corroborating canonical files are `paper_a_ems.tex` and `submission_package/ems/paper_a_ems.tex`.
* New-framing manuscript: `paper_package/main.tex` and included `paper_package/sections/*.tex` at HEAD; exact title introduced by committed commit `96572640a4043de2859058db9d56390a7e76c214`, not an uncommitted-only text.
* Relevant commits/tags: `87f3390` (EMS package), `c1ceda8` (first decision-audit rewrite, then titled Model Selection), `9657264` (exact Model Eligibility title), `3d1b435` (current package closeout), `a8604fd` (current HEAD), tag `paper-a-row-level-evidence-v1` at `4e3a97a`, and tag `v1.0.0`.

## 2. Identity verdict

`SAME_PAPER_WITH_SCIENTIFIC_DRIFT`

La nueva versión conserva el mismo benchmark, estaciones, modelos, horizontes, validación rolling-origin, baseline y diagnósticos base. Sin embargo, cambia la pregunta dominante: de diagnosticar el desacoplamiento skill--fidelity a cuantificar el efecto de dos filtros de elegibilidad construidos post-evaluación. Rule A/Rule B, los umbrales conjuntos y el porcentaje 277-to-8 son una nueva capa analítica explícita, aunque se derivan determinísticamente de una tabla agregada existente y no de nuevas predicciones. La deriva es real pero acotable: no llega a una versión científicamente distinta si `eligibility` se limita a screening descriptivo de celdas y no a selección normativa de modelos.

## 3. Scientific question comparison

| Dimension | Canonical | New version | Verdict |
| --------- | --------- | ----------- | ------- |
| Object of study | Desacoplamiento entre persistence-relative RMSE skill, variance retention, dinámica y episodios en PM10 diario | Consecuencia de añadir requisitos de fidelidad a un filtro de elegibilidad basado en skill/DM | `SCIENTIFICALLY_CHANGED` |
| Dominant question | Puede coexistir skill positivo con pérdida de variabilidad/fidelidad dinámica | ¿Cambia el conjunto de celdas que pasa cuando se añade Rule B a Rule A? | `SCIENTIFICALLY_CHANGED` |
| Empirical hypothesis | Error relativo y fidelidad dinámica pueden divergir | La divergencia produce cambios cuantificables en la elegibilidad de celdas | `SCIENTIFICALLY_CHANGED` |
| Dataset and scope | PM10 diario, 17 estaciones MITECO/EMEP, cinco familias, siete horizontes | El mismo benchmark agregado de 595 celdas; se retira el periodo común no defendible | `REFRAMED_ONLY` |
| Interpretation | Diagnóstico post-evaluación, no teoría universal ni leaderboard | Auditoría de screening post-evaluación; el texto declara que no elige un ganador final | `REFRAMED_ONLY` with a normative-risk warning |

## 4. Methods diff

| Difference | Canonical version | New version | Scientific assessment / provenance |
| ---------- | ----------------- | ----------- | ---------------------------------- |
| Main analytical layer | Skill, alpha, Skill_VP auxiliary, exceedance behaviour, Murphy decomposition and threshold diagnostics | Rule A: `skill > 0 AND DM_BH significant`; Rule B: Rule A plus `alpha >= 0.50 AND recall_P75 >= 0.20` | Genuine post-evaluation analytical expansion in the manuscript; no refit or forecast generation. Defined in `paper_package/sections/framework.tex` and `scripts/15_decision_change_analysis.py`. |
| Eligibility algorithm | Canonical narrative used a joint skill--fidelity diagnostic template; Rule A/B was not the dominant scientific question | Explicit deterministic formulas and cell-status transition | Supported as a cell-level transformation of the existing aggregate table, not as a validated universal selection algorithm. |
| Threshold sensitivity | Canonical alpha-collapse sensitivity around 0.4/0.5/0.6 | Joint 3 x 3 grid: alpha `{0.40, 0.50, 0.60}` x P75 recall `{0.10, 0.20, 0.30}` | Existing deterministic audit artifact; no forecasts recomputed. The paper reports 91.3--98.2% changes in this displayed grid. |
| Variance retention definition | `alpha = Var(y_pred)/Var(y_true)`, population variance (`ddof=0`) | Same definition | `UNCHANGED`; confirmed in `paper_package/sections/metrics.tex` and canonical traceability. |
| Skill_VP | Present only as auxiliary diagnostic, explicitly not a standard | Omitted from the new main text | No misuse detected; omission changes emphasis, not the underlying empirical results. It must not be replaced implicitly by Rule B as a universal score. |
| Murphy decomposition | Reported as supporting interpretation in the canonical manuscript | Removed from the new main narrative | Existing diagnostic is not contradicted, but the new story no longer uses it as support. No new decomposition result is introduced. |
| Event diagnostics | P75/P90 recall, precision/FAR and event behaviour were central supporting diagnostics | P75 recall becomes a Rule-B gate; P90/FAR/precision are not central in the new text | Existing event table supports P75 values; using P75 as an acceptance gate is a new interpretive layer and must remain study-specific. |
| Statistical testing | BH-adjusted two-sided HLN/DM comparison against persistence | Same test is embedded in Rule A | `UNCHANGED` in implementation; the new version changes its role from evidence of skill to a gate component. |
| Models | HGB direct, Ridge direct, SARIMA, seasonal naive, STL+Ridge | Same five families | `UNCHANGED`; no new family and no LightGBM row in the new package. |
| Horizons and station--model--horizon grid | Seven horizons and 595 structured cells | Same | `UNCHANGED`; no station--model--horizon rows were added by the framing. |
| Rolling-origin and preprocessing | Rolling-origin, train-only safeguards and persistence baseline | Five expanding folds and train-only controls are stated more explicitly; exact current fold-level chain remains incompletely reproducible remotely | No new forecast design detected, but the five-fold claim carries a provenance warning because the available run manifest has a conflicting older protocol. |
| Dataset period and exact model settings | Canonical text stated 2017--2024 and more detailed settings | New text removes the common period and exact settings where provenance is conflicting or unrecoverable | Corrective narrowing of claims, not a new scientific method. |

## 5. Results and numeric provenance

| Claim/number | Source artifact | Traceable | Changed? | Risk |
| ------------ | --------------- | --------: | -------: | ---- |
| 595 station--model--horizon cells; 17 stations; 5 models; 7 horizons | `outputs/tables/master_diagnostic_table.csv`, SHA-256 `6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6`; `evidence/paper_a/manifests/provenance_manifest.csv` | YES locally | NO | Aggregate source is traceable, but `audit/final_submission/GRADE_A_REMOTE_AUDIT.md` classifies the full 17-station end-to-end chain as incomplete for Grade A. |
| Rule A = 277 | `audit/decision_change/recomputed_summary.json`; `audit/decision_change/recomputed_cell_table.csv`; `scripts/15_decision_change_analysis.py` and independent `audit/decision_change/verify_decision_change.py` | YES locally | NEW in central narrative | Deterministic aggregate transformation; not independently regenerated from a complete all-station row-level chain. |
| Rule B = 8 | Same as above; formula also in `paper_package/sections/framework.tex` and `paper_package/tables/table3_decision_rule.tex` | YES locally | NEW in central narrative | Threshold-dependent and heuristic; not a universal acceptance count. |
| 269 changes and 97.1% of Rule-A cells | `audit/decision_change/recomputed_summary.json` (`checks_pass: 18`, input hash above); `paper_package/TRACEABILITY.md` | YES locally | NEW in central narrative | Correct for Rule A versus Rule B; must not be described as a top-1 model-selection reversal or population rate. |
| Model breakdown 111->1 HGB, 117->1 Ridge, 44->1 SARIMA, 5->5 seasonal naive, 0->0 STL+Ridge | `audit/decision_change/recomputed_cell_table.csv`; `paper_package/tables/table3_decision_rule.tex`; `paper_package/scripts/generate_tables.py` | YES locally | NEW in central narrative | Cell counts, not family superiority or final model winners. |
| 101 formal discordant cells: Rule A, P75 recall >= 0.20, alpha < 0.50; 54 HGB, 43 Ridge, 4 SARIMA | `audit/decision_change/discordant_cases.csv`, `recomputed_summary.json`, `paper_package/TRACEABILITY.md` | YES locally | NEW in central narrative | Supported as a constructed discordance subset; upstream SARIMA/all-station provenance remains partial remotely. |
| Alpha collapse 118/119 HGB, 118/119 Ridge, 110/119 SARIMA | `outputs/tables/master_diagnostic_table.csv`; `paper_package/tables/table2_model_summary.tex`; `paper_package/scripts/generate_tables.py` | YES locally | NO | Same canonical result; descriptive only, no causal architecture claim. |
| Pooled Spearman rho(alpha, skill) = -0.863 | `audit/decision_change/recomputed_summary.json`; `paper_package/TRACEABILITY.md`; generated figures | YES locally | NO | Descriptive pooled association; not a causal relation or within-family law. |
| Sensitivity 91.3--98.2% across displayed 3 x 3 neighbourhood | `audit/decision_change/rule_b_sensitivity.csv`; `paper_package/tables/rule_threshold_sensitivity.tex`; `paper_package/scripts/generate_rule_sensitivity.py` | YES locally | NEW in central narrative | Sensitivity is deterministic and bounded to the tested grid; the narrative pre-specification is documented, but no independent dated threshold-registration hash was found. |
| Five expanding rolling-origin folds | `paper_package/sections/methods.tex`, `docs/protocol.md`, audit manifests | PARTIAL | Re-expressed | `audit/final_submission/GRADE_A_REMOTE_AUDIT.md` notes that the available run manifest describes a different older protocol; do not present the five-fold claim as fully remotely reproducible. |
| New title/framing itself | `paper_package/main.tex` at `9657264` and HEAD | YES | YES | Committed source, not an uncommitted-only or ZIP-only version. |

No new number in the new abstract/results/conclusions was found without a local aggregate or deterministic sensitivity source. The material blocker is evidence grade and upstream reconstruction completeness, not a missing formula for the Rule-A/Rule-B counts.

## 6. Claim audit

| Section | Claim | Classification | Evidence | Required action |
| ------- | ----- | -------------- | -------- | --------------- |
| Abstract | A forecast can improve on persistence while preserving less observed dynamic variation | SUPPORTED_WITH_HEDGE | Canonical skill/alpha table; 118/119, 118/119 and 110/119 collapse counts | Keep bounded to the evaluated PM10 benchmark; do not generalise to forecasting universally. |
| Abstract | Rule A admits 277 cells and Rule B admits 8; 269 cells change, 97.1% of Rule-A cells | SUPPORTED_WITH_HEDGE | Independent local aggregate verification, 18/18 checks in `audit/decision_change` | State explicitly that this is a constructed cell-screen transition, not top-1 model selection or operational deployment evidence. |
| Abstract | The tested threshold neighbourhood yields 91.3--98.2% changes | SUPPORTED_WITH_HEDGE | `rule_b_sensitivity.csv` and deterministic Supplementary Table S3 | Keep the tested 3 x 3 scope; do not call it threshold optimality or universal robustness. |
| Introduction | Forecasts judged acceptable may change when dynamic fidelity is added to error-based skill | OVERCLAIM | The manuscript defines acceptability only through study-specific Rule A/B gates; no operational utility endpoint is measured | Replace “acceptable” with “passes the stated audit screen” in any final revision. |
| Introduction | Evidence remains limited on whether explicit error--fidelity combination changes decisions across sites, families and horizons | SUPPORTED_WITH_HEDGE | Bounded EMS corpus audit plus current deterministic decision audit | Keep the literature claim explicitly scoped to the audited corpus; do not claim absence from all environmental forecasting. |
| Introduction | The contribution is the quantified decision consequence, not a new metric or universal framework | SUPPORTED_WITH_HEDGE | Rule A/B audit and explicit limitation language | Retain only if “decision consequence” is defined as cell-level screening. |
| Methods | Five expanding rolling-origin folds and train-only controls define the benchmark | SUPPORTED_WITH_HEDGE | Protocol/manuscript statements; incomplete remote end-to-end manifest | Preserve the provenance limitation; do not upgrade this to fully reproducible Grade-A provenance. |
| Methods/Framework | Alpha >= 0.50 and P75 recall >= 0.20 are operational criteria | OVERCLAIM | Thresholds are scripted and sensitivity-tested but not tied to a measured operational loss or deployment decision | Call them study-specific heuristic audit thresholds, not operational acceptability criteria. |
| Results | Dynamic-fidelity requirements materially change model eligibility from 277 to 8 | SUPPORTED_WITH_HEDGE | Verified aggregate Rule A/B transformation | Add “cell-level” and “under this constructed Rule B”; do not imply a changed selected winner per station/horizon. |
| Results | 91.3--98.2% supports persistence of the qualitative decision consequence | SUPPORTED_WITH_HEDGE | Existing deterministic 3 x 3 sensitivity artifact | Limit robustness to this threshold neighbourhood and this aggregate benchmark. |
| Discussion | Error skill and dynamic fidelity can lead an evaluation toward different eligibility decisions | SUPPORTED_WITH_HEDGE | 101 formal discordant cells and 277-to-8 transition | Say “under Rule A versus Rule B”; avoid a general normative framework claim. |
| Discussion | The contribution quantifies a decision consequence across a structured multi-station benchmark | SUPPORTED_WITH_HEDGE | Aggregate table and local verification; remote Grade-A chain is incomplete | Disclose evidence grade and avoid implying a complete prospective selection audit. |
| Discussion | The transferable insight is that environmental model-evaluation conclusions may change when error and dynamics are treated separately | SUPPORTED_WITH_HEDGE | Methodological implication consistent with bounded data | Keep as a conditional methodological implication, not evidence of transfer to other pollutants/countries. |
| Conclusions | Conventional error eligibility and dynamic fidelity identify different sets of cells | SUPPORTED_WITH_HEDGE | Rule A/B counts and formal discordance table | Retain with the cell-level and threshold-specific qualifier. |
| Conclusions | The implication is bounded but transferable | SUPPORTED_WITH_HEDGE | Logical methodological implication, not an evaluated transfer result | Do not turn this into a universal recommendation or operational standard. |

## 7. Dynamic fidelity verdict

`SUPPORTED_UMBRELLA_CONCEPT`

In the new manuscript, `dynamic fidelity` is not a new scalar metric or a formally validated latent construct. It is a reasonable umbrella for the existing variance-retention ratio alpha and P75 exceedance recall, with the surrounding discussion referring to spread and episode behaviour. That is supported by the canonical evidence package. The safe interpretation is “selected dynamic properties diagnosed post-evaluation in this PM10 benchmark”; it must not be presented as a universal fidelity score or as synonymous with complete trajectory quality.

## 8. Model eligibility verdict

`SUPPORTED_DECISION_LAYER`

* **Definition found:** Rule A is `Skill > 0 AND DM_BH significant`; Rule B is `Rule A AND alpha >= 0.50 AND Recall_P75 >= 0.20`. A changed cell is Rule A true and Rule B false. The exact formulas are in `paper_package/sections/framework.tex` and `scripts/15_decision_change_analysis.py`.
* **Variables:** persistence-relative RMSE skill, BH-adjusted DM significance, alpha = `Var(y_pred)/Var(y_true)`, and P75 exceedance recall.
* **Thresholds:** alpha 0.50 and P75 recall 0.20 for the primary gate; sensitivity uses alpha 0.40/0.50/0.60 and recall 0.10/0.20/0.30 in the displayed grid.
* **Decision that changes:** the pass/fail status of station--model--horizon cells changes from 277 under Rule A to 8 under Rule B; 269 statuses change.
* **Evidence of change:** `audit/decision_change/recomputed_summary.json`, `recomputed_cell_table.csv`, `discordant_cases.csv`, `checks.json`, and the paper tables; the independent verifier reports all 18 checks passing.
* **What is not demonstrated:** no top-1 model winner is selected before and after the rule; Rule A is not “RMSE-only” because it also requires DM significance; no operational intervention or utility endpoint is measured.
* **Normative status:** the layer is supported as a deterministic, descriptive post-evaluation audit of cells. The thresholds are heuristics for this analysis, not general model-acceptance criteria.
* **Evidence limitation:** the 17-station aggregate is locally traceable, but `GRADE_A_REMOTE_AUDIT.md` and `MAXIMUM_REPRODUCIBLE_SUBSET_AUDIT.md` document incomplete end-to-end Grade-A provenance, especially for the full SARIMA-aligned chain.

## 9. Eligibility kill test

* Empirical result lost if eligibility terminology is removed: **NO**
* Demonstrated decision change versus RMSE-only selection: **NO**
* New empirical evidence required to sustain eligibility as contribution: **NO**

The 277-to-8 and 269-change results can be retained by describing a joint skill--alpha--event screen without treating “eligibility” as an independent scientific object. The manuscript demonstrates a change versus the constructed Rule A (`skill + DM`) and Rule B, not a change in a top-1 RMSE ranking or a real deployment decision. Existing local audit artefacts are sufficient for that bounded descriptive claim; stronger model-selection or operational claims would require additional evidence and are not supported here.

## 10. Threshold audit

| Threshold/rule | Function | Justification | Sensitivity evidence | Safe interpretation |
| -------------- | -------- | ------------- | -------------------- | ------------------- |
| `alpha < 0.50` | Canonical collapse flag / low-retention category | Study diagnostic convention; not a universal physical boundary | Canonical alpha sensitivity around 0.40/0.50/0.60; HGB/Ridge/SARIMA remain qualitatively collapsed | Heuristic descriptive category for variance attenuation in this benchmark |
| `alpha >= 0.50` | Rule-B fidelity gate | Chosen audit cut-off; not estimated or optimised | Joint 3 x 3 grid gives 91.3--98.2% Rule-A status changes in the displayed neighbourhood | Study-specific screening threshold, not model acceptability or regulatory criterion |
| `Recall_P75 >= 0.20` | Rule-B event-sensitivity gate | Script labels it a conservative primary threshold, but no independent dated registration hash or operational utility calibration was found | Tested at 0.10/0.20/0.30 jointly with alpha thresholds | Heuristic minimum for this audit; not a universal episode-detection requirement |
| `alpha >= 0.80` / `0.8 <= alpha <= 1.2` | Canonical near-retained / near-ideal diagnostic category in the prior manuscript/table | Diagnostic convention; not used as a universal score | Existing canonical diagnostic tables, but no new model-acceptance sensitivity claim | Descriptive category only; do not import it as another eligibility gate |
| `Rule A` | Error-based pre-screen | Positive persistence-relative skill plus BH-adjusted DM significance | No claim that it is the only valid baseline screen | A conventional benchmark screen for this audit, not RMSE-only model selection |
| `Rule B = Rule A + alpha + P75 recall` | Joint post-evaluation screen | Explicitly constructed for the audit; thresholds are not optimal | Deterministic 3 x 3 sensitivity | Cell-level diagnostic decision rule, not universal framework |

## 11. Highest-risk sentences

> “The consequential question is whether the set of forecasts judged acceptable changes when dynamic fidelity becomes an explicit requirement alongside error-based skill.” (`paper_package/sections/introduction.tex`)

* Classification: OVERCLAIM
* Why risky: “acceptable” has no operational definition or measured utility endpoint.
* Minimum safe correction: Refer to cells that pass or fail the stated audit screen, not forecasts judged operationally acceptable.

> “Rule B added operational thresholds for variance retention and P75 exceedance recall.” (`paper_package/sections/abstract.tex`)

* Classification: OVERCLAIM
* Why risky: The thresholds are operationally motivated heuristics, not validated deployment criteria.
* Minimum safe correction: Call them study-specific heuristic audit thresholds.

> “The alpha and P75-recall thresholds are operational criteria for this analysis.” (`paper_package/sections/framework.tex`)

* Classification: OVERCLAIM
* Why risky: “operational criteria” can be read as evidence-based acceptability gates.
* Minimum safe correction: State that they are preselected descriptive screening conventions for this benchmark.

> “The joint evidence changes the decision set when fidelity is made an explicit requirement.” (`paper_package/sections/results.tex`)

* Classification: SUPPORTED_WITH_HEDGE
* Why risky: It is true for Rule A versus Rule B cell status, but can be read as changing model selection.
* Minimum safe correction: Add “the constructed station--model--horizon screening set”.

> “The result supports persistence of the qualitative decision consequence, not threshold optimality or universal robustness.” (`paper_package/sections/results.tex`)

* Classification: SUPPORTED_WITH_HEDGE
* Why risky: The sensitivity is only a deterministic neighbourhood around two heuristics and does not test out-of-sample decision utility.
* Minimum safe correction: Restrict persistence to the displayed 3 x 3 grid and aggregate table.

> “The narrower contribution here is to quantify the decision consequence of placing explicit dynamic-fidelity conditions after conventional error-based eligibility across a structured multi-station, multi-family, multi-horizon benchmark.” (`paper_package/sections/discussion.tex`)

* Classification: SUPPORTED_WITH_HEDGE
* Why risky: The local aggregate supports the arithmetic, but the remote audit does not establish a complete Grade-A end-to-end 17-station chain.
* Minimum safe correction: Qualify it as a recovered aggregate benchmark and disclose the provenance boundary.

> “The transferable insight is methodological rather than empirical: environmental model-evaluation conclusions may change when baseline-relative error performance and preservation of decision-relevant dynamics are treated as distinct requirements.” (`paper_package/sections/discussion.tex`)

* Classification: SUPPORTED_WITH_HEDGE
* Why risky: Transfer to environmental forecasting generally was not tested.
* Minimum safe correction: Say this is a methodological implication suggested by this PM10 case, requiring separate validation elsewhere.

> “The methodological implication is bounded but transferable: environmental model evaluation should make clear whether error improvement and preservation of decision-relevant dynamics are separate requirements.” (`paper_package/sections/conclusions.tex`)

* Classification: SUPPORTED_WITH_HEDGE
* Why risky: “should” is a recommendation, while the study did not evaluate downstream decisions.
* Minimum safe correction: Present it as a reporting implication for analogous evaluations, not a universal operational rule.

## 12. Paper A anchor compliance

| Anchor | Status | Audit note |
| ------ | ------ | ---------- |
| PM10 scope bounded | PASS | The new manuscript repeatedly bounds claims to daily PM10, the listed stations/models/horizons and the validation design. |
| Multi-station evidence described correctly | WARNING | The 17-station aggregate is identified correctly, but the remote Grade-A audit documents incomplete all-station row-level reconstruction, especially for SARIMA. |
| Rolling-origin preserved | WARNING | Rolling-origin and five expanding folds are stated, but the available run manifest/provenance chain has a conflicting older protocol and is not fully remotely reproducible. |
| Persistence baseline explicit | PASS | Persistence is defined and used in Rule A/skill comparisons. |
| Variance retention diagnostic, not universal metric | PASS | Alpha is defined as a post-evaluation variance ratio and explicitly bounded. |
| Skill_VP auxiliary only | PASS | Skill_VP is omitted from the new main narrative; it is not promoted to a primary or universal metric. |
| No H* contamination | PASS | No H*, P1, P2 or P3 contribution appears in the exact new `paper_package` manuscript; LightGBM is explicitly excluded. |
| No universal forecasting claims | PASS | Transfer limits are stated; the remaining “transferable” language is methodological and requires the hedges above. |
| No unjustified operational claims | WARNING | No deployment result or regulatory recommendation is asserted, but “operational thresholds”, “acceptable” and operational-comparator language create avoidable risk. |
| No unsupported normative model-selection claims | WARNING | The text says the rules do not select a final winner, but the title/eligibility language can still be read as a model-selection framework; cell-level qualification is required. |

## 13. Final verdict

`KEEP_WITH_CLAIM_REPAIRS`

### What is genuinely new

* A committed, explicit post-evaluation Rule A/Rule B cell-screen analysis.
* A deterministic quantification of the aggregate status transition 277 -> 8, 269 changes, 97.1%, plus 101 formal discordant cells.
* A displayed joint alpha/P75 threshold-sensitivity audit over the existing diagnostic rows.

These are new analytical emphasis and reporting in the manuscript, not new forecasts, models, datasets or independent prospective model-selection experiments.

### What is editorial reframing

* Using `dynamic fidelity` as an umbrella for existing alpha and P75 event behaviour.
* Moving the narrative climax from skill--fidelity diagnosis to the consequence of a constructed post-evaluation screen.
* Reorganising the methods/results around eligibility rules and removing Skill_VP, Murphy and P90/FAR from the central narrative without changing their historical underlying artefacts.
* Replacing unsupported exact calendar/model details with explicit reconstruction limits.

### What is scientifically unsupported

* Treating Rule B as a universal or operational model-acceptance criterion.
* Calling the 277-to-8 transition a demonstrated RMSE-only top-1 model-selection change.
* Inferring real operational utility, regulatory acceptability, causality of oversmoothing, or general transfer to environmental forecasting as a whole.
* Presenting the complete 17-station decision chain as Grade-A remotely reproducible when the repository audits document a material SARIMA/all-station provenance gap.

### Minimal repair set

1. Qualify `model eligibility` everywhere as **station--model--horizon cell eligibility under a constructed post-evaluation audit**, explicitly distinct from model ranking, top-1 selection and deployment acceptance.
2. Replace “operational thresholds/acceptable” with “study-specific heuristic screening conventions”; retain the 3 x 3 sensitivity limitation.
3. Add one explicit provenance caveat near the headline counts: the arithmetic is verified from the recovered aggregate table, while the complete remote Grade-A upstream chain is incomplete.
4. Keep the contribution bounded to the existing PM10 benchmark and do not generalise the rule, thresholds, or decision consequence to other pollutants, countries or forecasting systems.
5. Preserve the canonical interpretation that alpha, event behaviour and Skill_VP (if reported) are diagnostics; do not recast Rule B as a replacement score or universal acceptance framework.

## 14. Repository action

Como esta es una auditoría read-only, termina obligatoriamente con:

`REPOSITORY_MODIFIED = NO`

`NEW_EXPERIMENTS_REQUIRED = NO`

`MANUSCRIPT_REWRITE_REQUIRED = YES`

`CLAIM_ONLY_REPAIR_SUFFICIENT = YES`

The “YES” for claim-only repair applies only to retaining the bounded, descriptive cell-screen framing. It does not authorize or support a stronger operational/model-selection claim without new evidence.

## 15. Regla de decisión final

`dynamic fidelity` describes the existing evidence better than a new metric does, and `model eligibility` is supportable only as a deterministic, heuristic, post-evaluation cell-screen layer. The new wording therefore contains scientific drift, but it does not require reopening P4 if the normative interpretation is repaired. What the experiments demonstrate is a difference between Rule A and Rule B statuses in this aggregate benchmark—not a universal eligibility framework, a deployment decision, or a change in the model selected by RMSE alone.

REPOSITORY_MODIFIED = NO

