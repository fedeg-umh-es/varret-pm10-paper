# P4 Environmental Modelling & Software Final Kill Test

## 1. Repository and manuscript state

HEAD = a53583e8b591be6e11b926a3368e54d2f62c35af
BRANCH = codex/p4-lightgbm-ems-gap-audit
MANUSCRIPT = Dynamic Fidelity and Model Eligibility in Daily PM10 Forecasting: A Multi-Station Decision Audit
PACKAGE = dist/p4_ems_submission_final/
PACKAGE_SHA256 = cb88cdadc07418a66425623934a6ce38d59f59e12963c7289e5a0d3b77b267de
WORKTREE = /Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper

## 2. Current official EMS criteria

Official sources:

* Elsevier Environmental Modelling & Software Journal Overview & Guide for Authors (https://www.elsevier.com/journals/environmental-modelling-and-software/1364-8152/guide-for-authors, accessed August 2026)
* ScienceDirect Environmental Modelling & Software Aims and Scope (https://www.sciencedirect.com/journal/environmental-modelling-and-software, accessed August 2026)

CURRENT_SCOPE_VERIFIED = YES

Concise summary of:

* **Scope**: EMS seeks to advance the capacity to represent, understand, predict, and manage the behaviour of natural environmental systems across air, water, and land at all practical scales. It publishes research articles on generic frameworks, model development and evaluation, software & decision support systems, and integrated assessment.
* **Contribution expectations**: Articles must go beyond simple case studies or benchmark applications; they must contribute new knowledge regarding modelling methodology, general frameworks, software tools, or fundamental limitations of the environmental modelling process.
* **Software/methodology expectations**: Software contributions must present reusable software, decision-support tools, or generic computational frameworks. Methodological papers must provide transferable advances rather than dataset-specific empirical audits.
* **Relevant article requirements**: Research papers must demonstrate broad generalisability, explicit decision support or software capability, and rigorous evaluation embedded within an environmental modelling context.

## 3. Previous EMS editorial evidence

### ENVSOFT-D-26-00523

DATE = 2026-03-01
OUTCOME = PRE-REVIEW REJECT
RECORDED_REASON = No specific substantive reason recorded
EVIDENCE_STRENGTH = WEAK_SIGNAL

### ENVSOFT-D-26-01081

DATE = 2026-04-13
OUTCOME = EDITORIAL REJECT
RECORDED_REASON = scope + contribution
EVIDENCE_STRENGTH = STRONG_SIGNAL

HISTORICAL_RECORD_DIRECTLY_VERIFIED = NO

## 4. Manuscript identity as written

CENTRAL_OBJECT = Multi-station post-evaluation cell-screening audit comparing persistence-relative error skill with dynamic fidelity (variance retention and P75 recall) in daily PM10 forecasting.
CONTRIBUTION_TYPE = Empirical diagnostic evaluation / post-evaluation audit of existing models.
ENVIRONMENTAL_MODELLING_CONTENT = MINIMAL (Descriptive empirical time-series evaluation from 17 monitoring stations; no atmospheric physics, chemistry, meteorology integration, or process-based environmental modelling).
SOFTWARE_TOOL_CONTRIBUTION = REPOSITORY_ONLY (Reproducibility scripts and provenance manifest in GitHub repository; no reusable software platform, toolkit, or standalone modeling package).
TRANSFERABILITY_EVIDENCE = BOUNDED (The quantitative 97.1% transition rate and 277-to-8 cell counts are strictly bounded to this PM10 benchmark, stations, and threshold neighborhood; transferable insight is qualitative/conceptual only).

## 5. Three-minute editor test

1. **What is the paper's central object?** A post-evaluation screening audit evaluating 595 station–model–horizon cells in daily PM10 forecasting to compare an error-only screen (Rule A) with a joint error-plus-fidelity screen (Rule B).
2. **What exactly is new?** The empirical demonstration and quantification of cell-eligibility reduction (277 eligible cells down to 8, a 97.1% change rate) when variance retention ($\alpha \ge 0.50$) and P75 recall ($\ge 0.20$) are required alongside persistence-relative RMSE skill.
3. **Is the contribution a new method, new tool/software, new modelling capability, new scientific insight, or mainly an evaluation/diagnostic protocol?** It is predominantly an empirical evaluation and diagnostic audit protocol applied to an existing benchmark, not a new algorithm, tool, software, or atmospheric modelling capability.
4. **What would an EMS reader learn that is transferable beyond these PM10 experiments?** A cautionary evaluation principle: that baseline-relative error improvement can coexist with variance suppression, and that adding fidelity criteria alters which cells pass heuristic screens.
5. **Is that transferable insight explicit or only inferable?** Explicitly stated in the Discussion and Conclusions as an evaluation-design caution, but empirically demonstrated exclusively on the 17 Spanish PM10 series.
6. **Does the editor need to accept study-specific heuristic thresholds to see the contribution?** Yes. The core numerical consequence (277 $\to$ 8 cells, 97.1% change) explicitly depends on the adopted cut-offs ($\alpha = 0.50$, P75 recall = 0.20), though supported by a 3 $\times$ 3 sensitivity neighborhood (91.3%–98.2%).
7. **Does the paper look like an environmental-modelling paper or like a forecast-verification paper using PM10 as the test bed?** It looks distinctly like a statistical forecast-verification paper using PM10 monitoring data as an empirical test bed.

## 6. EMS kill question

PM10_REMOVAL_TEST = NO

Explanation: If PM10 is removed from the title and abstract, the paper becomes an abstract time-series verification exercise demonstrating that penalizing variance suppression reduces the number of models passing an error-based screen. It contains no atmospheric process mechanisms, no simulation engine, no decision-support software, and no generalized environmental-modelling methodology.

## 7. Methodological contribution

VARIANCE_RETENTION = Standard descriptive variance ratio ($\alpha = \operatorname{Var}(\hat{y})/\operatorname{Var}(y)$); not methodologically novel, but utilized as an explicit post-evaluation diagnostic.
DYNAMIC_FIDELITY = Supported umbrella framing and interpretive label over existing verification dimensions (variance ratio and event recall), not a new formal mathematical construct.
MODEL_ELIGIBILITY = Heuristic post-evaluation cell-screening layer (Rule A vs Rule B); explicitly disclaims being an operational acceptance rule, top-1 selector, ranking method, or deployment certification framework.
RULE_A_RULE_B_CONSEQUENCE = Establishes a concrete empirical consequence (277 $\to$ 8 cells, 97.1% change, 101 discordant cells) within this specific benchmark; it is not a universally reusable algorithmic advance.
NON_OBVIOUS_CONTRIBUTION = MODERATE (The magnitude and multi-station ubiquity of the collapse—97.1% reduction across 17 stations—provide striking empirical evidence for air quality forecasting, even though the statistical principle that minimum-MSE predictors attenuate variance is well known in verification theory).

## 8. Comparison with previous EMS submissions

| Dimension | ENVSOFT-D-26-00523 | ENVSOFT-D-26-01081 | P4 | Materially different? |
| --------- | ------------------ | ------------------ | -- | --------------------- |
| Central object | Operational predictability limits in multi-step PM10 forecasting | Long-term skill and evaluation limits in air quality forecasting | Dynamic fidelity and model eligibility audit in PM10 forecasting | Partly (shifted from predictability/skill limits to post-evaluation fidelity audit) |
| Contribution type | Empirical forecast evaluation / limit study | Empirical forecast verification critique | Empirical diagnostic post-evaluation audit | NO (remains purely an empirical evaluation study) |
| Forecasting/eval orientation | Operational multi-step forecasting | Air quality forecast verification | Multi-station rolling-origin post-evaluation | NO (same core methodological paradigm) |
| Environmental interpretation | Atmospheric PM10 forecasting | Air quality forecasting | PM10 monitoring series (descriptive) | NO (no process modelling or meteorology) |
| Software/tool contribution | None (scripts) | None (scripts) | Repository only (reproducible scripts and data) | NO (no reusable software tool or package) |
| Evidence breadth | Unknown / narrow | Multi-station air quality | 17 stations, 5 model families, 7 horizons (595 cells) | YES (much more structured and robustly audited) |
| Operational framing | Operational predictability limits | Operational forecasting critique | Heuristic post-evaluation cell screen | YES (bounded claims, strictly non-operational) |
| Methodological consequence | Predictability horizon bound | Skill inflation diagnostic | Quantified screening divergence (277 $\to$ 8) | YES (concrete empirical audit metric) |

P4_CORRECTS_01081_SCOPE_CONTRIBUTION_WEAKNESS = NO

*Rationale*: While P4 substantially improves empirical rigor, data breadth (17 stations, 595 cells), and claim modesty, it does not change the underlying contribution type. It remains an empirical forecasting/verification evaluation without a reusable software tool, generic modeling framework, or environmental process modelling component—the exact weakness flagged in ENVSOFT-D-26-01081.

## 9. EMS desk-reject gates

| Gate | Status | Evidence | Likely editor objection | Different paper required? |
| ---- | ------ | -------- | ----------------------- | ------------------------- |
| 1. Formal gate | PASS | Complete elsarticle package, verified builds, clean formatting. | None. | NO |
| 2. Policy gate | PASS | Explicit declarations, CRediT, AI disclosure, reproducibility link. | None. | NO |
| 3. Scope gate | WARNING | Environmental forecasting is listed under EMS application areas, but core focus is on modelling software and generic frameworks. | "This manuscript is an empirical forecast verification case study rather than environmental modelling and software development." | NO |
| 4. Contribution-type gate | FAIL | P4 is an empirical diagnostic audit without a software tool, new algorithm, or generic modeling platform. | "EMS prioritises generic modelling frameworks, software tools, and transferable computational systems. Pure empirical benchmark audits belong in domain or forecasting journals." | YES |
| 5. Evidence-breadth gate | PASS | 17 Spanish MITECO/EMEP stations, 5 model families, 7 horizons, 5 expanding rolling folds (595 cells). | None (evidence is solid for the chosen benchmark). | NO |
| 6. Problem/framing gate | WARNING | Eligibility is explicitly defined as heuristic cell screening with study-specific thresholds, denying universal or operational validity. | "The primary finding relies on heuristic cut-offs without providing a validated, generalizable decision framework." | NO |
| 7. Execution/context gate | PASS | Provenance manifest, deterministic generators, DM/HLN testing, BH adjustment, threshold sensitivity. | None (execution is rigorous). | NO |

## 10. Repeated rejection signal

COMBINED_SIGNAL = STRONG_SIGNAL

STABLE_SIGNAL = The P4 manuscript line is inherently an empirical time-series forecasting and forecast-verification evaluation.
VENUE_SPECIFIC_SIGNAL = EMS has consistently rejected manuscripts from this line at the desk stage (ENVSOFT-D-26-00523 pre-review reject; ENVSOFT-D-26-01081 editorial reject for "scope + contribution") because empirical evaluation alone does not meet the journal's software/modelling contribution threshold.

## 11. Strongest case for EMS

* **Empirical and statistical rigor**: Comprehensive benchmark covering 17 MITECO/EMEP monitoring stations across 5 diverse model families and 7 horizons (595 structured cells), evaluated via expanding rolling-origin validation with formal Diebold–Mariano / HLN testing and Benjamini–Hochberg FDR control.
* **Striking, quantified finding**: Demonstrates that 97.1% of error-eligible cells (269 of 277) fail joint dynamic fidelity criteria, exposing a critical blind spot in standard air quality model evaluation.
* **Methodological sensitivity analysis**: Proves that the decision consequence persists robustly across a 3 $\times$ 3 threshold neighborhood (91.3%–98.2% status change), showing that the finding is not an artifact of a single arbitrary threshold.
* **Exemplary reproducibility**: Fully documented open-science package with canonical provenance manifest, deterministic figure/table generators, and complete traceability.
* **Formal alignment with model evaluation**: Directly addresses environmental model evaluation, which is explicitly recognized within EMS's stated scope.

## 12. Strongest case against EMS

* **Absence of software or tool contribution**: The manuscript provides no reusable software library, decision-support platform, simulation software, or computational tool, which represents a primary pillar of EMS's editorial mission.
* **Contribution-type mismatch**: The paper is fundamentally an empirical forecast-verification diagnostic audit, whereas EMS seeks advances in environmental modeling systems, process representation, or generic simulation frameworks.
* **Heuristic and non-universal framing**: The paper explicitly disclaims being a universal decision framework, top-1 selector, or deployment certification method, limiting its methodological footprint to a bounded empirical demonstration.
* **Minimal environmental process content**: The models are statistical time-series models evaluated purely on empirical monitoring records without atmospheric chemistry, physical transport, meteorology integration, or environmental process modeling.
* **Direct editorial precedent**: ENVSOFT-D-26-01081 was rejected editorially for "scope + contribution", and P4 maintains the exact same core contribution type and domain framing without introducing software or general modeling methods.

## 13. Risk estimate

DESK_REJECT_RISK = HIGH

Main failure gate: Contribution-type gate (EMS desk rejection due to lack of a software tool or generic modelling framework, echoing the ENVSOFT-D-26-01081 editorial rejection).

## 14. Final decision

DO_NOT_SEND_TO_EMS

## 15. Decision rationale

P4 is an empirical diagnostic audit that rigorously quantifies the tension between error skill and dynamic fidelity in daily PM10 forecasting. However, its fixed scientific identity provides neither a software platform nor a generic environmental modeling framework. Official EMS editorial criteria and recent precedent (ENVSOFT-D-26-01081, rejected for "scope + contribution") establish that pure empirical forecasting verification audits fall below the journal's contribution threshold. Resolving this mismatch would require fundamentally altering P4 into a software package or process-modelling paper, which violates the frozen manuscript identity. Submitting P4 as written to EMS presents a very high probability of immediate desk rejection.

## 16. Consequence for current package

PACKAGE_STATUS = PRESERVE_DO_NOT_SUBMIT

## 17. Repository integrity

MANUSCRIPT_MODIFIED = NO
SCIENTIFIC_CODE_MODIFIED = NO
CANONICAL_EVIDENCE_MODIFIED = NO
NUMERICAL_RESULTS_MODIFIED = NO
NEW_EXPERIMENTS_RUN = NO
PACKAGE_MODIFIED = NO
