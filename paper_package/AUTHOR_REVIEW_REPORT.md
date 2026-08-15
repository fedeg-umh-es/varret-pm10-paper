# P4 Paper A Author Review

## 1. Repository state

* HEAD: `a8604fd4a1fca681e3e526b8e1827c2d3ce15281`
* Branch: `codex/p4-lightgbm-ems-gap-audit`
* Worktree: DIRTY, unchanged during this review; `git diff --check` PASS. Pre-existing ZIP, audit/build artefacts, claim-repair files, and other worktree state were preserved.
* QA verdict entering review: `READY_FOR_AUTHOR_REVIEW`

The controlling `PRE_SUBMISSION_QA.md`, `CLAIM_REPAIR_MATRIX.md`, and `P4_DYNAMIC_FIDELITY_MODEL_ELIGIBILITY_AUDIT.md` were read before reviewing the four authorial zones.

## 2. Abstract

ABSTRACT_AUTHOR_GATE = PASS

The Abstract states the empirical problem, design, benchmark scope, central Rule A/Rule B result, and tested sensitivity range. It defines eligibility as post-evaluation cell screening and explicitly excludes ranking and operational acceptance. Dynamic fidelity is used as a bounded umbrella for the reported diagnostics. No sentence requires repair.

| Sentence/claim | Status | Reason | Minimum repair |
| -------------- | ------ | ------ | -------------- |

No non-KEEP sentence identified.

## 3. Results turning point

TURNING_POINT_AUTHOR_GATE = MINOR_REPAIR

CENTRAL_RESULT_WORDING =
"Thus, positive error eligibility and adequate P75 recall can coexist with low retained variance in identifiable station--model--horizon cells."

TURNING_POINT_INTERPRETATION = SUPPORTED

The central turning point is directly supported by the 101 discordant cells, the skill--alpha figure, and the existing tables. It reports the observed skill--fidelity decoupling without causal explanation or deployment inference. The surrounding Rule A/Rule B transition is also bounded to cell status rather than final model choice. One adjacent sensitivity sentence contains an ambiguous residual use of “operational”; the required local repair is recorded in Section 6 and proposed separately without application.

Repairs, if any:

* Replace “one operational threshold pair” with “one study-specific threshold pair” in `paper_package/sections/results.tex:71`.

## 4. Discussion limitations

DISCUSSION_BOUNDARY_GATE = PASS

Check:

* heuristic eligibility: PASS
* no ranking implication: PASS
* no deployment certification: PASS
* thresholds bounded: PASS
* PM10 scope bounded: PASS
* dynamic fidelity complementary: PASS
* Skill_VP auxiliary: PASS

Discussion explicitly calls the thresholds study-specific heuristic screening conventions, describes cell-level screening outcomes, denies operational deployment utility, denies universal rules, states that no causal mechanism is established, and limits transfer to separate evaluations. Skill_VP is not present in this new central narrative and is not promoted or replaced by Rule B.

## 5. Conclusions

CONCLUSION_AUTHOR_GATE = PASS

FINAL_SENTENCE_STATUS = SAFE

The Conclusions end with a bounded negative claim: the results do not establish a universal rate, acceptance rule, operational benefit, or mechanism for other forecasting systems. The preceding sentence limits the observed counts to the evaluated pollutant, station network, model set, horizon range, and validation design. The conclusion is diagnostic and complementary, not a ranking, deployment, or metric-replacement rule.

Minimum repair, if needed:

NONE

## 6. Residual operational language

| File | Wording | Classification | Action |
| ---- | ------- | -------------- | ------ |
| `paper_package/sections/results.tex:71` | “one operational threshold pair” | AMBIGUOUS | Replace with “one study-specific threshold pair”; this preserves the same thresholds and result while removing any acceptance/deployment reading. |

Other occurrences in the four reviewed zones are `SAFE_CONTEXTUAL`: Discussion says the rate is not an operational acceptance rate and does not demonstrate operational deployment utility; Conclusions say the findings do not establish an operational benefit; Abstract says eligibility is not operational acceptance. No OVERREACH occurrence was found.

## 7. Required manuscript changes

1. Replace the single ambiguous phrase “one operational threshold pair” with “one study-specific threshold pair” in `paper_package/sections/results.tex`.

No changes are required to metrics, thresholds, numerical results, tables, figures, evidence, code, or scientific interpretation.

## 8. Final author-review verdict

MINOR_AUTHOR_REPAIRS_REQUIRED

## 9. Next permitted action

Author approval and application of the single proposed wording repair in `AUTHOR_REPAIR_PATCH.md`; no repository synchronisation or submission packaging before that approval.
