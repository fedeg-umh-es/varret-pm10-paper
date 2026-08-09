# Figure data audit

All scientific figures in the manuscript are deterministic data plots.
They are generated from the canonical table or the verified
deterministic Rule-B sensitivity artifact. No scientific figure was
created with generative AI.

| Figure | Canonical data | Subset | Rows | Models | Script | Transformation | LightGBM rows | Status |
|---|---|---|---:|---:|---|---|---:|---|
| Figure 1: horizon profiles | `outputs/tables/master_diagnostic_table.csv` | all rows, grouped by model and horizon | 595 | 5 | `generate_figures.py::figure_horizon_profiles` | median skill and alpha by horizon | 0 | DETERMINISTIC_DATA_PLOT / PASS |
| Figure 2: faceted skill--alpha | `outputs/tables/master_diagnostic_table.csv` | all rows, faceted by model; formal discordance contoured | 595 | 5 | `generate_result_set_figures.py::candidate_a` | direct scatter; open contours for formal discordance | 0 | DETERMINISTIC_DATA_PLOT / PASS |
| Figure 3: Rule-B sensitivity | verified `audit/decision_change/rule_b_sensitivity.csv` | alpha .40/.50/.60 x P75 recall .10/.20/.30 | 595-derived | 5 | `generate_rule_sensitivity.py` | retained-cell change percentage | 0 | DETERMINISTIC_DATA_PLOT / PASS |

Figure 2's shaded region denotes only positive skill with alpha below
0.50. It is not the formal discordance definition. Open contours identify
the 101 cells that also satisfy Rule A and P75 recall at least 0.20.

The existing `figures/graphical_abstract_eligibility.pdf` is an
`AI_ASSISTED_DRAFT_NOT_FOR_SUBMISSION`; it is not a scientific figure and
is excluded from the v5 archive. The old 486-abstract figure, former
eligibility bar chart, and unused legacy figures are also excluded.

`FIGURE_2_VISUAL_QA = PASS`: the final file uses five common-axis facets
and a sixth key cell; formal discordance is shown with open contours and
is distinguished from the shaded positive-skill/low-alpha region.
