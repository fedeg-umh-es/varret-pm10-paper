# Figure data audit

All scientific figures in the submission package are deterministic data
plots generated from the frozen canonical evidence or its verified
deterministic Rule-B sensitivity artifact. No scientific figure was
created with generative AI.

| Figure | Source | Rows | Models | Script | LightGBM rows | Artwork class | Status |
|---|---|---:|---|---|---:|---|---|
| Figure 1: horizon profiles | `outputs/tables/master_diagnostic_table.csv` | 595 | 5 | `generate_figures.py::figure_horizon_profiles` | 0 | DETERMINISTIC_DATA_PLOT | PASS |
| Figure 2: skill--alpha | `outputs/tables/master_diagnostic_table.csv` | 595 | 5 | `generate_figures.py::figure_skill_alpha` | 0 | DETERMINISTIC_DATA_PLOT | PASS |
| Figure 3: Rule-B sensitivity | `audit/decision_change/rule_b_sensitivity.csv` verified against canonical table | 595-derived | 5 | `generate_rule_sensitivity.py` | 0 | DETERMINISTIC_DATA_PLOT | PASS |

Figure 2 uses shaded positive-skill/low-alpha geometry and black open
contours for the 101 formal discordant cells. The shaded region is not
itself the formal discordance definition.

The removed 486-abstract reporting figure and the former Rule-A/Rule-B
bar chart are excluded from the revised Paper-A manuscript and Overleaf
archive. `graphical_abstract_eligibility.pdf` is an
`AI_GENERATED_ARTWORK` draft and is marked
`AI_ASSISTED_DRAFT_NOT_FOR_SUBMISSION`; it is excluded from the v3
submission archive.
