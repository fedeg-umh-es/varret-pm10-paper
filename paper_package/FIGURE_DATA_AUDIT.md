# Figure data audit

All primary figures were regenerated locally from `outputs/tables/master_diagnostic_table.csv` with `paper_package/scripts/generate_figures.py`. The script asserts 595 rows, the five canonical models, no duplicate station--model--horizon keys, and zero excluded model rows.

| Figure | Source | Rows | Models | Script |
|---|---|---:|---|---|
| Figure 1: Skill versus alpha | `outputs/tables/master_diagnostic_table.csv` | 595 | 5 | `paper_package/scripts/generate_figures.py::figure_skill_alpha` |
| Figure 2: Horizon-wise behaviour | `outputs/tables/master_diagnostic_table.csv` | 595 | 5 | `paper_package/scripts/generate_figures.py::figure_horizon_profiles` |
| Figure 3: Rule A versus Rule B | `outputs/tables/master_diagnostic_table.csv` | 595 | 5 | `paper_package/scripts/generate_figures.py::figure_eligibility` |
| Supplementary reporting audit | EMS corpus audit artefacts | N/A | N/A | pre-existing audited artefact |

The three primary PDFs are vector outputs and are the only empirical figures included by the main text.
