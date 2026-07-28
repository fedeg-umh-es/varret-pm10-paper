# Paper A — canonical figures

Each figure points to a source table/script. Figures do not introduce numbers
absent from the source tables.

| figure_id | file | source_table | script | shows |
|---|---|---|---|---|
| F3 | `figure3_skill_profiles.pdf` | `outputs/tables/variance_retention_summary.csv` | `scripts/10_build_figures.py` | median persistence-relative RMSE skill by horizon (17 stations) |
| F4 (main) | `figure4_alpha_profiles.pdf` | `outputs/tables/variance_retention_summary.csv` | `scripts/10_build_figures.py` | median variance-retention ratio α by horizon; grey band α<0.5 (collapsed) |
| F5 (main) | `figure5_scatter_skill_alpha.pdf` | `outputs/tables/variance_retention_summary.csv` | `scripts/14_generate_skill_alpha_figure.py` | skill–α diagnostic space (single-reading trade-off) |
| F-thr | `figure_threshold_sensitivity.pdf` | threshold outputs | `scripts/06_threshold_sensitivity.py` | collapse-rate sensitivity to α threshold |
| F-exc | `figure_exceedance_recall.pdf` | exceedance outputs | `scripts/plot_exceedance_figure.py` | exceedance recall / false-alarm behaviour |
| F-mur | `figure_murphy_decomposition.pdf` | Murphy outputs | `scripts/plot_murphy_decomposition.py` | Murphy MSE decomposition by model |
| F6 | `figure6_station_collapse_rates.pdf` | `outputs/tables/model_family_diagnostic_summary.csv` | `scripts/10_build_figures.py` | ML collapse-rate station map |

**Main figure:** F4 (α profiles) paired with F5 (skill–α space) — they carry the
central skill-with-collapse story.

**Note:** the grey band in F4 marks α<0.5 on the **variance ratio** scale
(consistent with the corrected definition); it must not be relabelled as an
SD-ratio threshold.
