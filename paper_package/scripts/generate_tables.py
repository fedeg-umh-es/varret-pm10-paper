"""
Generate all manuscript tables from canonical evidence.
Run from paper_package/scripts/ or repo root with:
    python3 paper_package/scripts/generate_tables.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path('/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper')
CANONICAL = REPO / 'outputs' / 'tables' / 'master_diagnostic_table.csv'
TABLES_OUT = REPO / 'paper_package' / 'tables'
TABLES_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CANONICAL)
assert len(df) == 595, f"Expected 595 rows, got {len(df)}"
assert 'lightgbm_direct' not in df['model'].values, "LightGBM contamination detected!"

MODEL_LABELS = {
    'hgb_direct': 'HGB direct',
    'ridge_direct': 'Ridge direct',
    'sarima': 'SARIMA',
    'seasonal_naive': 'Seasonal naive',
    'stl_ridge_direct': 'STL+Ridge',
}
MODEL_ORDER = ['hgb_direct', 'ridge_direct', 'sarima', 'seasonal_naive', 'stl_ridge_direct']

# ── TABLE 2: Model-level performance summary ──────────────────────────────────
rows = []
for model in MODEL_ORDER:
    m = df[df['model'] == model]
    n = len(m)  # should be 119
    skill_med = m['skill'].median()
    alpha_med = m['alpha'].median()
    recall_mean = m['recall_p75'].mean()
    dm_sig = m['dm_significant'].sum()
    alpha_lt05 = (m['alpha'] < 0.5).sum()
    rule_a = ((m['skill'] > 0) & (m['dm_significant'])).sum()
    rule_b = ((m['skill'] > 0) & (m['dm_significant']) & (m['alpha'] >= 0.5) & (m['recall_p75'] >= 0.20)).sum()
    discordant = ((m['skill'] > 0) & (m['dm_significant']) & (m['recall_p75'] >= 0.20) & (m['alpha'] < 0.5)).sum()
    rows.append({
        'Model': MODEL_LABELS[model],
        'Cells': n,
        'Median Skill': f'{skill_med:.3f}',
        'Median $\\alpha$': f'{alpha_med:.3f}',
        'Collapse ($\\alpha<0.5$)': f'{alpha_lt05}/{n}',
        'DM sig.': f'{dm_sig}/{n}',
        'Rule A': f'{rule_a}/{n}',
        'Rule B': f'{rule_b}/{n}',
        'Discordant': f'{discordant}/{n}',
    })
table2_df = pd.DataFrame(rows)

tex2 = r"""\begin{table}[ht]
\centering
\caption{Per-model diagnostic summary across 595 station--model--horizon cells
(17 stations $\times$ 5 models $\times$ 7 horizons). Skill is persistence-relative
RMSE skill; $\alpha$ is variance retention; DM sig.\ counts BH-adjusted
Diebold--Mariano cells significant at $p<0.05$; Rule A: skill $>0$ and DM significant;
Rule B: Rule A and $\alpha\geq0.5$ and recall$_{P75}\geq0.20$; Discordant: Rule A,
recall$\geq0.20$, and $\alpha<0.5$. Medians and means computed over all cells per model.
Cells are not independent (shared stations and series); the central tendency columns are medians and the remaining columns are counts.}
\label{tab:model_summary}
\begin{tabular}{lrrrrrrrrr}
\toprule
Model & $n$ & Median Skill & Median $\alpha$ & Collapse & DM sig. & Rule A & Rule B & Discordant \\
\midrule
"""
for _, row in table2_df.iterrows():
    model = row['Model']
    cells = row['Cells']
    skill = row['Median Skill']
    alpha = row['Median $\\alpha$']
    collapse = row['Collapse ($\\alpha<0.5$)']
    dm = row['DM sig.']
    rule_a = row['Rule A']
    rule_b = row['Rule B']
    discordant = row['Discordant']
    tex2 += f"{model} & {cells} & {skill} & {alpha} & {collapse} & {dm} & {rule_a} & {rule_b} & {discordant} \\\\\n"

tex2 += r"""\bottomrule
\end{tabular}
\end{table}
"""

(TABLES_OUT / 'table2_model_summary.tex').write_text(tex2)
print("Wrote table2_model_summary.tex")

# ── TABLE 3: Decision rule effect ─────────────────────────────────────────────
rule_a_total = ((df['skill'] > 0) & (df['dm_significant'])).sum()
rule_b_total = ((df['skill'] > 0) & (df['dm_significant']) & (df['alpha'] >= 0.5) & (df['recall_p75'] >= 0.20)).sum()
changes = rule_a_total - rule_b_total
pct = changes / rule_a_total * 100

rho, pval = spearmanr(df['alpha'], df['skill'])

# Breakdown by model
rows3 = []
for model in MODEL_ORDER:
    m = df[df['model'] == model]
    ra = int(((m['skill'] > 0) & (m['dm_significant'])).sum())
    rb = int(((m['skill'] > 0) & (m['dm_significant']) & (m['alpha'] >= 0.5) & (m['recall_p75'] >= 0.20)).sum())
    rows3.append({'Model': MODEL_LABELS[model], 'Rule A': ra, 'Rule B': rb, 'Changes': ra - rb})

tex3 = r"""\begin{table}[ht]
\centering
\caption{Effect of adding dynamic-fidelity requirements (Rule B) to
error-based eligibility (Rule A) across the 595-cell empirical benchmark.
Rule A: skill $>0$ and BH-adjusted DM significant; Rule B: Rule A and
$\alpha\geq0.5$ and recall$_{P75}\geq0.20$. Decision changes indicate the
number of cells eligible under Rule A but not Rule B. Cells are not
independent observations.}
\label{tab:decision_rule}
\begin{tabular}{lrrr}
\toprule
Model & Rule A & Rule B & Changes \\
\midrule
"""
for r in rows3:
    tex3 += f"{r['Model']} & {r['Rule A']} & {r['Rule B']} & {r['Changes']} \\\\\n"

tex3 += r"""\midrule
"""
tex3 += f"\\textbf{{Total}} & \\textbf{{{rule_a_total}}} & \\textbf{{{rule_b_total}}} & \\textbf{{{changes}}} ({pct:.1f}\\%) \\\\\n"
tex3 += r"""\bottomrule
\end{tabular}
\end{table}
"""

(TABLES_OUT / 'table3_decision_rule.tex').write_text(tex3)
print("Wrote table3_decision_rule.tex")
print(f"\nRule A={rule_a_total}, Rule B={rule_b_total}, Changes={changes}, {pct:.1f}%")
print(f"rho(alpha,skill) Spearman = {rho:.3f}, p = {pval:.2e}")
