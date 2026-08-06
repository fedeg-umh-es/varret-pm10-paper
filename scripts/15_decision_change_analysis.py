"""
Script 15 — Decision-Change Analysis (Rule A vs Rule B)
=======================================================
Implements the minimum decisive experiment from the Novelty-Gap Memo (§7).

Rule A (Standard evaluation):
    Accept if skill > 0 AND dm_significant == True

Rule B (Operational joint evaluation):
    Accept if Rule A AND alpha >= ALPHA_THRESH AND recall_p75 >= RECALL_THRESH

Primary thresholds (pre-registered):
    ALPHA_THRESH  = 0.50  (collapse boundary already documented in paper)
    RECALL_THRESH = 0.20  (conservative minimum: 1 in 5 episodes detected)

Sensitivity analysis: 3×3 grid of (alpha, recall) thresholds.
Added-value analysis: incremental contribution of alpha beyond skill + recall.

Outputs
-------
outputs/tables/decision_change_results.csv     — cell-level table
outputs/tables/decision_change_summary.csv     — aggregated by model/horizon/type
outputs/tables/decision_sensitivity_matrix.csv — 9-threshold grid
outputs/tables/incremental_value_alpha.csv     — alpha added-value analysis
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
INPUT  = REPO / "outputs" / "tables" / "master_diagnostic_table.csv"
OUT    = REPO / "outputs" / "tables"

# ---------------------------------------------------------------------------
# Pre-registered primary thresholds
# ---------------------------------------------------------------------------
ALPHA_PRIMARY  = 0.50
RECALL_PRIMARY = 0.20

# Sensitivity grid
ALPHA_GRID  = [0.40, 0.50, 0.60]
RECALL_GRID = [0.10, 0.20, 0.30]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# Rule A
# ---------------------------------------------------------------------------
df["rule_a"] = (df["skill"] > 0) & (df["dm_significant"] == True)

# ---------------------------------------------------------------------------
# Rule B (primary thresholds)
# ---------------------------------------------------------------------------
df["rule_b"] = (
    df["rule_a"]
    & (df["alpha"] >= ALPHA_PRIMARY)
    & (df["recall_p75"] >= RECALL_PRIMARY)
)

# Decision change: would have been accepted by A, rejected by B
df["decision_change"] = df["rule_a"] & ~df["rule_b"]

# Reason for change (among cells that change)
def classify_reason(row):
    if not row["rule_a"]:
        return "n/a"
    alpha_fail  = row["alpha"] < ALPHA_PRIMARY
    recall_fail = row["recall_p75"] < RECALL_PRIMARY
    if alpha_fail and recall_fail:
        return "both"
    elif alpha_fail:
        return "collapse_only"
    elif recall_fail:
        return "recall_only"
    else:
        return "no_change"

df["reason_change"] = df.apply(classify_reason, axis=1)

# Save cell-level table
cell_cols = [
    "station_id", "station_name", "station_type", "model", "horizon",
    "skill", "alpha", "dm_significant", "recall_p75",
    "rule_a", "rule_b", "decision_change", "reason_change",
    "collapse_flag", "f1_p75", "recall_p90", "f1_p90",
]
df[cell_cols].to_csv(OUT / "decision_change_results.csv", index=False)
print("Saved: decision_change_results.csv")

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
rule_a_cells = df[df["rule_a"]]
n_rule_a = len(rule_a_cells)
n_change  = df["decision_change"].sum()

print(f"\n=== PRIMARY RESULT (α≥{ALPHA_PRIMARY}, recall_p75≥{RECALL_PRIMARY}) ===")
print(f"Total cells:             {len(df)}")
print(f"Pass Rule A:             {n_rule_a}  ({100*n_rule_a/len(df):.1f}%)")
print(f"Pass Rule B:             {df['rule_b'].sum()}  ({100*df['rule_b'].sum()/n_rule_a:.1f}% of Rule A)")
print(f"Decision change:         {n_change}  ({100*n_change/n_rule_a:.1f}% of Rule A)")
print()
print("Reason breakdown (among changed decisions):")
changed = df[df["decision_change"]]
print(changed["reason_change"].value_counts().to_string())

# --- By model ---
by_model = []
for model, grp in df.groupby("model"):
    a = grp["rule_a"].sum()
    b = grp["rule_b"].sum()
    ch = grp["decision_change"].sum()
    pct = 100 * ch / a if a > 0 else np.nan
    by_model.append({
        "model": model,
        "rule_a_n": a,
        "rule_b_n": b,
        "decision_change_n": ch,
        "decision_change_pct_of_ruleA": round(pct, 1),
    })
by_model_df = pd.DataFrame(by_model).sort_values("decision_change_pct_of_ruleA", ascending=False)
print("\n--- By model ---")
print(by_model_df.to_string(index=False))

# --- By horizon ---
by_horizon = []
for h, grp in df.groupby("horizon"):
    a = grp["rule_a"].sum()
    b = grp["rule_b"].sum()
    ch = grp["decision_change"].sum()
    pct = 100 * ch / a if a > 0 else np.nan
    by_horizon.append({
        "horizon": h,
        "rule_a_n": a,
        "rule_b_n": b,
        "decision_change_n": ch,
        "decision_change_pct_of_ruleA": round(pct, 1),
    })
by_horizon_df = pd.DataFrame(by_horizon).sort_values("horizon")
print("\n--- By horizon ---")
print(by_horizon_df.to_string(index=False))

# --- By station type ---
by_type = []
for stype, grp in df.groupby("station_type"):
    a = grp["rule_a"].sum()
    b = grp["rule_b"].sum()
    ch = grp["decision_change"].sum()
    pct = 100 * ch / a if a > 0 else np.nan
    by_type.append({
        "station_type": stype,
        "rule_a_n": a,
        "rule_b_n": b,
        "decision_change_n": ch,
        "decision_change_pct_of_ruleA": round(pct, 1),
    })
by_type_df = pd.DataFrame(by_type)
print("\n--- By station type ---")
print(by_type_df.to_string(index=False))

# Save summary
summary_rows = []
summary_rows.append({"group": "overall", "label": "all",
    "rule_a_n": n_rule_a, "rule_b_n": df["rule_b"].sum(),
    "decision_change_n": int(n_change),
    "decision_change_pct_of_ruleA": round(100*n_change/n_rule_a, 1)})
for _, r in by_model_df.iterrows():
    summary_rows.append({"group": "model", "label": r["model"],
        "rule_a_n": r["rule_a_n"], "rule_b_n": r["rule_b_n"],
        "decision_change_n": r["decision_change_n"],
        "decision_change_pct_of_ruleA": r["decision_change_pct_of_ruleA"]})
for _, r in by_horizon_df.iterrows():
    summary_rows.append({"group": "horizon", "label": f"h{int(r['horizon'])}",
        "rule_a_n": r["rule_a_n"], "rule_b_n": r["rule_b_n"],
        "decision_change_n": r["decision_change_n"],
        "decision_change_pct_of_ruleA": r["decision_change_pct_of_ruleA"]})
for _, r in by_type_df.iterrows():
    summary_rows.append({"group": "station_type", "label": r["station_type"],
        "rule_a_n": r["rule_a_n"], "rule_b_n": r["rule_b_n"],
        "decision_change_n": r["decision_change_n"],
        "decision_change_pct_of_ruleA": r["decision_change_pct_of_ruleA"]})

pd.DataFrame(summary_rows).to_csv(OUT / "decision_change_summary.csv", index=False)
print("\nSaved: decision_change_summary.csv")

# ---------------------------------------------------------------------------
# Sensitivity matrix (3×3 grid)
# ---------------------------------------------------------------------------
sens_rows = []
for alpha_t, recall_t in product(ALPHA_GRID, RECALL_GRID):
    rule_b_s = (
        df["rule_a"]
        & (df["alpha"] >= alpha_t)
        & (df["recall_p75"] >= recall_t)
    )
    ch_s = (df["rule_a"] & ~rule_b_s).sum()
    pct_s = 100 * ch_s / n_rule_a if n_rule_a > 0 else np.nan
    # Further breakdown by reason
    alpha_fail_s  = df["rule_a"] & (df["alpha"] < alpha_t)
    recall_fail_s = df["rule_a"] & (df["recall_p75"] < recall_t)
    both_s  = (alpha_fail_s & recall_fail_s).sum()
    a_only  = (alpha_fail_s & ~recall_fail_s).sum()
    r_only  = (~alpha_fail_s & recall_fail_s).sum()
    sens_rows.append({
        "alpha_thresh": alpha_t,
        "recall_thresh": recall_t,
        "rule_a_n": n_rule_a,
        "pass_rule_b_n": rule_b_s.sum(),
        "decision_change_n": int(ch_s),
        "decision_change_pct": round(pct_s, 1),
        "collapse_only_n": int(a_only),
        "recall_only_n": int(r_only),
        "both_n": int(both_s),
        "primary": (alpha_t == ALPHA_PRIMARY and recall_t == RECALL_PRIMARY),
    })

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(OUT / "decision_sensitivity_matrix.csv", index=False)
print("Saved: decision_sensitivity_matrix.csv")

print("\n=== SENSITIVITY MATRIX (decision_change_pct_of_ruleA) ===")
pivot = sens_df.pivot(index="alpha_thresh", columns="recall_thresh", values="decision_change_pct")
print(pivot.to_string())

# ---------------------------------------------------------------------------
# Added value of alpha (§8 of memo)
# ---------------------------------------------------------------------------
av_rows = []

# Spearman correlations of alpha with other metrics (full corpus)
from scipy.stats import spearmanr

metrics_to_correlate = {
    "recall_p75": df["recall_p75"],
    "recall_p90": df["recall_p90"],
    "skill":      df["skill"],
    "dm_significant": df["dm_significant"].astype(int),
    "f1_p75":     df["f1_p75"],
}
print("\n=== ALPHA vs OTHER METRICS — Spearman correlations (full corpus) ===")
for name, col in metrics_to_correlate.items():
    valid = df[["alpha", name]].dropna()
    if len(valid) < 10:
        continue
    rho, pval = spearmanr(valid["alpha"], valid[name])
    print(f"  α vs {name:20s}  rho={rho:+.3f}  p={pval:.3e}")
    av_rows.append({"comparison": f"alpha_vs_{name}", "spearman_rho": round(rho, 4),
                    "p_value": round(pval, 6), "n": len(valid)})

# Cases discordant: pass Rule A, pass recall ≥ 0.20, but FAIL alpha < 0.50
# → Only detectable with alpha, not with recall or skill alone
discordant = df[
    df["rule_a"] &
    (df["recall_p75"] >= RECALL_PRIMARY) &
    (df["alpha"] < ALPHA_PRIMARY)
]
print(f"\n=== DISCORDANT CASES (only detectable with alpha) ===")
print(f"Pass Rule A + recall ≥ {RECALL_PRIMARY} but fail alpha < {ALPHA_PRIMARY}: {len(discordant)} cells")
if len(discordant) > 0:
    print("By model:")
    print(discordant.groupby("model").size().to_string())
    print("By horizon:")
    print(discordant.groupby("horizon").size().to_string())
    print("Alpha distribution in discordant cells:")
    print(discordant["alpha"].describe().to_string())

av_rows.append({"comparison": "discordant_alpha_only",
    "spearman_rho": np.nan, "p_value": np.nan,
    "n": len(discordant),
    "note": f"Pass RuleA+recall>={RECALL_PRIMARY} but fail alpha<{ALPHA_PRIMARY}"})

# Logistic incremental: can alpha predict "fails Rule B" beyond skill + recall?
# Target: among Rule A passers, does alpha add signal for "would be rejected by B"?
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

target_df = df[df["rule_a"]].copy()
target_df["target"] = (~target_df["rule_b"]).astype(int)  # 1 = decision change

if target_df["target"].nunique() > 1 and len(target_df) >= 20:
    # Model 1: skill + recall_p75 only
    X1 = target_df[["skill", "recall_p75"]].fillna(0).values
    # Model 2: skill + recall_p75 + alpha
    X2 = target_df[["skill", "recall_p75", "alpha"]].fillna(0).values
    y  = target_df["target"].values

    scaler1 = StandardScaler().fit(X1)
    scaler2 = StandardScaler().fit(X2)

    lr1 = LogisticRegression(max_iter=1000).fit(scaler1.transform(X1), y)
    lr2 = LogisticRegression(max_iter=1000).fit(scaler2.transform(X2), y)

    auc1 = roc_auc_score(y, lr1.predict_proba(scaler1.transform(X1))[:, 1])
    auc2 = roc_auc_score(y, lr2.predict_proba(scaler2.transform(X2))[:, 1])

    print(f"\n=== LOGISTIC AUC — predicting 'decision change' among Rule A passers ===")
    print(f"  Model 1 (skill + recall_p75):         AUC = {auc1:.3f}")
    print(f"  Model 2 (skill + recall_p75 + alpha): AUC = {auc2:.3f}")
    print(f"  Incremental AUC from alpha:           Δ = {auc2-auc1:+.3f}")

    av_rows.append({"comparison": "logistic_auc_without_alpha",
        "spearman_rho": np.nan, "p_value": np.nan, "n": len(target_df),
        "note": f"AUC predicting decision_change: {auc1:.4f}"})
    av_rows.append({"comparison": "logistic_auc_with_alpha",
        "spearman_rho": np.nan, "p_value": np.nan, "n": len(target_df),
        "note": f"AUC predicting decision_change: {auc2:.4f}"})
    av_rows.append({"comparison": "logistic_delta_auc_alpha",
        "spearman_rho": np.nan, "p_value": np.nan, "n": len(target_df),
        "note": f"Incremental AUC from adding alpha: {auc2-auc1:+.4f}"})

pd.DataFrame(av_rows).to_csv(OUT / "incremental_value_alpha.csv", index=False)
print("\nSaved: incremental_value_alpha.csv")

# ---------------------------------------------------------------------------
# Final summary printout for report
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY FOR GO/NO-GO EVALUATION")
print("="*60)
print(f"Rule A passers:          {n_rule_a} / {len(df)} ({100*n_rule_a/len(df):.1f}%)")
print(f"Decision changes:        {n_change} / {n_rule_a} ({100*n_change/n_rule_a:.1f}% of accepted)")
print(f"  — collapse_only:       {(df['reason_change']=='collapse_only').sum()}")
print(f"  — recall_only:         {(df['reason_change']=='recall_only').sum()}")
print(f"  — both:                {(df['reason_change']=='both').sum()}")
print(f"Discordant (α only):     {len(discordant)} cells (pass recall, fail alpha)")
print()
print("Sensitivity range (min–max decision_change_pct across 9 thresholds):")
print(f"  Min: {sens_df['decision_change_pct'].min():.1f}%  Max: {sens_df['decision_change_pct'].max():.1f}%")
print()
print("All outputs written to:", OUT)
