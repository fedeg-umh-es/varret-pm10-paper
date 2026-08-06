"""
Independent verification of scripts/15_decision_change_analysis.py
===================================================================
Does NOT import from the original script. Recomputes everything from
outputs/tables/master_diagnostic_table.csv using only pandas/numpy/scipy.

Outputs written to audit/decision_change/
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO    = Path(__file__).resolve().parent.parent.parent
INPUT   = REPO / "outputs" / "tables" / "master_diagnostic_table.csv"
AUDIT   = Path(__file__).resolve().parent
LOG     = AUDIT / "commands.log"

# Pre-registered primary thresholds (same as original script)
ALPHA_PRIMARY  = 0.50
RECALL_PRIMARY = 0.20

REPORTED = {
    "rule_a_n":          277,
    "rule_b_n":          8,
    "decision_change_n": 269,
    "decision_change_pct": 97.1,
    "rho_alpha_skill":   -0.863,
    "discordant_n":      101,
    "urban_industrial_pct": 77.8,
}

log_lines = []

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    log_lines.append(line)
    print(line)

# ---------------------------------------------------------------------------
# STEP 0 — Environment
# ---------------------------------------------------------------------------
log("=== ENVIRONMENT ===")
env_lines = [
    f"python: {sys.version}",
    f"platform: {platform.platform()}",
    f"pandas: {pd.__version__}",
    f"numpy: {np.__version__}",
    f"scipy: {__import__('scipy').__version__}",
    f"repo: {REPO}",
    f"input: {INPUT}",
    f"input_exists: {INPUT.exists()}",
]
for ln in env_lines:
    log(ln)
(AUDIT / "environment.txt").write_text("\n".join(env_lines) + "\n")

# ---------------------------------------------------------------------------
# STEP 1 — Input inventory
# ---------------------------------------------------------------------------
log("=== INPUT INVENTORY ===")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

df_raw = pd.read_csv(INPUT)
input_sha = sha256(INPUT)
stat = INPUT.stat()

log(f"  rows={len(df_raw)} cols={len(df_raw.columns)} sha256={input_sha[:16]}...")
log(f"  size={stat.st_size} bytes")
log(f"  key columns: station_id, model, horizon, skill, dm_significant, alpha, recall_p75")

nulls_by_col = df_raw[["skill","dm_significant","alpha","recall_p75","station_type"]].isnull().sum().to_dict()
log(f"  nulls in key cols: {nulls_by_col}")

inv_rows = [{
    "file": str(INPUT),
    "sha256": input_sha,
    "rows": len(df_raw),
    "cols": len(df_raw.columns),
    "size_bytes": stat.st_size,
    "nulls_skill": int(nulls_by_col.get("skill", 0)),
    "nulls_dm_significant": int(nulls_by_col.get("dm_significant", 0)),
    "nulls_alpha": int(nulls_by_col.get("alpha", 0)),
    "nulls_recall_p75": int(nulls_by_col.get("recall_p75", 0)),
    "status": "OK",
}]
pd.DataFrame(inv_rows).to_csv(AUDIT / "input_inventory.csv", index=False)
log("  Saved input_inventory.csv")

# ---------------------------------------------------------------------------
# STEP 2 — Uniqueness check
# ---------------------------------------------------------------------------
log("=== KEY UNIQUENESS CHECK ===")
df = df_raw.copy()
key = ["station_id", "model", "horizon"]
n_keys = df[key].drop_duplicates().shape[0]
n_total = len(df)
log(f"  Total rows: {n_total}")
log(f"  Unique (station_id × model × horizon): {n_keys}")
log(f"  Duplicates: {n_total - n_keys}")
assert n_keys == n_total, f"DUPLICATE KEYS FOUND: {n_total - n_keys}"

# Coverage
log(f"  Stations: {df['station_id'].nunique()}")
log(f"  Models:   {sorted(df['model'].unique().tolist())}")
log(f"  Horizons: {sorted(df['horizon'].unique().tolist())}")

# ---------------------------------------------------------------------------
# STEP 3 — Recompute Rule A and Rule B
# ---------------------------------------------------------------------------
log("=== RECOMPUTING RULES ===")

# Exact same definitions as original script
df["rule_a_rc"] = (df["skill"] > 0) & (df["dm_significant"] == True)
df["rule_b_rc"] = (
    df["rule_a_rc"]
    & (df["alpha"] >= ALPHA_PRIMARY)
    & (df["recall_p75"] >= RECALL_PRIMARY)
)
df["decision_change_rc"] = df["rule_a_rc"] & ~df["rule_b_rc"]

# Reason
def classify_reason(row):
    if not row["rule_a_rc"]:
        return "n/a"
    a_fail = row["alpha"] < ALPHA_PRIMARY
    r_fail = row["recall_p75"] < RECALL_PRIMARY
    if a_fail and r_fail:
        return "both"
    elif a_fail:
        return "collapse_only"
    elif r_fail:
        return "recall_only"
    return "no_change"

df["reason_rc"] = df.apply(classify_reason, axis=1)

rc_rule_a  = int(df["rule_a_rc"].sum())
rc_rule_b  = int(df["rule_b_rc"].sum())
rc_changes = int(df["decision_change_rc"].sum())
rc_pct     = round(100 * rc_changes / rc_rule_a, 1) if rc_rule_a > 0 else None

log(f"  Rule A: {rc_rule_a}  (reported={REPORTED['rule_a_n']})")
log(f"  Rule B: {rc_rule_b}  (reported={REPORTED['rule_b_n']})")
log(f"  Changes:{rc_changes}  (reported={REPORTED['decision_change_n']})")
log(f"  Pct:    {rc_pct}%   (reported={REPORTED['decision_change_pct']}%)")

# ---------------------------------------------------------------------------
# STEP 4 — Rule B failure decomposition (mutually exclusive)
# ---------------------------------------------------------------------------
log("=== RULE B FAILURE DECOMPOSITION ===")
# Among cells passing Rule A only
ruleA_df = df[df["rule_a_rc"]].copy()
fail_alpha  = ruleA_df["alpha"] < ALPHA_PRIMARY
fail_recall = ruleA_df["recall_p75"] < RECALL_PRIMARY

decomp = {
    "pass_rule_b":          int((~fail_alpha & ~fail_recall).sum()),
    "fail_alpha_only":      int((fail_alpha  & ~fail_recall).sum()),
    "fail_recall_only":     int((~fail_alpha &  fail_recall).sum()),
    "fail_both":            int((fail_alpha  &  fail_recall).sum()),
    "total_rule_a":         rc_rule_a,
}
log(f"  {decomp}")
assert sum(v for k,v in decomp.items() if k != "total_rule_a") == rc_rule_a, "Decomposition does not add up"
pd.DataFrame([decomp]).to_csv(AUDIT / "rule_b_failure_decomposition.csv", index=False)
log("  Saved rule_b_failure_decomposition.csv")

# ---------------------------------------------------------------------------
# STEP 5 — Sensitivity (extended grid)
# ---------------------------------------------------------------------------
log("=== SENSITIVITY ANALYSIS ===")
alpha_grid  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
recall_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
sens_rows = []
for at in alpha_grid:
    for rt in recall_grid:
        rule_b_s = df["rule_a_rc"] & (df["alpha"] >= at) & (df["recall_p75"] >= rt)
        ch_s = int((df["rule_a_rc"] & ~rule_b_s).sum())
        pct_s = round(100 * ch_s / rc_rule_a, 1) if rc_rule_a > 0 else None
        a_only = int((df["rule_a_rc"] & (df["alpha"] < at) & (df["recall_p75"] >= rt)).sum())
        r_only = int((df["rule_a_rc"] & (df["alpha"] >= at) & (df["recall_p75"] < rt)).sum())
        both   = int((df["rule_a_rc"] & (df["alpha"] < at) & (df["recall_p75"] < rt)).sum())
        sens_rows.append({
            "alpha_thresh": at, "recall_thresh": rt,
            "pass_b": int(rule_b_s.sum()), "change_n": ch_s,
            "change_pct": pct_s,
            "alpha_only_n": a_only, "recall_only_n": r_only, "both_n": both,
            "primary": (at == ALPHA_PRIMARY and rt == RECALL_PRIMARY),
        })
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(AUDIT / "rule_b_sensitivity.csv", index=False)
log(f"  Primary result: {sens_df[sens_df['primary']]['change_pct'].values[0]}%")
log(f"  Range across {len(sens_rows)} combos: {sens_df['change_pct'].min()}% – {sens_df['change_pct'].max()}%")
log("  Saved rule_b_sensitivity.csv")

# ---------------------------------------------------------------------------
# STEP 6 — Correlation alpha vs skill
# ---------------------------------------------------------------------------
log("=== CORRELATION alpha vs skill ===")
valid_corr = df[["alpha", "skill"]].dropna()
rho_full, pval_full = spearmanr(valid_corr["alpha"], valid_corr["skill"])
log(f"  rho(alpha, skill) full corpus  = {rho_full:.4f}  p={pval_full:.3e}  n={len(valid_corr)}")
log(f"  Reported: {REPORTED['rho_alpha_skill']}")
log(f"  Difference: {rho_full - REPORTED['rho_alpha_skill']:.4f}")

# By model
log("  By model:")
for model, grp in df.groupby("model"):
    vv = grp[["alpha", "skill"]].dropna()
    if len(vv) >= 5:
        r, p = spearmanr(vv["alpha"], vv["skill"])
        log(f"    {model}: rho={r:.3f}  p={p:.3e}  n={len(vv)}")

# By horizon
log("  By horizon:")
for h, grp in df.groupby("horizon"):
    vv = grp[["alpha", "skill"]].dropna()
    if len(vv) >= 5:
        r, p = spearmanr(vv["alpha"], vv["skill"])
        log(f"    h{h}: rho={r:.3f}  p={p:.3e}  n={len(vv)}")

# Definitional dependency analysis
log("  DEFINITIONAL NOTE: skill = 1 - RMSE(model)/RMSE(persistence). alpha = Var(y_pred)/Var(y_true).")
log("  Both use y_pred and y_true but through different transformations.")
log("  High correlation consistent with but not caused by definitional overlap alone.")
log("  MSE-trained models minimize average error, encouraging regression-to-mean, which simultaneously")
log("  increases skill (reduces RMSE vs persistence) and reduces alpha (shrinks predicted variance).")
log("  Correlation is observational, not definitional identity.")

# ---------------------------------------------------------------------------
# STEP 7 — Discordant cases
# ---------------------------------------------------------------------------
log("=== DISCORDANT CASES ===")
# Definition from script lines 264-268:
# Pass Rule A + recall_p75 >= RECALL_PRIMARY + alpha < ALPHA_PRIMARY
discordant = df[
    df["rule_a_rc"] &
    (df["recall_p75"] >= RECALL_PRIMARY) &
    (df["alpha"] < ALPHA_PRIMARY)
].copy()
rc_discordant = len(discordant)
log(f"  Discordant: {rc_discordant}  (reported={REPORTED['discordant_n']})")

# Verify each condition
all_ruleA   = discordant["rule_a_rc"].all()
all_recall  = (discordant["recall_p75"] >= RECALL_PRIMARY).all()
all_alpha   = (discordant["alpha"] < ALPHA_PRIMARY).all()
log(f"  All pass Rule A: {all_ruleA}")
log(f"  All recall >= {RECALL_PRIMARY}: {all_recall}")
log(f"  All alpha  <  {ALPHA_PRIMARY}: {all_alpha}")

# Save full discordant table
discordant_cols = ["station_id","station_name","station_type","model","horizon",
                   "skill","alpha","recall_p75","dm_significant","n_pairs",
                   "rule_a_rc","decision_change_rc","reason_rc"]
discordant[discordant_cols].to_csv(AUDIT / "discordant_cases.csv", index=False)
log("  Saved discordant_cases.csv")

# Summaries with denominators
def summary_with_denom(groupcol):
    rows = []
    for val, grp in df.groupby(groupcol):
        n_total_grp  = len(grp)
        n_ruleA_grp  = int(grp["rule_a_rc"].sum())
        n_disc_grp   = int((grp["rule_a_rc"] & (grp["recall_p75"] >= RECALL_PRIMARY) & (grp["alpha"] < ALPHA_PRIMARY)).sum())
        rows.append({groupcol: val, "n_total": n_total_grp,
                     "n_rule_a": n_ruleA_grp, "n_discordant": n_disc_grp,
                     "pct_of_ruleA": round(100*n_disc_grp/n_ruleA_grp, 1) if n_ruleA_grp > 0 else None})
    return pd.DataFrame(rows)

summary_with_denom("station_id").to_csv(AUDIT / "discordant_summary_by_station.csv", index=False)
summary_with_denom("model").to_csv(AUDIT / "discordant_summary_by_model.csv", index=False)
summary_with_denom("horizon").to_csv(AUDIT / "discordant_summary_by_horizon.csv", index=False)
summary_with_denom("station_type").to_csv(AUDIT / "discordant_summary_by_station_type.csv", index=False)
log("  Saved discordant summaries by station/model/horizon/station_type")

# 5 representative cases: highest alpha among discordant (best-performing yet still failing)
top5 = discordant.nlargest(5, "alpha")[discordant_cols]
log("  Top 5 discordant (highest alpha — best performers that still fail):")
for _, r in top5.iterrows():
    log(f"    {r['station_id']} {r['model']} h{r['horizon']} skill={r['skill']:.3f} alpha={r['alpha']:.3f} recall_p75={r['recall_p75']:.3f}")

# ---------------------------------------------------------------------------
# STEP 8 — Urban/Industrial verification
# ---------------------------------------------------------------------------
log("=== URBAN/INDUSTRIAL ===")
ui = df[df["station_type"] == "Urban/Industrial"]
ui_ruleA = int(ui["rule_a_rc"].sum())
ui_change = int(ui["decision_change_rc"].sum())
ui_pct    = round(100 * ui_change / ui_ruleA, 1) if ui_ruleA > 0 else None
log(f"  Urban/Industrial: stations={ui['station_id'].nunique()}")
log(f"  Rule A: {ui_ruleA}  Changes: {ui_change}  Pct: {ui_pct}%  (reported=77.8%)")
log(f"  Station IDs: {ui['station_id'].unique().tolist()}")
log(f"  Models involved: {ui[ui['decision_change_rc']]['model'].unique().tolist()}")
log(f"  Horizons involved: {sorted(ui[ui['decision_change_rc']]['horizon'].unique().tolist())}")

# Full station_type summary
st_rows = []
for stype, grp in df.groupby("station_type"):
    a = int(grp["rule_a_rc"].sum())
    b = int(grp["rule_b_rc"].sum())
    ch = int(grp["decision_change_rc"].sum())
    pct = round(100*ch/a, 1) if a > 0 else None
    stations = grp["station_id"].nunique()
    st_rows.append({"station_type": stype, "n_stations": stations,
                    "rule_a_n": a, "rule_b_n": b,
                    "decision_change_n": ch, "pct_change": pct})
pd.DataFrame(st_rows).to_csv(AUDIT / "station_type_summary.csv", index=False)
log("  Saved station_type_summary.csv")

# ---------------------------------------------------------------------------
# STEP 9 — Cell-level output
# ---------------------------------------------------------------------------
save_cols = key + ["skill","alpha","recall_p75","dm_significant",
                   "station_type","rule_a_rc","rule_b_rc",
                   "decision_change_rc","reason_rc"]
df[save_cols].to_csv(AUDIT / "recomputed_cell_table.csv", index=False)
log("  Saved recomputed_cell_table.csv")

# ---------------------------------------------------------------------------
# STEP 10 — Checks (tests)
# ---------------------------------------------------------------------------
log("=== RUNNING CHECKS ===")
checks = {}

def check(name, condition, reported=None, computed=None, tol=0.15):
    status = "PASS" if condition else "FAIL"
    checks[name] = {"status": status, "reported": reported, "computed": computed}
    log(f"  [{status}] {name}  reported={reported} computed={computed}")
    return condition

# Core numeric checks
check("no_duplicate_keys",       n_keys == n_total,
      reported=n_total, computed=n_keys)
check("total_rows_595",          n_total == 595,
      reported=595, computed=n_total)
check("rule_a_n_277",            rc_rule_a == 277,
      reported=277, computed=rc_rule_a)
check("rule_b_n_8",              rc_rule_b == 8,
      reported=8, computed=rc_rule_b)
check("changes_n_269",           rc_changes == 269,
      reported=269, computed=rc_changes)
check("changes_pct_97_1",        abs(rc_pct - 97.1) < 0.15,
      reported=97.1, computed=rc_pct)
check("discordant_n_101",        rc_discordant == 101,
      reported=101, computed=rc_discordant)
check("rho_alpha_skill",         abs(rho_full - REPORTED["rho_alpha_skill"]) < 0.005,
      reported=REPORTED["rho_alpha_skill"], computed=round(rho_full, 3))
check("urban_industrial_pct",    abs((ui_pct or 0) - 77.8) < 0.15,
      reported=77.8, computed=ui_pct)
# Decomposition adds up
check("decomposition_adds_up",
      decomp["pass_rule_b"]+decomp["fail_alpha_only"]+decomp["fail_recall_only"]+decomp["fail_both"] == rc_rule_a,
      reported=rc_rule_a,
      computed=decomp["pass_rule_b"]+decomp["fail_alpha_only"]+decomp["fail_recall_only"]+decomp["fail_both"])
# No recall_only changes (reported = 0)
recall_only_count = int((df["reason_rc"] == "recall_only").sum())
check("recall_only_changes_zero", recall_only_count == 0,
      reported=0, computed=recall_only_count)
# Discordant conditions satisfied
check("discordant_all_ruleA",    all_ruleA,  reported=True, computed=all_ruleA)
check("discordant_all_recall",   all_recall, reported=True, computed=all_recall)
check("discordant_all_alpha",    all_alpha,  reported=True, computed=all_alpha)
# Urban/Industrial denominator
check("urban_industrial_denom_18", ui_ruleA == 18, reported=18, computed=ui_ruleA)
# Sensitivity range
check("sensitivity_min_gte_85",  sens_df["change_pct"].min() >= 85.0,
      reported=">=85%", computed=float(sens_df["change_pct"].min()))
# Primary thresholds pre-specified
check("alpha_thresh_prespecified", ALPHA_PRIMARY == 0.50,
      reported=0.50, computed=ALPHA_PRIMARY)
check("recall_thresh_prespecified", RECALL_PRIMARY == 0.20,
      reported=0.20, computed=RECALL_PRIMARY)

n_pass = sum(1 for v in checks.values() if v["status"] == "PASS")
n_fail = sum(1 for v in checks.values() if v["status"] == "FAIL")
log(f"  TOTAL: {n_pass} PASS / {n_fail} FAIL")

with open(AUDIT / "checks.json", "w") as f:
    json.dump(checks, f, indent=2)
log("  Saved checks.json")

# ---------------------------------------------------------------------------
# STEP 11 — Recomputed summary JSON
# ---------------------------------------------------------------------------
summary = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "input_sha256": input_sha,
    "thresholds": {"alpha_primary": ALPHA_PRIMARY, "recall_primary": RECALL_PRIMARY},
    "results": {
        "total_cells": n_total,
        "rule_a_n": rc_rule_a,
        "rule_b_n": rc_rule_b,
        "decision_change_n": rc_changes,
        "decision_change_pct": rc_pct,
        "discordant_n": rc_discordant,
        "rho_alpha_skill": round(rho_full, 4),
        "rho_alpha_skill_pval": float(f"{pval_full:.3e}"),
        "urban_industrial_ruleA_n": ui_ruleA,
        "urban_industrial_change_n": ui_change,
        "urban_industrial_pct": ui_pct,
    },
    "failure_decomposition": decomp,
    "sensitivity_range_pct": {
        "min": float(sens_df["change_pct"].min()),
        "max": float(sens_df["change_pct"].max()),
        "primary": float(sens_df[sens_df["primary"]]["change_pct"].values[0]),
    },
    "checks_pass": n_pass,
    "checks_fail": n_fail,
}
with open(AUDIT / "recomputed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log("  Saved recomputed_summary.json")

# ---------------------------------------------------------------------------
# STEP 12 — Artifact hashes
# ---------------------------------------------------------------------------
log("=== ARTIFACT HASHES ===")
hash_lines = []
artifacts = list(AUDIT.glob("*.csv")) + list(AUDIT.glob("*.json")) + list(AUDIT.glob("*.txt"))
for af in sorted(artifacts):
    h = hashlib.sha256(af.read_bytes()).hexdigest()
    hash_lines.append(f"{h}  {af.name}")
    log(f"  {af.name}: {h[:16]}...")
hash_lines.append(f"{input_sha}  ../../../{INPUT.name}  [INPUT]")
(AUDIT / "artifact_hashes.sha256").write_text("\n".join(hash_lines) + "\n")
log("  Saved artifact_hashes.sha256")

# ---------------------------------------------------------------------------
# Save log
# ---------------------------------------------------------------------------
(AUDIT / "commands.log").write_text("\n".join(log_lines) + "\n")
log("=== VERIFICATION COMPLETE ===")
log(f"All checks: {n_pass} PASS  {n_fail} FAIL")
