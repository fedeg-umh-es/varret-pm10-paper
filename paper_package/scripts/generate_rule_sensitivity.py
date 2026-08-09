"""Render the existing canonical Rule-B threshold sensitivity audit."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "outputs/tables/master_diagnostic_table.csv"
AUDIT = ROOT / "audit/decision_change/rule_b_sensitivity.csv"
OUT_TABLE = ROOT / "paper_package/tables/rule_threshold_sensitivity.tex"
OUT_FIGURE = ROOT / "paper_package/figures/figure3_rule_threshold_sensitivity.pdf"
OUT_REPORT = ROOT / "paper_package/RULE_THRESHOLD_SENSITIVITY.md"
PRIMARY = (0.50, 0.20)
ALPHA_GRID = (0.40, 0.50, 0.60)
RECALL_GRID = (0.10, 0.20, 0.30)


def main() -> None:
    df = pd.read_csv(CANONICAL)
    sens = pd.read_csv(AUDIT)
    assert len(df) == 595
    assert "lightgbm_direct" not in set(df.model)
    required = {"alpha_thresh", "recall_thresh", "pass_b", "change_n", "change_pct", "primary"}
    assert required <= set(sens.columns)
    primary = sens[(sens.alpha_thresh == PRIMARY[0]) & (sens.recall_thresh == PRIMARY[1])].iloc[0]
    assert int(primary.pass_b) == 8 and int(primary.change_n) == 269
    sub = sens[sens.alpha_thresh.isin(ALPHA_GRID) & sens.recall_thresh.isin(RECALL_GRID)].copy()
    sub = sub.sort_values(["alpha_thresh", "recall_thresh"])
    assert len(sub) == 9
    assert float(sub.change_pct.min()) >= 90.0
    assert float(sub.change_pct.max()) <= 99.0

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\textbf{Supplementary Table S3.} Each cell reports retained Rule-B eligible cells and, in parentheses, the percentage of Rule-A cells that change status. The 595 canonical diagnostic rows are reused; no forecasts are recomputed.\par\smallskip",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"$\alpha$ threshold $\backslash$ recall threshold & 0.10 & 0.20 & 0.30 \\",
        r"\midrule",
    ]
    for at in ALPHA_GRID:
        vals = []
        for rt in RECALL_GRID:
            row = sub[(sub.alpha_thresh == at) & (sub.recall_thresh == rt)].iloc[0]
            vals.append(f"{int(row.pass_b)} ({float(row.change_pct):.1f}\%)")
        lines.append(f"{at:.2f} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    OUT_TABLE.write_text("\n".join(lines), encoding="utf-8")

    pivot = sub.pivot(index="alpha_thresh", columns="recall_thresh", values="change_pct").reindex(index=ALPHA_GRID, columns=RECALL_GRID)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=90, vmax=100, aspect="auto")
    ax.set_xticks(range(3), [f"{x:.2f}" for x in RECALL_GRID])
    ax.set_yticks(range(3), [f"{x:.2f}" for x in ALPHA_GRID])
    ax.set_xlabel("P75 recall threshold")
    ax.set_ylabel(r"$\alpha$ threshold")
    ax.set_title("Rule-B eligibility changes across threshold neighbourhood")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((0.5, 0.5), 1.0, 1.0, fill=False,
                           edgecolor="#333333", linewidth=1.4, zorder=3))
    fig.colorbar(im, ax=ax, label="Rule-A cells changing (%)")
    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Rule-B threshold sensitivity

Source: existing `audit/decision_change/rule_b_sensitivity.csv`, verified against the canonical 595-row table.

Primary thresholds: alpha = 0.50 and P75 recall = 0.20. Primary result: Rule A = 277, Rule B = 8, changes = 269, change proportion = 97.1%.

The displayed neighbourhood uses alpha thresholds {{0.40, 0.50, 0.60}} and P75-recall thresholds {{0.10, 0.20, 0.30}}. Across these nine combinations, the proportion of Rule-A cells changing ranges from {sub.change_pct.min():.1f}% to {sub.change_pct.max():.1f}%; retained Rule-B cells range from {sub.pass_b.max()} to {sub.pass_b.min()}. The qualitative conclusion that fidelity materially changes eligibility persists, so the sensitivity verdict is **ROBUST** within this deterministic neighbourhood.

This is a post-evaluation sensitivity analysis of existing rows, not a forecasting experiment, threshold optimisation, or claim of universal threshold validity.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Sensitivity PASS: {sub.change_pct.min():.1f}%--{sub.change_pct.max():.1f}%")


if __name__ == "__main__":
    main()
