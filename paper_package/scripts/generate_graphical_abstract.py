"""Create a minimal conceptual EMS graphical abstract."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_package/figures/graphical_abstract_eligibility.pdf"

fig, ax = plt.subplots(figsize=(8, 2.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
boxes = [
    (0.4, "Error eligibility", "Rule A: 277 cells", "#DCEAF7"),
    (3.55, "Dynamic-fidelity audit", r"alpha $\geq$ 0.50; P75 recall $\geq$ 0.20", "#E7F3E8"),
    (6.9, "Eligibility consequence", "Rule B: 8 cells; 269 changes (97.1%)", "#FBE6D5"),
]
for x, title, detail, color in boxes:
    patch = FancyBboxPatch((x, 0.75), 2.55, 1.45, boxstyle="round,pad=0.05,rounding_size=0.08",
                           facecolor=color, edgecolor="#444444", linewidth=0.9)
    ax.add_patch(patch)
    ax.text(x + 1.275, 1.72, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + 1.275, 1.22, detail, ha="center", va="center", fontsize=9, wrap=True)
for x in (3.02, 6.37):
    ax.annotate("", xy=(x + 0.35, 1.48), xytext=(x, 1.48), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#444444"})
ax.text(5, 2.65, "Daily PM$_{10}$ multi-station decision audit", ha="center", fontsize=12, weight="bold")
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(OUT)
