"""Generate figures 3-6 for the varret-pm10-paper (three-station E1-RR results)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Station metadata ───────────────────────────────────────────────────────────
STATIONS = {
    "Elx-Agroalimentari": {
        "file":   "outputs/tables/variance_retention_summary.csv",
        "color":  "#d62728",
        "lat":    38.24222,
        "lon":    -0.68278,
        "alt":    44,
        "dem":    "ES1624A",
        "type":   "Suburban / industrial",
    },
    "València-Vivers": {
        "file":   "outputs/tables/variance_retention_valencia_vivers.csv",
        "color":  "#1f77b4",
        "lat":    39.47806,
        "lon":    -0.36833,
        "alt":    11,
        "dem":    "ES1619A",
        "type":   "Urban / residential",
    },
    "Zarra (EMEP)": {
        "file":   "outputs/tables/variance_retention_zarra_emep.csv",
        "color":  "#2ca02c",
        "lat":    39.08278,
        "lon":    -1.10083,
        "alt":    855,
        "dem":    "ES0012R",
        "type":   "Rural remote / EMEP",
    },
}
MODEL_MARKER = {"hgb_direct": "o", "ridge_direct": "s"}
MODEL_LS     = {"hgb_direct": "-", "ridge_direct": "--"}

dfs = {name: pd.read_csv(meta["file"]) for name, meta in STATIONS.items()}

# ── Figure 3 — Skill profiles by horizon ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, model in zip(axes, ["hgb_direct", "ridge_direct"]):
    for name, meta in STATIONS.items():
        d = dfs[name][dfs[name]["model"] == model].sort_values("horizon")
        ax.plot(d["horizon"], d["skill"],
                color=meta["color"], marker=MODEL_MARKER[model],
                linestyle=MODEL_LS[model], linewidth=1.8,
                markersize=5, label=name)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_title(f"model: {model}", fontsize=9)
    ax.set_xlabel("Forecast horizon $h$")
    ax.set_xticks(range(1, 8))
    ax.set_ylim(-0.02, 0.33)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("Persistence-relative skill")
axes[0].legend(loc="lower right")
fig.suptitle("Skill profiles by horizon — three background stations", fontsize=9, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"figure3_skill_profiles.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure3_skill_profiles  →  {OUTDIR}/")

# ── Figure 4 — Alpha profiles by horizon ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, model in zip(axes, ["hgb_direct", "ridge_direct"]):
    ax.fill_between(range(1, 8), 0, 0.5, alpha=0.07, color="gray", zorder=0)
    for name, meta in STATIONS.items():
        d = dfs[name][dfs[name]["model"] == model].sort_values("horizon")
        ax.plot(d["horizon"], d["alpha"],
                color=meta["color"], marker=MODEL_MARKER[model],
                linestyle=MODEL_LS[model], linewidth=1.8,
                markersize=5, label=name)
    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--",
               label="Collapse threshold ($\\alpha=0.5$)")
    ax.set_title(f"model: {model}", fontsize=9)
    ax.set_xlabel("Forecast horizon $h$")
    ax.set_xticks(range(1, 8))
    ax.set_ylim(0, 0.58)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("Variance-retention ratio $\\alpha$")
axes[0].legend(loc="upper right")
fig.suptitle("Variance-retention profiles by horizon — three background stations", fontsize=9, y=1.01)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"figure4_alpha_profiles.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure4_alpha_profiles  →  {OUTDIR}/")

# ── Figure 5 — Skill–alpha trajectories (2 panels, one per model) ─────────────
# Layout: 2 data panels + 1 narrow legend panel on the right.
# No inline text annotations — all labelling via the shared legend panel.
fig = plt.figure(figsize=(11, 4.6))
gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.32], wspace=0.08)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharey=ax0, sharex=ax0)
ax_leg = fig.add_subplot(gs[2])
ax_leg.axis("off")

# h=1 annotation offsets per station (manually tuned to avoid overlap)
H1_OFFSETS = {
    "Elx-Agroalimentari": (-28,  4),
    "València-Vivers":    (-28, -10),
    "Zarra (EMEP)":       (-28,  4),
}

for ax, model in zip([ax0, ax1], ["hgb_direct", "ridge_direct"]):
    ax.fill_between([0, 0.32], [0, 0], [0.5, 0.5],
                    alpha=0.07, color="gray", zorder=0)
    ax.axhline(0.5, color="#555", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(0.285, 0.505, "$\\alpha = 0.5$", fontsize=7,
            color="#555", va="bottom", ha="right")
    ax.text(0.285, 0.015, "COLLAPSE ZONE", fontsize=6.5,
            color="#aaa", va="bottom", ha="right", style="italic")

    for name, meta in STATIONS.items():
        d   = dfs[name][dfs[name]["model"] == model].sort_values("horizon")
        xs  = d["skill"].values
        ys  = d["alpha"].values

        # connecting line
        ax.plot(xs, ys, color=meta["color"], linewidth=1.5,
                alpha=0.55, zorder=2)

        # bubbles — size encodes horizon
        sizes = [22 + 13 * int(h) for h in d["horizon"]]
        ax.scatter(xs, ys, c=meta["color"], s=sizes, alpha=0.9,
                   zorder=3, edgecolors="white", linewidths=0.5)

        # direction arrow on last segment only
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=meta["color"],
                                   lw=1.2, mutation_scale=9), zorder=4)

        # h=1 label (one per trajectory, offset tuned per station)
        dx, dy = H1_OFFSETS[name]
        ax.annotate("$h\\!=\\!1$", (xs[0], ys[0]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6, color=meta["color"],
                    arrowprops=dict(arrowstyle="-", color=meta["color"],
                                   lw=0.5, shrinkA=0, shrinkB=3))

    ax.set_xlabel("Persistence-relative skill", fontsize=9)
    ax.set_title(f"model: {model}", fontsize=9, pad=6)
    ax.set_xlim(0.02, 0.30)
    ax.set_ylim(0.0,  0.55)
    ax.grid(True, alpha=0.15)

ax0.set_ylabel("Variance-retention ratio $\\alpha$", fontsize=9)
plt.setp(ax1.get_yticklabels(), visible=False)

# ── Legend panel ─────────────────────────────────────────────────────────────
# Station colours
for name, meta in STATIONS.items():
    ax_leg.plot([], [], color=meta["color"], linewidth=2.5,
                label=name)
# Horizon size scale
for h_val, lab in [(1, "$h = 1$"), (4, "$h = 4$"), (7, "$h = 7$")]:
    ax_leg.scatter([], [], c="#888", s=22 + 13 * h_val,
                   edgecolors="white", linewidths=0.5, label=lab)
# Collapse threshold
ax_leg.plot([], [], color="#555", linewidth=0.9, linestyle="--",
            label="Collapse\nthreshold")

leg = ax_leg.legend(loc="center left", fontsize=7.5, frameon=True,
                    framealpha=0.95, edgecolor="#ccc",
                    handlelength=1.6, handletextpad=0.5,
                    borderpad=0.7, labelspacing=0.55)
leg.set_title("Station / Horizon", prop={"size": 7.5})

fig.suptitle(
    "Skill–$\\alpha$ trajectories ($h = 1 \\to 7$) — all trajectories remain in the collapse zone",
    fontsize=9, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"figure5_scatter_skill_alpha.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure5_scatter_skill_alpha  →  {OUTDIR}/")

# ── Figure 6 — Station map ────────────────────────────────────────────────────
def _make_map_cartopy(ax_in=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig_m = plt.figure(figsize=(6, 5.5))
    ax_m  = fig_m.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_m.set_extent([-1.8, 0.3, 37.8, 40.2], crs=ccrs.PlateCarree())
    ax_m.add_feature(cfeature.LAND.with_scale("10m"),  facecolor="#f4f2ee")
    ax_m.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#cce5f0")
    ax_m.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6)
    ax_m.add_feature(cfeature.BORDERS.with_scale("10m"),   linewidth=0.4, linestyle=":")
    label_offsets = {
        "Elx-Agroalimentari": ( 0.06, -0.14),
        "València-Vivers":    ( 0.06,  0.06),
        "Zarra (EMEP)":       (-0.55,  0.06),
    }
    for name, meta in STATIONS.items():
        ax_m.plot(meta["lon"], meta["lat"], "o",
                  color=meta["color"], markersize=10,
                  transform=ccrs.PlateCarree(), zorder=5)
        dx, dy = label_offsets[name]
        ax_m.text(meta["lon"] + dx, meta["lat"] + dy,
                  f"{name}\n{meta['dem']}  {meta['alt']} m a.s.l.",
                  fontsize=7, transform=ccrs.PlateCarree(),
                  color=meta["color"], fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
    gl = ax_m.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax_m.set_title(
        "Monitoring station locations — Valencian Community (Spain)", fontsize=9)
    return fig_m

def _make_map_simple():
    fig_m, ax_m = plt.subplots(figsize=(6, 5.5))
    for name, meta in STATIONS.items():
        ax_m.plot(meta["lon"], meta["lat"], "o",
                  color=meta["color"], markersize=11, zorder=5)
        ax_m.annotate(
            f"{name}\n{meta['dem']}  {meta['alt']} m",
            (meta["lon"], meta["lat"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7.5, color=meta["color"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))
    ax_m.set_xlim(-1.55, 0.15)
    ax_m.set_ylim(37.9, 40.0)
    ax_m.set_xlabel("Longitude (°E)")
    ax_m.set_ylabel("Latitude (°N)")
    ax_m.grid(True, alpha=0.3)
    ax_m.set_title(
        "Monitoring station locations — Valencian Community (Spain)", fontsize=9)
    fig_m.tight_layout()
    return fig_m

try:
    fig6 = _make_map_cartopy()
    backend = "cartopy"
except Exception:
    fig6 = _make_map_simple()
    backend = "matplotlib fallback"

for ext in ("pdf", "png"):
    fig6.savefig(OUTDIR / f"figure6_station_map.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] figure6_station_map  →  {OUTDIR}/  [{backend}]")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nAll 4 figures saved to:  {OUTDIR.resolve()}/")
print("  figure3_skill_profiles.[pdf|png]")
print("  figure4_alpha_profiles.[pdf|png]")
print("  figure5_scatter_skill_alpha.[pdf|png]")
print("  figure6_station_map.[pdf|png]")
