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

# ── Figure 5 — Scatter skill–alpha (all 42 points) ────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.fill_between([0, 0.32], [0, 0], [0.5, 0.5],
                alpha=0.07, color="gray", label="Collapse region ($\\alpha < 0.5$)")
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(0,   color="gray", linewidth=0.8, linestyle=":")

for name, meta in STATIONS.items():
    for model in ("hgb_direct", "ridge_direct"):
        d = dfs[name][dfs[name]["model"] == model].sort_values("horizon")
        ax.scatter(d["skill"], d["alpha"],
                   c=meta["color"], marker=MODEL_MARKER[model],
                   s=42, alpha=0.88, zorder=3,
                   label=f"{name} / {model}")
        for _, row in d.iterrows():
            ax.annotate(str(int(row["horizon"])),
                        (row["skill"], row["alpha"]),
                        textcoords="offset points", xytext=(3, 2),
                        fontsize=5.5, color=meta["color"], alpha=0.65)

ax.set_xlabel("Persistence-relative skill")
ax.set_ylabel("Variance-retention ratio $\\alpha$")
ax.set_xlim(-0.01, 0.31)
ax.set_ylim(0, 0.58)
ax.legend(fontsize=6.5, loc="upper left", ncol=1,
          framealpha=0.9, edgecolor="lightgray")
ax.set_title("Skill–variance retention: all 42 model/station/horizon cells", fontsize=9)
fig.tight_layout()
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
