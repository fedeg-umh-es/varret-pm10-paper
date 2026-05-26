"""Generate unified multi-station figures for variance-retention diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE_PATH = Path("outputs/tables/variance_retention_all_stations.csv")
OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
ML_STATION_RATE_PATH = Path("outputs/tables/station_ml_only_collapse_rates.csv")
ML_MAP_CAPTION_PATH = Path("outputs/audit/station_map_ml_only_collapse_rate_caption.md")
ML_MODELS = ("hgb_direct", "ridge_direct")

CLASS_COLORS = {
    "industrial": "#c23b35",
    "urban": "#2878b5",
    "rural": "#2f8f4e",
}
MODEL_MARKER = {
    "hgb_direct": "o",
    "ridge_direct": "s",
    "sarima": "p",
    "seasonal_naive": "^",
    "stl_ridge_direct": "D",
}
MODEL_LABEL = {
    "hgb_direct": "HGB direct",
    "ridge_direct": "Ridge direct",
    "sarima": "SARIMA",
    "seasonal_naive": "Seasonal naive",
    "stl_ridge_direct": "STL+Ridge",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load() -> pd.DataFrame:
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing unified table: {TABLE_PATH}")
    df = pd.read_csv(TABLE_PATH)
    required = {
        "station_id",
        "station_name",
        "station_type",
        "station_class",
        "model",
        "horizon",
        "skill",
        "alpha",
        "collapse_flag",
        "lat",
        "lon",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unified table missing columns: {sorted(missing)}")
    return df


def _station_order(df: pd.DataFrame) -> list[str]:
    rates = df.groupby("station_id")["collapse_flag"].mean().sort_values(ascending=False)
    return rates.index.tolist()


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    root_dir = Path("/Users/federicogarciacrespi/Public/varret-pm10-paper")
    for ext in ("pdf", "png"):
        fig.savefig(root_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name} -> {OUTDIR}/ & {root_dir}/")


def _ml_station_collapse_rates(df: pd.DataFrame) -> pd.DataFrame:
    ml = df[df["model"].isin(ML_MODELS)].copy()
    ml["collapsed_ml_cell"] = ml["alpha"] < 0.5
    rates = (
        ml.groupby(
            ["station_id", "station_name", "station_type", "station_class", "lat", "lon"],
            as_index=False,
        )
        .agg(
            collapsed_ml_cells=("collapsed_ml_cell", "sum"),
            total_ml_cells=("collapsed_ml_cell", "size"),
        )
        .sort_values(["collapsed_ml_cells", "station_name"], ascending=[True, True])
    )
    rates["collapse_rate"] = rates["collapsed_ml_cells"] / rates["total_ml_cells"]
    rates["collapse_rate_pct"] = (rates["collapse_rate"] * 100).round(1)
    if not rates["total_ml_cells"].eq(14).all():
        raise ValueError("ML-only station collapse rates must use 14 cells per station.")
    exceptions = set(rates.loc[rates["collapsed_ml_cells"].eq(13), "station_name"])
    expected_exceptions = {"Huesca", "Barcelona Vall d'Hebron"}
    if exceptions != expected_exceptions or not rates["collapsed_ml_cells"].isin({13, 14}).all():
        raise ValueError(
            "Unexpected ML-only collapse-rate pattern: expected Huesca and "
            "Barcelona Vall d'Hebron at 13/14 and all other stations at 14/14."
        )
    return rates


def figure3_skill_profiles(df: pd.DataFrame) -> None:
    models = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 10.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes_flat[i]
        d_model = df[df["model"] == model]
        for station_id, grp in d_model.groupby("station_id"):
            grp = grp.sort_values("horizon")
            station_class = grp["station_class"].iloc[0]
            ax.plot(
                grp["horizon"],
                grp["skill"],
                color=CLASS_COLORS.get(station_class, "#777777"),
                marker=MODEL_MARKER[model],
                linewidth=0.9,
                markersize=2.8,
                alpha=0.55,
            )
        mean_profile = d_model.groupby("horizon")["skill"].mean()
        ax.plot(mean_profile.index, mean_profile.values, color="black", linewidth=2.0, marker="D", markersize=3.2)
        ax.axhline(0, color="#666666", linewidth=0.7, linestyle=":")
        ax.set_title(MODEL_LABEL[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Forecast horizon h", fontsize=8.5)
        ax.set_xticks(range(1, 8))
        ax.grid(True, alpha=0.18)
        
    for row in range(3):
        axes[row, 0].set_ylabel("Persistence-relative skill", fontsize=8.5)
        
    legend_ax = axes_flat[5]
    legend_ax.axis("off")
    
    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=label.title())
        for label, color in CLASS_COLORS.items()
    ]
    handles.append(plt.Line2D([], [], color="black", marker="D", linewidth=2, label="Station mean"))
    legend_ax.legend(handles=handles, loc="center", frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=9)
    
    fig.suptitle("Skill profiles by horizon across 17 PM10 stations", y=0.98, fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "figure3_skill_profiles")


def figure4_alpha_profiles(df: pd.DataFrame) -> None:
    models = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 10.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes_flat[i]
        d_model = df[df["model"] == model]
        ax.fill_between([1, 7], [0, 0], [0.5, 0.5], color="#999999", alpha=0.10)
        for station_id, grp in d_model.groupby("station_id"):
            grp = grp.sort_values("horizon")
            station_class = grp["station_class"].iloc[0]
            ax.plot(
                grp["horizon"],
                grp["alpha"],
                color=CLASS_COLORS.get(station_class, "#777777"),
                marker=MODEL_MARKER[model],
                linewidth=0.9,
                markersize=2.8,
                alpha=0.55,
            )
        mean_profile = d_model.groupby("horizon")["alpha"].mean()
        ax.plot(mean_profile.index, mean_profile.values, color="black", linewidth=2.0, marker="D", markersize=3.2)
        ax.axhline(0.5, color="#333333", linewidth=0.8, linestyle="--")
        ax.axhline(1.0, color="#777777", linewidth=0.7, linestyle=":")
        ax.set_title(MODEL_LABEL[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Forecast horizon h", fontsize=8.5)
        ax.set_xticks(range(1, 8))
        ax.grid(True, alpha=0.18)
        
    for row in range(3):
        axes[row, 0].set_ylabel("Variance-retention ratio alpha", fontsize=8.5)
        
    legend_ax = axes_flat[5]
    legend_ax.axis("off")
    
    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=label.title())
        for label, color in CLASS_COLORS.items()
    ]
    handles.append(plt.Line2D([], [], color="black", marker="D", linewidth=2, label="Station mean"))
    legend_ax.legend(handles=handles, loc="center", frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=9)
    
    max_alpha = max(2.2, float(df["alpha"].quantile(0.98)) * 1.08)
    axes[0, 0].set_ylim(0, max_alpha)
    
    fig.suptitle("Variance-retention profiles by horizon across 17 PM10 stations", y=0.98, fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "figure4_alpha_profiles")


def figure5_skill_alpha(df: pd.DataFrame) -> None:
    # Premium Horizon-Trajectory plot of Skill vs. Alpha for 5 models
    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=300)

    # Shaded band for near-ideal variance band (0.8 <= alpha <= 1.2)
    ax.axvspan(0.8, 1.2, color="#E8F8F5", alpha=0.5, label="Near-Ideal Variance Band [0.8, 1.2]", zorder=1)

    # Reference lines
    ax.axhline(0.0, color="#5D6D7E", linestyle="--", linewidth=1.0, label="Persistence Reference (Skill = 0.0)", zorder=2)
    ax.axvline(0.5, color="#7F8C8D", linestyle=":", linewidth=1.0, label="Collapse Boundary (alpha = 0.5)", zorder=2)
    ax.axvline(1.0, color="#27AE60", linestyle="-.", linewidth=1.0, label="Perfect Variance (alpha = 1.0)", zorder=2)

    # Plot Faint Background Scatter Cloud (All 595 points)
    model_colors = {
        "hgb_direct": "#008080",       # Teal
        "ridge_direct": "#2E4053",     # Slate Blue
        "sarima": "#8E44AD",           # Purple
        "seasonal_naive": "#F39C12",   # Gold/Orange
        "stl_ridge_direct": "#E74C3C"  # Crimson/Red
    }
    model_families = {
        "hgb_direct": "Direct ML (HGB)",
        "ridge_direct": "Direct ML (Ridge)",
        "sarima": "Statistical (SARIMA)",
        "seasonal_naive": "Naive (Seasonal)",
        "stl_ridge_direct": "Decomposition (STL+Ridge)"
    }
    horizon_markers = {1: "o", 2: "^", 3: "s", 4: "p", 5: "h", 6: "8", 7: "D"}
    models = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]

    for model in models:
        mdf = df[df["model"] == model]
        ax.scatter(
            mdf["alpha"],
            mdf["skill"],
            color=model_colors[model],
            alpha=0.08,
            s=20,
            edgecolors="none",
            zorder=3
        )

    # Compute and Plot Median Trajectories (Connected lines h=1 to h=7)
    for model in models:
        mdf = df[df["model"] == model]
        traj = mdf.groupby("horizon")[["alpha", "skill"]].median().sort_index().reset_index()
        
        ax.plot(
            traj["alpha"],
            traj["skill"],
            color=model_colors[model],
            linewidth=3.0,
            linestyle="-",
            zorder=4,
            alpha=0.9
        )
        
        for _, row in traj.iterrows():
            h = int(row["horizon"])
            ax.scatter(
                row["alpha"],
                row["skill"],
                color=model_colors[model],
                marker=horizon_markers[h],
                s=75,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                alpha=1.0
            )

        h1_row = traj[traj["horizon"] == 1].iloc[0]
        h7_row = traj[traj["horizon"] == 7].iloc[0]
        
        offset_x = 0.02
        offset_y = 0.02
        if model == "stl_ridge_direct":
            offset_y = -0.04
        elif model == "seasonal_naive":
            offset_x = 0.03
            offset_y = -0.02

        ax.text(
            h1_row["alpha"] + offset_x, h1_row["skill"] + offset_y,
            r"$h=1$", fontsize=8, fontweight="semibold", color=model_colors[model],
            zorder=6, va="center"
        )
        ax.text(
            h7_row["alpha"] + offset_x, h7_row["skill"] + offset_y,
            r"$h=7$", fontsize=8, fontweight="semibold", color=model_colors[model],
            zorder=6, va="center"
        )

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.90, edgecolor="#D5D8DC", linewidth=0.8)
    ax.text(0.04, 0.28, "I: High Skill / Collapsed\n(HGB, Ridge, SARIMA)", fontsize=9.5, fontweight="semibold", color="#2C3E50", bbox=bbox_props, va="top", ha="left")
    ax.text(0.80, 0.28, "II: High Skill / Retained", fontsize=9.5, fontweight="semibold", color="#27AE60", bbox=bbox_props, va="top", ha="left")
    ax.text(0.04, -2.0, "III: Low Skill / Collapsed", fontsize=9.5, fontweight="semibold", color="#7F8C8D", bbox=bbox_props, va="bottom", ha="left")
    ax.text(1.20, -2.0, "IV: Low Skill / Retained\n(STL+Ridge, Seasonal)", fontsize=9.5, fontweight="semibold", color="#C0392B", bbox=bbox_props, va="bottom", ha="left")

    ax.set_xlabel(r"Variance Retention Coefficient ($\alpha = s_{\hat{y}} / s_{y}$)", fontsize=12, fontweight="semibold", labelpad=8)
    ax.set_ylabel(r"Forecasting Skill (Persistence-Relative $1 - \text{MSE}/\text{MSE}_{\text{pers}}$)", fontsize=12, fontweight="semibold", labelpad=8)
    ax.set_title("The Predictability-Variance Frontier in PM10 Forecasting", fontsize=14, fontweight="bold", pad=15)
    
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(-2.2, 0.35)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#BDC3C7", alpha=0.5)

    legend_handles = []
    for model in models:
        legend_handles.append(
            plt.Line2D(
                [], [],
                color=model_colors[model],
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=model_families[model]
            )
        )
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, color="#E8F8F5", alpha=0.6, label="Near-Ideal Band [0.8, 1.2]")
    )
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=9.5, framealpha=0.95)
    fig.tight_layout()
    _save(fig, "figure5_scatter_skill_alpha")


def figure6_collapse_rates(df: pd.DataFrame) -> None:
    station_rates = (
        df.groupby(["station_id", "station_name", "station_class"])["collapse_flag"]
        .mean()
        .reset_index()
        .sort_values("collapse_flag", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    colors = [CLASS_COLORS.get(v, "#777777") for v in station_rates["station_class"]]
    ax.barh(station_rates["station_name"], station_rates["collapse_flag"] * 100, color=colors, alpha=0.85)
    ax.set_xlabel("Collapse rate (% station x model x horizon cells with alpha < 0.5)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.18)
    handles = [
        plt.Line2D([], [], color=color, linewidth=5, label=label.title())
        for label, color in CLASS_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout()
    _save(fig, "figure6_station_collapse_rates")


def _map_with_cartopy(points: pd.DataFrame) -> plt.Figure:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
    
    # Premium High-Contrast Colors: warm-white land, soft-blue ocean, distinct dark-slate borders/coastlines
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#FCFAF7")
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#EAF2F8")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#2E4053", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#4D5656", linewidth=0.8, linestyle="-")
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#AED6F1", linewidth=0.3, alpha=0.45)
    
    # Regional autonomous community boundaries for premium local context inside Spain
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#BDC3C7", linewidth=0.5, linestyle=":")
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    _plot_map_points(ax, points, transform=ccrs.PlateCarree())
    ax.set_title("PM10 station map: collapse rate encoded by marker size", fontsize=10, fontweight="bold", pad=8)
    return fig


def _ml_map_with_cartopy(points: pd.DataFrame) -> plt.Figure:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
    
    # Premium High-Contrast Colors: warm-white land, soft-blue ocean, distinct dark-slate borders/coastlines
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#FCFAF7")
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#EAF2F8")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#2E4053", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#4D5656", linewidth=0.8, linestyle="-")
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#AED6F1", linewidth=0.3, alpha=0.45)
    
    # Regional autonomous community boundaries for premium local context inside Spain
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#BDC3C7", linewidth=0.5, linestyle=":")
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    _plot_map_points(
        ax,
        points,
        transform=ccrs.PlateCarree(),
        size_legend_rates=sorted(points["collapse_rate"].unique()),
        legend_title="ML collapse rate",
    )
    ax.set_title("PM10 station map: ML-only collapse rate encoded by marker size", fontsize=10, fontweight="bold", pad=8)
    return fig


def _map_simple(points: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(-10, 5)
    ax.set_ylim(35, 45)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    _plot_map_points(ax, points, transform=None)
    ax.set_title("PM10 station map: collapse rate encoded by marker size")
    fig.tight_layout()
    return fig


def _ml_map_simple(points: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(-10, 5)
    ax.set_ylim(35, 45)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    _plot_map_points(
        ax,
        points,
        transform=None,
        size_legend_rates=sorted(points["collapse_rate"].unique()),
        legend_title="ML collapse rate",
    )
    ax.set_title("PM10 station map: ML-only collapse rate encoded by marker size")
    fig.tight_layout()
    return fig


def _plot_map_points(
    ax,
    points: pd.DataFrame,
    transform,
    size_legend_rates: list[float] | tuple[float, ...] = (0.4, 0.7, 1.0),
    legend_title: str = "Collapse rate",
) -> None:
    for _, row in points.iterrows():
        color = CLASS_COLORS.get(row["station_class"], "#777777")
        # Larger base size and scaling for much better visual clarity on markers
        size = 40 + 220 * row["collapse_rate"]
        kwargs = {"transform": transform} if transform is not None else {}
        ax.scatter(
            row["lon"],
            row["lat"],
            s=size,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            **kwargs,
        )
    handles = [
        plt.Line2D([], [], color=color, marker="o", linestyle="", markersize=7, label=label.title())
        for label, color in CLASS_COLORS.items()
    ]
    for rate in size_legend_rates:
        # High contrast slate markers in size legend for flawless legibility
        handles.append(
            plt.scatter(
                [],
                [],
                s=40 + 220 * rate,
                color="#2C3E50",
                alpha=0.85,
                edgecolor="white",
                linewidths=0.8,
                label=f"{rate * 100:.1f}%",
            )
        )
    legend = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=7)
    legend.set_title(legend_title, prop={"size": 7})


def figure7_station_map(df: pd.DataFrame) -> None:
    points = (
        df.groupby(["station_id", "station_name", "station_class", "lat", "lon"], as_index=False)["collapse_flag"]
        .mean()
        .rename(columns={"collapse_flag": "collapse_rate"})
    )
    try:
        fig = _map_with_cartopy(points)
        backend = "cartopy"
    except Exception as exc:
        print(f"[WARN] cartopy map failed, using matplotlib fallback: {exc}")
        fig = _map_simple(points)
        backend = "matplotlib fallback"
    _save(fig, "figure7_station_map_spain")
    print(f"[OK] figure7 backend: {backend}")


def station_map_ml_only_collapse_rate(df: pd.DataFrame) -> None:
    points = _ml_station_collapse_rates(df)
    ML_STATION_RATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    points[
        [
            "station_name",
            "station_type",
            "collapsed_ml_cells",
            "total_ml_cells",
            "collapse_rate_pct",
        ]
    ].rename(columns={"station_name": "station"}).to_csv(ML_STATION_RATE_PATH, index=False)

    caption = (
        "Geographic distribution of the 17 MITECO stations. Marker colour denotes station type. "
        "Marker size denotes the ML-only collapse rate, computed over 14 model-horizon cells per "
        "station (2 direct ML models x 7 horizons). Fifteen stations show 14/14 collapsed ML cells; "
        "Huesca and Barcelona Vall d’Hebron show 13/14 due to one non-collapsed h=1 cell each."
    )
    ML_MAP_CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ML_MAP_CAPTION_PATH.write_text(caption + "\n", encoding="utf-8")

    try:
        fig = _ml_map_with_cartopy(points)
        backend = "cartopy"
    except Exception as exc:
        print(f"[WARN] cartopy ML map failed, using matplotlib fallback: {exc}")
        fig = _ml_map_simple(points)
        backend = "matplotlib fallback"
    _save(fig, "station_map_ml_only_collapse_rate")
    print(f"[OK] ML-only station map backend: {backend}")


def main() -> None:
    df = _load()
    figure3_skill_profiles(df)
    figure4_alpha_profiles(df)
    figure5_skill_alpha(df)
    figure6_collapse_rates(df)
    figure7_station_map(df)
    station_map_ml_only_collapse_rate(df)
    print(f"All figures saved to {OUTDIR.resolve()}/")


if __name__ == "__main__":
    main()
