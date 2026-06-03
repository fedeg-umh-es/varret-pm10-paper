"""Generate unified multi-station figures for variance-retention diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLE_PATH = Path("outputs/tables/variance_retention_all_stations.csv")
OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
ROOT_DIR = Path(__file__).resolve().parents[1]
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
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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
    for ext in ("pdf", "png"):
        fig.savefig(ROOT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name} -> {OUTDIR}/ & {ROOT_DIR}/")


def _save_many(fig: plt.Figure, names: list[str]) -> None:
    for name in names:
        for ext in ("pdf", "png"):
            fig.savefig(OUTDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
            root_path = ROOT_DIR / f"{name}.{ext}"
            if root_path.exists():
                fig.savefig(root_path, dpi=300, bbox_inches="tight")
        print(f"[OK] {name} -> {OUTDIR}/ & {ROOT_DIR}/")
    plt.close(fig)


MODEL_COLORS = {
    "hgb_direct": "#d62728",
    "ridge_direct": "#1f77b4",
    "sarima": "#9467bd",
    "seasonal_naive": "#ff7f0e",
    "stl_ridge_direct": "#2ca02c",
}


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
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for model in models:
        d_model = df[df["model"] == model]
        profile = d_model.groupby("horizon")["skill"].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        color = MODEL_COLORS[model]
        ax.plot(
            profile.index,
            profile["median"],
            color=color,
            linewidth=2.4,
            marker=MODEL_MARKER[model],
            markersize=5,
            label=MODEL_LABEL[model],
        )
        ax.fill_between(profile.index, profile["q25"], profile["q75"], color=color, alpha=0.12, linewidth=0)

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xlim(1, 7)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("Persistence-relative RMSE skill")
    ax.set_title("Median RMSE skill by horizon")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, "figure3_skill_profiles")


def figure4_alpha_profiles(df: pd.DataFrame) -> None:
    models = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhspan(0.0, 0.5, color="#d9d9d9", alpha=0.28, label="Collapsed region")
    ax.axhspan(0.8, 1.2, color="#d7f0df", alpha=0.32, label="Near-retained band")

    for model in models:
        d_model = df[df["model"] == model]
        profile = d_model.groupby("horizon")["alpha"].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        color = MODEL_COLORS[model]
        ax.plot(
            profile.index,
            profile["median"],
            color=color,
            linewidth=2.4,
            marker=MODEL_MARKER[model],
            markersize=5,
            label=MODEL_LABEL[model],
        )
        ax.fill_between(profile.index, profile["q25"], profile["q75"], color=color, alpha=0.12, linewidth=0)

    ax.axhline(0.5, color="#333333", linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle=":")
    ax.set_xlim(1, 7)
    ax.set_ylim(0, max(1.9, float(df["alpha"].quantile(0.98)) * 1.05))
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("Variance-retention ratio alpha")
    ax.set_title("Median variance retention by horizon")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, "figure4_alpha_profiles")


def figure5_skill_alpha(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=300)

    ax.axvspan(0.8, 1.2, color="#d7f0df", alpha=0.35, label="Near-retained band", zorder=1)
    ax.axvspan(0.0, 0.5, color="#d9d9d9", alpha=0.26, label="Collapsed region", zorder=1)

    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=2)
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0, zorder=2)
    ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1.0, zorder=2)

    models = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]

    for model in models:
        mdf = df[df["model"] == model]
        x = float(mdf["alpha"].median())
        y = float(mdf["skill"].median())
        xerr = [[x - float(mdf["alpha"].quantile(0.25))], [float(mdf["alpha"].quantile(0.75)) - x]]
        yerr = [[y - float(mdf["skill"].quantile(0.25))], [float(mdf["skill"].quantile(0.75)) - y]]
        color = MODEL_COLORS[model]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=MODEL_MARKER[model],
            markersize=9,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            label=MODEL_LABEL[model],
            zorder=5,
        )
    ax.text(0.25, 0.25, "skillful but\ncollapsed", fontsize=9, color="#333333", va="top")
    ax.text(0.85, 0.27, "desired region", fontsize=9, color="#1f7a3a", va="top")
    ax.text(1.25, -1.28, "STL+Ridge:\nretained or inflated,\nbut low skill", fontsize=9, color="#333333", va="top")
    ax.set_xlabel(r"Variance retention $\alpha$")
    ax.set_ylabel("Persistence-relative RMSE skill")
    ax.set_title("Skill-retention diagnostic by model")
    ax.set_xlim(0.0, 1.95)
    ax.set_ylim(-2.2, 0.35)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.42)
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    _save_many(fig, ["figure5_scatter_skill_alpha", "figure_skill_alpha_five_models"])


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
    ax.set_title("PM10 station map: ML-only collapse rate", fontsize=11, fontweight="bold", pad=8)
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
    ax.set_title("PM10 station map: ML-only collapse rate")
    fig.tight_layout()
    return fig


def _plot_map_points(
    ax,
    points: pd.DataFrame,
    transform,
    size_legend_rates: list[float] | tuple[float, ...] = (0.4, 0.7, 1.0),
    legend_title: str = "Collapse rate",
) -> None:
    def marker_size(rate: float) -> float:
        return 75.0 if rate < 0.99 else 160.0

    for _, row in points.iterrows():
        color = CLASS_COLORS.get(row["station_class"], "#777777")
        size = marker_size(float(row["collapse_rate"]))
        kwargs = {"transform": transform} if transform is not None else {}
        ax.scatter(
            row["lon"],
            row["lat"],
            s=size,
            color=color,
            alpha=0.94,
            edgecolor="#111111",
            linewidth=0.85,
            zorder=5,
            **kwargs,
        )
        if row.get("collapsed_ml_cells", 14) == 13:
            ax.scatter(
                row["lon"],
                row["lat"],
                s=size + 70,
                facecolors="none",
                edgecolors="#111111",
                linewidths=1.4,
                zorder=6,
                **kwargs,
            )
            label = "Vall d'Hebron" if "Hebron" in row["station_name"] else str(row["station_name"])
            ax.annotate(
                label,
                xy=(row["lon"], row["lat"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
                color="#111111",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#111111", "lw": 0.4, "alpha": 0.88},
                zorder=7,
                **kwargs,
            )
    handles = [
        plt.Line2D(
            [],
            [],
            color=color,
            marker="o",
            markeredgecolor="#111111",
            linestyle="",
            markersize=7,
            label=label.title(),
        )
        for label, color in CLASS_COLORS.items()
    ]
    for rate in size_legend_rates:
        handles.append(
            plt.scatter(
                [],
                [],
                s=marker_size(float(rate)),
                color="white",
                alpha=1.0,
                edgecolor="#111111",
                linewidths=1.0,
                label=f"{rate * 100:.1f}%",
            )
        )
    legend = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)
    legend.set_title(legend_title, prop={"size": 8})


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
