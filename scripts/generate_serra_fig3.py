#!/usr/bin/env python3
"""Render the accepted SERRA Figure 3 from the frozen cell-level CSVs.

This script is deliberately a presentation-only transformation.  It reads the
two final inferential ledgers and never recomputes forecasts, losses, test
statistics, or multiplicity adjustments.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]


def repo_file(filename: str) -> Path:
    """Resolve only the accepted final CSV layout in either repo mirror."""
    candidates = [ROOT / filename, ROOT / "outputs" / "tables" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"final SERRA cell-level CSV not found: {filename}")


Q0_PATH = repo_file("serra_dm_q0_cell_level.csv")
AUTO_NW_PATH = repo_file("serra_dm_auto_nw_cell_level.csv")
OUTPUT_PATH = (
    ROOT / "outputs" / "figures" / "serra_fig3_dm_inferential_landscape.png"
    if (ROOT / "outputs" / "figures").is_dir()
    else ROOT / "serra_fig3_dm_inferential_landscape.png"
)

SITE_ORDER = [
    "Madrid Casa de Campo",
    "Birr (Co. Offaly)",
    "Dublin Airport",
    "Dundalk (Co. Louth)",
    "Pearse St. Dublin",
    "Ringsend Dublin",
    "Edenderry (Co. Offaly)",
    "Henry St. Limerick",
    "Portlaoise (Co. Laois)",
]
HORIZONS = [1, 6, 12, 24]


def read_ledger(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(site, horizon) for site in SITE_ORDER for horizon in HORIZONS}
    observed = {(row["site"], int(row["horizon"])) for row in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"unexpected cell coverage in {path}: missing={missing}, extra={extra}")
    return {(row["site"], int(row["horizon"])): row for row in rows}


def main() -> None:
    ledgers = {
        "Primary: $q_{\\mathrm{overlap}}=0$": read_ledger(Q0_PATH),
        "Fixed automatic Newey--West": read_ledger(AUTO_NW_PATH),
    }
    all_values = [
        float(row["dm_hln_stat"])
        for ledger in ledgers.values()
        for row in ledger.values()
    ]
    vmax = max(1.0, math.ceil(max(abs(value) for value in all_values) * 10) / 10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
    cmap = "coolwarm"
    images = []
    short_titles = [
        "A  Primary: $q_{\\mathrm{overlap}}=0$",
        "B  Fixed automatic Newey--West",
    ]
    short_sites = [
        "Madrid Casa de Campo",
        "Birr (Co. Offaly)",
        "Dublin Airport",
        "Dundalk (Co. Louth)",
        "Pearse St. Dublin",
        "Ringsend Dublin",
        "Edenderry (Co. Offaly)",
        "Henry St. Limerick",
        "Portlaoise (Co. Laois)",
    ]

    for panel_index, (ax, (title, ledger)) in enumerate(zip(axes, ledgers.items())):
        values = np.array(
            [
                [float(ledger[(site, horizon)]["dm_hln_stat"]) for horizon in HORIZONS]
                for site in SITE_ORDER
            ]
        )
        image = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        images.append(image)
        for row_index, site in enumerate(SITE_ORDER):
            for col_index, horizon in enumerate(HORIZONS):
                cell = ledger[(site, horizon)]
                ax.text(
                    col_index,
                    row_index,
                    f"{float(cell['dm_hln_stat']):.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
                if cell["p_raw"].lower() == "true" or float(cell["p_raw"]) < 0.05:
                    ax.scatter(
                        col_index,
                        row_index,
                        marker="o",
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.5,
                        s=210,
                        zorder=3,
                    )
                if cell["bh_station_reject"].lower() == "true":
                    ax.scatter(
                        col_index,
                        row_index,
                        marker="*",
                        facecolors="#ffd92f",
                        edgecolors="black",
                        linewidths=1.0,
                        s=190,
                        zorder=4,
                    )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
        ax.set_xlabel(r"Physical horizon $h$ (hours)", fontsize=12)
        ax.set_yticks(range(len(SITE_ORDER)), short_sites if panel_index == 0 else [])
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(-0.5, len(HORIZONS) - 0.5)
        ax.set_ylim(len(SITE_ORDER) - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, len(HORIZONS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SITE_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        if panel_index == 0:
            ax.set_ylabel("Site", fontsize=12)

    colourbar = fig.colorbar(images[0], ax=axes, shrink=0.86, pad=0.02)
    colourbar.set_label(
        "HLN-corrected DM statistic (positive favours lags + meteorology)",
        fontsize=11,
    )
    colourbar.ax.tick_params(labelsize=10)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            linestyle="None",
            markersize=9,
            label=r"raw $p<0.05$",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="#ffd92f",
            linestyle="None",
            markersize=12,
            label="station-wise BH retained",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=11,
    )
    fig.suptitle(
        "Figure 3. Cell-level DM--HLN inferential landscape",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.25,
        0.035,
        "Global Bonferroni retained: 0/36",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.75,
        0.035,
        "Global Bonferroni retained: 0/36",
        ha="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.27, right=0.90, top=0.80, bottom=0.15, wspace=0.14)
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        metadata={
            "Description": (
                "SERRA final cell-level DM-HLN inferential landscape; "
                "q_overlap=0 and fixed automatic Bartlett Newey-West."
            )
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
