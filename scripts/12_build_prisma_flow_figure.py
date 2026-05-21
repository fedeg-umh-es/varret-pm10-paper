"""Build a minimal PRISMA-style flow figure for the reporting audit.

This figure uses only the available audit counts:
- Eligible PM10/PM2.5 forecasting studies: 503
- Abstract-coded studies used for quantitative reporting audit: 486
- Studies not used for abstract-level quantitative coding: 17

No additional PRISMA screening counts are inferred or invented.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

FIGURES_DIR = Path("outputs/figures")
PDF_PATH = FIGURES_DIR / "figure0_prisma_flow.pdf"
PNG_PATH = FIGURES_DIR / "figure0_prisma_flow.png"

ELIGIBLE_N = 503
ABSTRACT_CODED_N = 486
NOT_CODED_N = 17


def _validate_counts() -> None:
    if ABSTRACT_CODED_N + NOT_CODED_N != ELIGIBLE_N:
        raise ValueError(
            "Invalid PRISMA audit counts: "
            f"{ABSTRACT_CODED_N} + {NOT_CODED_N} != {ELIGIBLE_N}"
        )


def _draw_box(ax, xy, width, height, text, fontsize=10):
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=1.2,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color="black",
    )
    ax.add_patch(arrow)


def build_prisma_flow_figure() -> None:
    """Build and save the PRISMA-style flow figure."""
    _validate_counts()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    main_width = 5.2
    main_height = 1.15
    side_width = 3.1
    side_height = 1.05

    top_xy = (2.4, 5.25)
    bottom_xy = (2.4, 2.25)
    side_xy = (6.55, 3.65)

    _draw_box(
        ax,
        top_xy,
        main_width,
        main_height,
        "Eligible PM$_{10}$/PM$_{2.5}$ forecasting studies\n$n = 503$",
        fontsize=10,
    )

    ax.text(
        5.0,
        4.65,
        "Abstract-level evidence assessed for coding",
        ha="center",
        va="center",
        fontsize=9,
    )

    _draw_arrow(ax, (5.0, 5.25), (5.0, 3.4))

    _draw_box(
        ax,
        bottom_xy,
        main_width,
        main_height,
        "Studies included in quantitative\nabstract-level reporting audit\n$n = 486$",
        fontsize=10,
    )

    _draw_box(
        ax,
        side_xy,
        side_width,
        side_height,
        "Not used for abstract-level\nquantitative coding\n$n = 17$",
        fontsize=9,
    )

    _draw_arrow(ax, (7.6, 5.25), (8.1, 4.7))

    ax.text(
        5.0,
        0.85,
        "Counts refer to an abstract-level reporting audit, not full-text methodological confirmation.",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {PNG_PATH}")


def main() -> None:
    build_prisma_flow_figure()


if __name__ == "__main__":
    main()
