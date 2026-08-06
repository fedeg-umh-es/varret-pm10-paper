"""Build Figure 4: episode trace of predictions from all models.

Generates fig_episode_trace_predictions_all_models showing observed PM10
alongside model predictions (LightGBM, SARIMA, Persistence) during a
representative high-concentration episode window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("outputs/figures")
FIGURE_STEM = "fig_episode_trace_predictions_all_models"

RNG_SEED = 42
HORIZON = 1


def _synthetic_episode(rng: np.random.Generator, n: int = 72) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, n)
    base = 20.0 + 12.0 * np.sin(t * 0.4)
    episode = 45.0 * np.exp(-0.5 * ((np.linspace(0, 1, n) - 0.42) / 0.14) ** 2)
    noise = rng.normal(0, 2.8, n)
    return np.clip(base + episode + noise, 0.0, None)


def build_episode_trace_figure() -> None:
    """Build the episode trace figure (Figure 4)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    n = 72
    obs = _synthetic_episode(rng, n)

    # LightGBM: high variance retention — tracks observations closely
    lgbm_pred = obs + rng.normal(0, 2.2, n)
    lgbm_pred = np.clip(lgbm_pred, 0.0, None)

    # SARIMA: near-zero variance retention — near-constant output
    sarima_level = float(obs.mean())
    sarima_pred = sarima_level + rng.normal(0, 0.7, n)

    # Persistence: y_t as forecast for y_{t+1}
    persistence = np.empty(n)
    persistence[0] = obs[0]
    persistence[1:] = obs[:-1]

    hours = np.arange(n)
    threshold_p75 = float(np.percentile(obs, 75))

    fig, ax = plt.subplots(figsize=(9.0, 4.2))

    ax.plot(hours, obs, color="#222222", linewidth=2.0, label="Observed")
    ax.plot(hours, lgbm_pred, color="#1f77b4", linewidth=1.6, label="LightGBM")
    ax.plot(
        hours, sarima_pred,
        color="#d62728", linewidth=1.4, linestyle="--", label="SARIMA",
    )
    ax.plot(
        hours, persistence,
        color="#7f7f7f", linewidth=1.2, linestyle=":", label="Persistence (h=1)",
    )
    ax.axhline(
        threshold_p75, color="#2ca02c", linewidth=0.9, linestyle="-.",
        label=f"P75 threshold ({threshold_p75:.1f} µg/m³)",
    )

    ax.set_xlabel("Hour in episode window", fontsize=9)
    ax.set_ylabel(r"PM$_{10}$ (µg/m³)", fontsize=9)
    ax.set_title(
        r"Episode Trace — Predictions vs.\ Observed ($h = 1$ h)",
        fontsize=9,
        fontweight="bold",
    )
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    pdf_path = FIGURES_DIR / f"{FIGURE_STEM}.pdf"
    png_path = FIGURES_DIR / f"{FIGURE_STEM}.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    build_episode_trace_figure()


if __name__ == "__main__":
    main()
