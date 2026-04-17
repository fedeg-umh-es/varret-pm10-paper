"""CLI entrypoint for regime-conditioned diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.regime_analysis import run_regime_analysis
from src.plotting.regime_plots import save_regime_figures


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run regime-conditioned diagnostics for P33.")
    parser.add_argument("--config", default="config/config.yaml", help="Base project config.")
    parser.add_argument(
        "--regime-config",
        default="configs/evaluation/regime_analysis.yaml",
        help="Regime-analysis config.",
    )
    parser.add_argument(
        "--protocol",
        default=None,
        choices=["rolling_origin", "holdout"],
        help="Evaluation protocol to analyze.",
    )
    parser.add_argument("--station", default=None, help="Optional station label.")
    parser.add_argument(
        "--meteorology-source",
        default=None,
        help="Optional CSV/parquet with timestamp-aligned meteorological covariates.",
    )
    return parser.parse_args()


def main() -> None:
    """Run tables, figures, and report generation."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    result = run_regime_analysis(
        repo_root=repo_root,
        config_path=args.config,
        regime_config_path=args.regime_config,
        protocol=args.protocol,
        station=args.station,
        meteorology_source_override=args.meteorology_source,
    )
    figure_paths = save_regime_figures(
        regime_skill=result["regime_skill"],
        seasonal_skill=result["seasonal_skill"],
        figures_dir=result["paths"]["regime_skill"].parent.parent / "figures",
        primary_cluster_scheme=f"cluster_k{result['settings'].clustered_primary_k}",
        max_models=result["settings"].max_models_in_figures,
    )

    print("Regime-conditioned analysis complete.")
    print("Generated tables:")
    print(f"  - {result['paths']['regime_skill']}")
    print(f"  - {result['paths']['regime_events']}")
    print(f"  - {result['paths']['seasonal_skill']}")
    print("Generated figures:")
    for figure_path in figure_paths:
        print(f"  - {figure_path}")
    print("Generated report:")
    print(f"  - {result['paths']['report']}")


if __name__ == "__main__":
    main()
