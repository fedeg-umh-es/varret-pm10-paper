"""
Regime-conditioned diagnostics for P33.

Internal implementation plan:
1. Read existing predictions and preserve the current rolling-origin outputs.
2. Discover origin-time meteorological covariates from repo-local or sibling
   datasets, falling back to temporal proxies already present in the feature
   matrix when meteorology is unavailable.
3. Align each prediction with the correct target timestamp using the feature
   index -> observation index mapping implied by sample_idx + horizon.
4. Fit fold-specific regime assignment components on training data only:
   percentile thresholds for simple regimes, imputers/scalers/KMeans for
   clustered regimes, and event thresholds for exceedance diagnostics.
5. Aggregate overall, physical-regime, clustered-regime, and seasonal metrics
   without changing the legacy evaluation tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


METEOROLOGY_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "wind": ("wind", "vv", "dv", "u10", "v10"),
    "thermodynamic": ("temp", "temperature", "t_", "_t", "solar", "rad"),
    "moisture": ("humidity", "rh", "dew", "vap", "moist"),
    "pressure": ("pressure", "press", "pb"),
    "precipitation": ("precip", "rain", "snow"),
    "temporal_proxies": ("hour", "dayofweek", "weekday", "month", "season"),
    "boundary_layer_stability_proxies": ("blh", "mix", "stability", "inversion"),
}

CANONICAL_METEO_COLUMNS: dict[str, tuple[str, ...]] = {
    "wind_speed_ms": ("wind_speed_ms", "wind_speed", "vv"),
    "wind_dir_deg": ("wind_dir_deg", "wind_dir", "dv"),
    "temp_c": ("temp_c", "temperature_c", "temperature", "temp", "t"),
    "humidity_pct": ("humidity_pct", "humidity", "rh", "hr"),
    "pressure_hpa": ("pressure_hpa", "pressure", "pb"),
    "solar_rad_wm2": ("solar_rad_wm2", "solar_rad", "rs", "radiation"),
    "precip_mm": ("precip_mm", "precip", "rain", "p"),
}


@dataclass(frozen=True)
class RegimeAnalysisSettings:
    """Typed settings for the regime analysis pipeline."""

    protocol: str
    station: str
    horizons: list[int]
    primary_event_percentile: int
    clustered_k_values: list[int]
    clustered_primary_k: int
    min_samples_per_group: int
    min_events_per_group: int
    ventilation_quantile: float
    humidity_quantile: float
    warm_months: set[int]
    random_state: int
    n_init: int
    max_iter: int
    meteorology_candidate_sources: list[str]
    max_models_in_figures: int


def load_regime_settings(
    repo_root: Path,
    config_path: str | Path,
    protocol_override: str | None = None,
    station_override: str | None = None,
) -> RegimeAnalysisSettings:
    """Load the dedicated regime-analysis configuration."""
    with open(repo_root / config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    analysis = cfg["analysis"]
    physical = cfg["physical_regimes"]
    clustering = cfg["clustering"]
    figures = cfg["figures"]

    return RegimeAnalysisSettings(
        protocol=protocol_override or analysis["default_protocol"],
        station=station_override or analysis["station_default"],
        horizons=list(analysis["horizons"]),
        primary_event_percentile=int(analysis["primary_event_percentile"]),
        clustered_k_values=[int(value) for value in analysis["clustered_k_values"]],
        clustered_primary_k=int(analysis["clustered_primary_k"]),
        min_samples_per_group=int(analysis["min_samples_per_group"]),
        min_events_per_group=int(analysis["min_events_per_group"]),
        ventilation_quantile=float(physical["ventilation_quantile"]),
        humidity_quantile=float(physical["humidity_quantile"]),
        warm_months=set(int(value) for value in physical["warm_months"]),
        random_state=int(clustering["random_state"]),
        n_init=int(clustering["n_init"]),
        max_iter=int(clustering["max_iter"]),
        meteorology_candidate_sources=list(cfg["paths"]["meteorology_candidate_sources"]),
        max_models_in_figures=int(figures["max_models"]),
    )


def run_regime_analysis(
    repo_root: Path,
    config_path: str | Path = "config/config.yaml",
    regime_config_path: str | Path = "configs/evaluation/regime_analysis.yaml",
    protocol: str | None = None,
    station: str | None = None,
    meteorology_source_override: str | None = None,
) -> dict[str, Any]:
    """Run the regime-conditioned evaluation layer and write outputs."""
    base_cfg = _load_yaml(repo_root / config_path)
    settings = load_regime_settings(repo_root, regime_config_path, protocol, station)

    output_dirs = _ensure_output_directories(repo_root, base_cfg)
    observations = pd.read_parquet(repo_root / base_cfg["paths"]["processed_dir"] / "pm10_preprocessed.parquet")
    features = pd.read_parquet(repo_root / base_cfg["paths"]["processed_dir"] / "features_lgbm.parquet")
    horizons = [h for h in settings.horizons if h in set(_load_yaml(repo_root / "config/horizons.yaml")["horizons"])]
    if not horizons:
        raise ValueError("No valid horizons found for regime analysis.")

    origin_covariates, discovery_summary = build_origin_covariates(
        repo_root=repo_root,
        features=features,
        candidate_sources=settings.meteorology_candidate_sources,
        meteorology_source_override=meteorology_source_override,
    )
    discovery_path = output_dirs["metadata"] / "available_meteorological_variables.json"
    discovery_path.write_text(json.dumps(discovery_summary, indent=2), encoding="utf-8")

    splits = load_protocol_splits(repo_root, base_cfg, settings.protocol)
    master = build_prediction_master(
        repo_root=repo_root,
        base_cfg=base_cfg,
        settings=settings,
        observations=observations,
        features=features,
        origin_covariates=origin_covariates,
        splits=splits,
        horizons=horizons,
    )
    if master.empty:
        raise ValueError("No prediction rows were available for the selected protocol.")

    master, cluster_metadata = assign_regimes(master, origin_covariates, observations, features, splits, settings)

    cluster_path = output_dirs["metadata"] / "clustered_regime_centers.json"
    cluster_path.write_text(json.dumps(cluster_metadata, indent=2), encoding="utf-8")

    regime_skill = build_regime_skill_summary(master, settings)
    regime_events = build_regime_event_summary(master, settings)
    seasonal_skill = build_seasonal_skill_summary(master, settings)

    regime_skill_path = output_dirs["tables"] / "regime_skill_summary.csv"
    regime_events_path = output_dirs["tables"] / "regime_event_summary.csv"
    seasonal_skill_path = output_dirs["tables"] / "seasonal_skill_summary.csv"
    regime_skill.to_csv(regime_skill_path, index=False)
    regime_events.to_csv(regime_events_path, index=False)
    seasonal_skill.to_csv(seasonal_skill_path, index=False)

    report_path = output_dirs["reports"] / "regime_analysis_report.md"
    report_path.write_text(
        build_markdown_report(
            discovery_summary=discovery_summary,
            regime_skill=regime_skill,
            regime_events=regime_events,
            seasonal_skill=seasonal_skill,
            cluster_metadata=cluster_metadata,
            settings=settings,
        ),
        encoding="utf-8",
    )

    return {
        "master": master,
        "regime_skill": regime_skill,
        "regime_events": regime_events,
        "seasonal_skill": seasonal_skill,
        "discovery_summary": discovery_summary,
        "cluster_metadata": cluster_metadata,
        "paths": {
            "metadata": output_dirs["metadata"],
            "regime_skill": regime_skill_path,
            "regime_events": regime_events_path,
            "seasonal_skill": seasonal_skill_path,
            "report": report_path,
        },
        "settings": settings,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_output_directories(repo_root: Path, base_cfg: dict[str, Any]) -> dict[str, Path]:
    directories = {
        "tables": repo_root / base_cfg["paths"]["tables_dir"],
        "figures": repo_root / base_cfg["paths"]["figures_dir"],
        "reports": repo_root / "outputs" / "reports",
        "metadata": repo_root / "outputs" / "metadata",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def build_origin_covariates(
    repo_root: Path,
    features: pd.DataFrame,
    candidate_sources: list[str],
    meteorology_source_override: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the origin-time covariate frame and discovery metadata."""
    origin_covariates = features.copy()
    sources_used: list[dict[str, Any]] = []
    external_source = meteorology_source_override or _find_first_existing_source(repo_root, candidate_sources)

    if external_source is not None:
        external_df, source_summary = _load_external_meteorology_source(Path(external_source))
        aligned_external = external_df.reindex(origin_covariates.index)
        for column in aligned_external.columns:
            if column not in origin_covariates.columns:
                origin_covariates[column] = aligned_external[column]
        sources_used.append(source_summary)

    categories = _categorize_columns(origin_covariates.columns.tolist())
    category_payload = {
        category: sorted(columns) for category, columns in categories.items()
    }

    summary = {
        "sources_used": sources_used,
        "selected_external_source": external_source,
        "available_columns": origin_covariates.columns.tolist(),
        "category_columns": category_payload,
        "fallback_used": not bool(category_payload["wind"] or category_payload["moisture"] or category_payload["thermodynamic"] or category_payload["pressure"] or category_payload["precipitation"]),
    }
    return origin_covariates, summary


def _find_first_existing_source(repo_root: Path, candidate_sources: list[str]) -> str | None:
    for source in candidate_sources:
        candidate = (repo_root / source).resolve() if not Path(source).is_absolute() else Path(source)
        if candidate.exists():
            return str(candidate)
    return None


def _load_external_meteorology_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an external meteorology file and harmonize its columns."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported meteorology source format: {path}")

    timestamp_column = _infer_timestamp_column(df.columns.tolist())
    if timestamp_column is None:
        raise ValueError(f"Could not infer a timestamp column from {path}")

    df = df.rename(columns={timestamp_column: "timestamp"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    harmonized = pd.DataFrame(index=pd.Index(df["timestamp"], name="timestamp"))

    for canonical, aliases in CANONICAL_METEO_COLUMNS.items():
        for column in df.columns:
            normalized = column.lower()
            if normalized == "timestamp":
                continue
            if any(alias in normalized for alias in aliases):
                harmonized[canonical] = pd.to_numeric(df[column], errors="coerce").to_numpy()
                break

    summary = {
        "path": str(path),
        "columns": harmonized.columns.tolist(),
        "row_count": int(len(harmonized)),
        "start": str(harmonized.index.min()) if len(harmonized) else None,
        "end": str(harmonized.index.max()) if len(harmonized) else None,
        "coverage_pct": {
            column: float(harmonized[column].notna().mean() * 100.0)
            for column in harmonized.columns
        },
    }
    return harmonized.sort_index(), summary


def _infer_timestamp_column(columns: list[str]) -> str | None:
    for column in columns:
        lowered = column.lower()
        if "timestamp" in lowered or lowered == "date" or "datetime" in lowered:
            return column
    return None


def _categorize_columns(columns: list[str]) -> dict[str, list[str]]:
    categories = {category: [] for category in METEOROLOGY_CATEGORY_PATTERNS}
    for column in columns:
        lowered = column.lower()
        for category, patterns in METEOROLOGY_CATEGORY_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                categories[category].append(column)
    return categories


def load_protocol_splits(repo_root: Path, base_cfg: dict[str, Any], protocol: str) -> dict[str, Any]:
    """Load the JSON split description for the requested protocol."""
    processed_dir = repo_root / base_cfg["paths"]["processed_dir"]
    if protocol == "rolling_origin":
        path = processed_dir / "splits_rolling_origin.json"
    elif protocol == "holdout":
        path = processed_dir / "splits_holdout.json"
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prediction_master(
    repo_root: Path,
    base_cfg: dict[str, Any],
    settings: RegimeAnalysisSettings,
    observations: pd.DataFrame,
    features: pd.DataFrame,
    origin_covariates: pd.DataFrame,
    splits: dict[str, Any],
    horizons: list[int],
) -> pd.DataFrame:
    """Combine predictions, truth, persistence, and origin-time covariates."""
    observations_series = observations.iloc[:, 0]
    feature_to_obs_pos = observations_series.index.get_indexer(features.index)
    if np.any(feature_to_obs_pos < 0):
        raise ValueError("Feature timestamps could not be aligned to observation timestamps.")

    predictions_dir = repo_root / base_cfg["paths"]["predictions_dir"]
    suffix = settings.protocol

    persistence_path = predictions_dir / f"persistence_{suffix}.parquet"
    if not persistence_path.exists():
        raise FileNotFoundError(f"Missing persistence baseline: {persistence_path}")
    persistence = pd.read_parquet(persistence_path).copy()
    persistence["fold"] = persistence["fold"] if "fold" in persistence.columns else 0
    persistence = persistence[persistence["horizon"].isin(horizons)]
    persistence_map = persistence.set_index(["fold", "sample_idx", "horizon"])["y_pred"]

    prediction_frames: list[pd.DataFrame] = []
    for prediction_path in sorted(predictions_dir.glob(f"*_{suffix}.parquet")):
        model = prediction_path.stem[: -len(f"_{suffix}")]
        if model == "persistence":
            continue
        df_model = pd.read_parquet(prediction_path).copy()
        df_model["fold"] = df_model["fold"] if "fold" in df_model.columns else 0
        df_model["model"] = model
        df_model = df_model[df_model["horizon"].isin(horizons)]
        prediction_frames.append(df_model)

    if not prediction_frames:
        raise FileNotFoundError(f"No model predictions found for protocol '{settings.protocol}'.")

    master = pd.concat(prediction_frames, ignore_index=True)
    origin_obs_pos = feature_to_obs_pos[master["sample_idx"].to_numpy(dtype=int)]
    target_obs_pos = origin_obs_pos + master["horizon"].to_numpy(dtype=int)
    valid_mask = target_obs_pos < len(observations_series)
    master = master.loc[valid_mask].copy()
    origin_obs_pos = origin_obs_pos[valid_mask]
    target_obs_pos = target_obs_pos[valid_mask]

    persistence_keys = list(zip(master["fold"], master["sample_idx"], master["horizon"]))
    master["y_persist"] = [float(persistence_map.loc[key]) for key in persistence_keys]
    master["origin_obs_pos"] = origin_obs_pos
    master["target_obs_pos"] = target_obs_pos
    master["origin_timestamp"] = features.index[master["sample_idx"].to_numpy(dtype=int)]
    master["target_timestamp"] = observations_series.index[target_obs_pos]
    master["y_true"] = observations_series.iloc[target_obs_pos].to_numpy(dtype=float)
    master["station"] = settings.station

    aligned_covariates = origin_covariates.iloc[master["sample_idx"].to_numpy(dtype=int)].reset_index(drop=True)
    for column in aligned_covariates.columns:
        master[column] = aligned_covariates[column].to_numpy()

    fold_thresholds = _build_fold_horizon_thresholds(
        settings=settings,
        observations_series=observations_series,
        feature_to_obs_pos=feature_to_obs_pos,
        splits=splits,
        horizons=horizons,
        protocol=settings.protocol,
    )
    master["event_threshold"] = [
        fold_thresholds[(int(fold), int(horizon))]
        for fold, horizon in zip(master["fold"], master["horizon"])
    ]
    master["origin_month"] = pd.to_datetime(master["origin_timestamp"]).dt.month
    master["origin_hour"] = pd.to_datetime(master["origin_timestamp"]).dt.hour
    master["origin_season"] = pd.to_datetime(master["origin_timestamp"]).map(_month_to_season)
    master["warm_cold_season"] = np.where(master["origin_month"].isin(settings.warm_months), "warm_season", "cold_season")
    return master.sort_values(["model", "fold", "horizon", "sample_idx"]).reset_index(drop=True)


def _build_fold_horizon_thresholds(
    settings: RegimeAnalysisSettings,
    observations_series: pd.Series,
    feature_to_obs_pos: np.ndarray,
    splits: dict[str, Any],
    horizons: list[int],
    protocol: str,
) -> dict[tuple[int, int], float]:
    """Calibrate event thresholds on training targets only."""
    if protocol == "rolling_origin":
        fold_iterable = splits["folds"]
    else:
        fold_iterable = [{"fold": 0, "train_idx": splits["train_idx"]}]

    thresholds: dict[tuple[int, int], float] = {}
    for fold_info in fold_iterable:
        fold = int(fold_info["fold"])
        train_idx = np.asarray(fold_info["train_idx"], dtype=int)
        train_obs_pos = feature_to_obs_pos[train_idx]
        for horizon in horizons:
            target_positions = train_obs_pos + horizon
            valid_positions = target_positions[target_positions < len(observations_series)]
            if len(valid_positions) == 0:
                raise ValueError(f"No valid training targets for fold={fold}, horizon={horizon}")
            thresholds[(fold, horizon)] = float(
                np.percentile(observations_series.iloc[valid_positions].to_numpy(dtype=float), settings.primary_event_percentile)
            )
    return thresholds


def assign_regimes(
    master: pd.DataFrame,
    origin_covariates: pd.DataFrame,
    observations: pd.DataFrame,
    features: pd.DataFrame,
    splits: dict[str, Any],
    settings: RegimeAnalysisSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign physical and clustered regimes using training data only per fold."""
    enriched = master.copy()
    cluster_metadata: dict[str, Any] = {"k_values": {}}
    enriched["physical_ventilation_regime"] = "unavailable"
    enriched["physical_moisture_regime"] = "unavailable"
    enriched["physical_season_regime"] = np.where(
        enriched["origin_month"].isin(settings.warm_months),
        "warm_season",
        "cold_season",
    )

    if settings.protocol == "rolling_origin":
        fold_iterable = splits["folds"]
    else:
        fold_iterable = [{"fold": 0, "train_idx": splits["train_idx"]}]

    wind_column = _pick_first_available(origin_covariates.columns, ["wind_speed_ms", "wind_speed", "vv"])
    humidity_column = _pick_first_available(origin_covariates.columns, ["humidity_pct", "humidity", "rh", "hr"])
    cluster_columns = _select_cluster_columns(origin_covariates.columns.tolist())

    for fold_info in fold_iterable:
        fold = int(fold_info["fold"])
        train_idx = np.asarray(fold_info["train_idx"], dtype=int)
        fold_mask = enriched["fold"] == fold
        fold_train_covariates = origin_covariates.iloc[train_idx].copy()

        if wind_column is not None:
            wind_threshold = float(fold_train_covariates[wind_column].dropna().quantile(settings.ventilation_quantile))
            enriched.loc[fold_mask, "physical_ventilation_regime"] = np.where(
                enriched.loc[fold_mask, wind_column] <= wind_threshold,
                "low_ventilation",
                "high_ventilation",
            )

        if humidity_column is not None:
            humidity_threshold = float(fold_train_covariates[humidity_column].dropna().quantile(settings.humidity_quantile))
            enriched.loc[fold_mask, "physical_moisture_regime"] = np.where(
                enriched.loc[fold_mask, humidity_column] <= humidity_threshold,
                "dry",
                "humid",
            )

        for k in settings.clustered_k_values:
            column_name = f"cluster_k{k}_regime"
            if column_name not in enriched.columns:
                enriched[column_name] = "unavailable"

            cluster_payload = _fit_predict_cluster_regimes(
                full_covariates=origin_covariates,
                fold_train_idx=train_idx,
                fold_test_idx=enriched.loc[fold_mask, "sample_idx"].to_numpy(dtype=int),
                columns=cluster_columns,
                k=k,
                settings=settings,
            )
            if cluster_payload is None:
                continue

            labels = cluster_payload["test_labels"]
            sample_indices = enriched.index[fold_mask].to_numpy()
            enriched.loc[sample_indices, column_name] = labels
            cluster_metadata["k_values"].setdefault(f"k{k}", {})[f"fold_{fold}"] = {
                "columns": cluster_payload["columns"],
                "centers_original_units": cluster_payload["centers_original_units"],
                "cluster_labels": cluster_payload["cluster_labels"],
                "n_train": int(cluster_payload["n_train"]),
                "n_test": int(cluster_payload["n_test"]),
            }

    return enriched, cluster_metadata


def _pick_first_available(columns: pd.Index | list[str], candidates: list[str]) -> str | None:
    available = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        for lowered, original in available.items():
            if candidate in lowered:
                return original
    return None


def _select_cluster_columns(columns: list[str]) -> list[str]:
    preferred = [
        "wind_speed_ms",
        "wind_dir_deg",
        "temp_c",
        "humidity_pct",
        "pressure_hpa",
        "solar_rad_wm2",
        "precip_mm",
        "hour",
        "month",
        "dayofweek",
    ]
    selected = [column for column in preferred if column in columns]
    if not selected:
        selected = list(columns)
    return selected


def _fit_predict_cluster_regimes(
    full_covariates: pd.DataFrame,
    fold_train_idx: np.ndarray,
    fold_test_idx: np.ndarray,
    columns: list[str],
    k: int,
    settings: RegimeAnalysisSettings,
) -> dict[str, Any] | None:
    """Fit fold-specific KMeans regimes and return readable cluster labels."""
    if len(columns) == 0:
        return None

    train = full_covariates.iloc[fold_train_idx][columns].copy()
    test = full_covariates.iloc[fold_test_idx][columns].copy()
    usable_columns = [column for column in columns if train[column].notna().sum() > k]
    if len(usable_columns) == 0:
        return None

    train = train[usable_columns]
    test = test[usable_columns]
    train_medians = train.median()
    train_filled = train.fillna(train_medians)
    test_filled = test.fillna(train_medians)

    if len(train_filled) < max(k * 10, k + 1):
        return None

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)
    test_scaled = scaler.transform(test_filled)

    model = KMeans(
        n_clusters=k,
        random_state=settings.random_state,
        n_init=settings.n_init,
        max_iter=settings.max_iter,
    )
    model.fit(train_scaled)
    test_cluster_ids = model.predict(test_scaled)
    centers_original = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=usable_columns,
    )
    cluster_labels = _label_cluster_centers(centers_original, train_filled)
    labeled_predictions = [cluster_labels[int(cluster_id)] for cluster_id in test_cluster_ids]

    return {
        "columns": usable_columns,
        "test_labels": labeled_predictions,
        "cluster_labels": cluster_labels,
        "centers_original_units": centers_original.round(4).to_dict(orient="records"),
        "n_train": len(train_filled),
        "n_test": len(test_filled),
    }


def _label_cluster_centers(centers: pd.DataFrame, train_reference: pd.DataFrame) -> dict[int, str]:
    """Generate compact, human-readable labels from cluster centers."""
    labels: dict[int, str] = {}
    train_quantiles = train_reference.quantile([0.33, 0.67])
    for cluster_id, row in centers.iterrows():
        descriptors: list[str] = []
        if "wind_speed_ms" in row.index:
            if row["wind_speed_ms"] <= train_quantiles.loc[0.33, "wind_speed_ms"]:
                descriptors.append("calm")
            elif row["wind_speed_ms"] >= train_quantiles.loc[0.67, "wind_speed_ms"]:
                descriptors.append("ventilated")
        if "humidity_pct" in row.index:
            if row["humidity_pct"] <= train_quantiles.loc[0.33, "humidity_pct"]:
                descriptors.append("dry")
            elif row["humidity_pct"] >= train_quantiles.loc[0.67, "humidity_pct"]:
                descriptors.append("humid")
        if "temp_c" in row.index:
            if row["temp_c"] <= train_quantiles.loc[0.33, "temp_c"]:
                descriptors.append("cool")
            elif row["temp_c"] >= train_quantiles.loc[0.67, "temp_c"]:
                descriptors.append("warm")
        if "precip_mm" in row.index and row["precip_mm"] > 0.1:
            descriptors.append("wet")
        if "solar_rad_wm2" in row.index and row["solar_rad_wm2"] >= train_quantiles.loc[0.67, "solar_rad_wm2"]:
            descriptors.append("sunny")

        if not descriptors:
            descriptors = [f"cluster_{cluster_id}"]
        labels[int(cluster_id)] = f"cluster_{cluster_id}_" + "_".join(dict.fromkeys(descriptors[:3]))
    return labels


def build_regime_skill_summary(master: pd.DataFrame, settings: RegimeAnalysisSettings) -> pd.DataFrame:
    """Aggregate overall, physical, and clustered skill diagnostics."""
    frames = [
        _aggregate_scheme(master, settings, scheme_type="overall", scheme_name="overall", regime_column=None),
        _aggregate_scheme(master, settings, scheme_type="physical", scheme_name="ventilation", regime_column="physical_ventilation_regime"),
        _aggregate_scheme(master, settings, scheme_type="physical", scheme_name="moisture", regime_column="physical_moisture_regime"),
        _aggregate_scheme(master, settings, scheme_type="physical", scheme_name="season", regime_column="physical_season_regime"),
    ]
    for k in settings.clustered_k_values:
        frames.append(
            _aggregate_scheme(
                master,
                settings,
                scheme_type="clustered",
                scheme_name=f"cluster_k{k}",
                regime_column=f"cluster_k{k}_regime",
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["scheme_type", "scheme_name", "regime", "model", "horizon"]).reset_index(drop=True)


def build_regime_event_summary(master: pd.DataFrame, settings: RegimeAnalysisSettings) -> pd.DataFrame:
    """Return the event-oriented subset of the regime summary."""
    skill_summary = build_regime_skill_summary(master, settings)
    columns = [
        "protocol",
        "station",
        "scheme_type",
        "scheme_name",
        "regime",
        "model",
        "horizon",
        "sample_count",
        "event_count",
        "base_rate",
        "event_recall",
        "event_precision",
        "flag_rate",
        "insufficient_samples",
        "insufficient_events",
    ]
    return skill_summary[columns].copy()


def build_seasonal_skill_summary(master: pd.DataFrame, settings: RegimeAnalysisSettings) -> pd.DataFrame:
    """Aggregate diagnostics for seasonal robustness views."""
    frames = [
        _aggregate_scheme(master, settings, scheme_type="seasonal", scheme_name="warm_cold", regime_column="warm_cold_season"),
        _aggregate_scheme(master, settings, scheme_type="seasonal", scheme_name="four_seasons", regime_column="origin_season"),
    ]
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["scheme_name", "regime", "model", "horizon"]).reset_index(drop=True)


def _aggregate_scheme(
    master: pd.DataFrame,
    settings: RegimeAnalysisSettings,
    scheme_type: str,
    scheme_name: str,
    regime_column: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if regime_column is None:
        grouped = [(("all", model, horizon), df_group) for (model, horizon), df_group in master.groupby(["model", "horizon"], sort=True)]
    else:
        grouped = [
            ((regime, model, horizon), df_group)
            for (regime, model, horizon), df_group in master.groupby([regime_column, "model", "horizon"], dropna=False, sort=True)
        ]

    for key, df_group in grouped:
        if regime_column is None:
            regime, model, horizon = key
        else:
            regime, model, horizon = key
        rows.append(
            _compute_group_metrics(
                df_group=df_group,
                regime=str(regime),
                model=str(model),
                horizon=int(horizon),
                scheme_type=scheme_type,
                scheme_name=scheme_name,
                settings=settings,
            )
        )
    return pd.DataFrame(rows)


def _compute_group_metrics(
    df_group: pd.DataFrame,
    regime: str,
    model: str,
    horizon: int,
    scheme_type: str,
    scheme_name: str,
    settings: RegimeAnalysisSettings,
) -> dict[str, Any]:
    y_true = df_group["y_true"].to_numpy(dtype=float)
    y_pred = df_group["y_pred"].to_numpy(dtype=float)
    y_persist = df_group["y_persist"].to_numpy(dtype=float)
    thresholds = df_group["event_threshold"].to_numpy(dtype=float)

    sample_count = int(len(df_group))
    event_true = y_true > thresholds
    event_pred = y_pred > thresholds
    event_count = int(event_true.sum())
    predicted_event_count = int(event_pred.sum())

    insufficient_samples = sample_count < settings.min_samples_per_group
    insufficient_events = event_count < settings.min_events_per_group
    rmse_model = _rmse(y_true, y_pred) if not insufficient_samples else np.nan
    mae_model = _mae(y_true, y_pred) if not insufficient_samples else np.nan
    rmse_persist = _rmse(y_true, y_persist) if not insufficient_samples else np.nan
    skill_rmse = 1.0 - (rmse_model / rmse_persist) if not insufficient_samples and rmse_persist and rmse_persist > 0 else np.nan
    var_obs = float(np.var(y_true)) if not insufficient_samples else np.nan
    var_pred = float(np.var(y_pred)) if not insufficient_samples else np.nan
    variance_ratio = (var_pred / var_obs) if not insufficient_samples and var_obs > 0 else np.nan
    vr = 100.0 * variance_ratio if variance_ratio == variance_ratio else np.nan
    skill_vp = skill_rmse * min(1.0, variance_ratio) if skill_rmse == skill_rmse and variance_ratio == variance_ratio else np.nan

    if insufficient_samples or insufficient_events:
        event_recall = np.nan
        event_precision = np.nan
    else:
        tp = int(np.sum(event_true & event_pred))
        fn = int(np.sum(event_true & ~event_pred))
        fp = int(np.sum(~event_true & event_pred))
        event_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        event_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan

    flag_rate = float(predicted_event_count / sample_count) if sample_count > 0 else np.nan
    base_rate = float(event_count / sample_count) if sample_count > 0 else np.nan

    return {
        "protocol": settings.protocol,
        "station": df_group["station"].iloc[0],
        "scheme_type": scheme_type,
        "scheme_name": scheme_name,
        "regime": regime,
        "model": model,
        "horizon": horizon,
        "sample_count": sample_count,
        "event_count": event_count,
        "rmse": rmse_model,
        "mae": mae_model,
        "rmse_persistence": rmse_persist,
        "skill_rmse": skill_rmse,
        "var_obs": var_obs,
        "var_pred": var_pred,
        "vr": vr,
        "skill_vp": skill_vp,
        "base_rate": base_rate,
        "event_recall": event_recall,
        "event_precision": event_precision,
        "flag_rate": flag_rate,
        "insufficient_samples": bool(insufficient_samples),
        "insufficient_events": bool(insufficient_events),
    }


def build_markdown_report(
    discovery_summary: dict[str, Any],
    regime_skill: pd.DataFrame,
    regime_events: pd.DataFrame,
    seasonal_skill: pd.DataFrame,
    cluster_metadata: dict[str, Any],
    settings: RegimeAnalysisSettings,
) -> str:
    """Generate the requested Markdown report."""
    available_columns = discovery_summary["available_columns"]
    category_columns = discovery_summary["category_columns"]
    sources_used = discovery_summary["sources_used"]

    headline_lines = _build_headline_findings(regime_skill, regime_events, seasonal_skill, settings)
    caution_lines = _build_cautions(discovery_summary, regime_skill)
    sample_table = _sample_size_markdown(regime_skill)

    lines = [
        "# Regime-conditioned analysis report",
        "",
        "## Detected meteorological variables",
        f"- Columns available at forecast origin: {', '.join(available_columns) if available_columns else 'none'}",
        f"- Wind: {', '.join(category_columns['wind']) if category_columns['wind'] else 'none'}",
        f"- Thermodynamic: {', '.join(category_columns['thermodynamic']) if category_columns['thermodynamic'] else 'none'}",
        f"- Moisture: {', '.join(category_columns['moisture']) if category_columns['moisture'] else 'none'}",
        f"- Pressure: {', '.join(category_columns['pressure']) if category_columns['pressure'] else 'none'}",
        f"- Precipitation: {', '.join(category_columns['precipitation']) if category_columns['precipitation'] else 'none'}",
        f"- Temporal proxies: {', '.join(category_columns['temporal_proxies']) if category_columns['temporal_proxies'] else 'none'}",
        f"- Boundary-layer/stability proxies: {', '.join(category_columns['boundary_layer_stability_proxies']) if category_columns['boundary_layer_stability_proxies'] else 'none'}",
        f"- External sources used: {', '.join(source['path'] for source in sources_used) if sources_used else 'none; fallback to repo-local covariates only'}",
        "",
        "## Regime construction logic",
        "- Physical regimes:",
        f"  - ventilation: fold-specific {settings.ventilation_quantile:.2f} quantile on wind-speed-like variable when available",
        f"  - moisture: fold-specific {settings.humidity_quantile:.2f} quantile on humidity-like variable when available",
        f"  - season: warm months {sorted(settings.warm_months)} vs complementary cold months",
        "- Clustered regimes:",
        f"  - KMeans fitted per fold on training data only with k in {settings.clustered_k_values}",
        "  - missing covariates imputed with training medians only",
        "  - standardization fitted on training covariates only and inverted for center reporting",
        "  - labels assigned from dominant center characteristics",
        "",
        "## Anti-leakage safeguards",
        "- Rolling-origin or holdout split definitions are reused without modification.",
        "- Regime thresholds are calibrated inside each fold using training origins only.",
        "- Cluster medians, scaling, and KMeans fits are trained on fold-specific training covariates only.",
        "- Event thresholds are recalibrated by fold and horizon using only training targets aligned to t+h.",
        "- The new analysis is additive and does not overwrite the legacy P33 tables.",
        "",
        "## Sample sizes by regime and horizon",
        "",
        sample_table,
        "",
        "## Headline findings",
    ]
    lines.extend(f"- {line}" for line in headline_lines)
    lines.extend(
        [
            "",
            "## Cluster-center metadata",
            f"- Stored in `outputs/metadata/clustered_regime_centers.json` with fold-specific centers for k in {settings.clustered_k_values}.",
            f"- Cluster metadata entries available: {len(cluster_metadata.get('k_values', {}))}",
            "",
            "## Cautions and limitations",
        ]
    )
    lines.extend(f"- {line}" for line in caution_lines)
    return "\n".join(lines) + "\n"


def _build_headline_findings(
    regime_skill: pd.DataFrame,
    regime_events: pd.DataFrame,
    seasonal_skill: pd.DataFrame,
    settings: RegimeAnalysisSettings,
) -> list[str]:
    findings: list[str] = []
    reliable = regime_skill[
        (~regime_skill["insufficient_samples"])
        & (regime_skill["scheme_type"] != "overall")
        & (regime_skill["vr"].notna())
    ].copy()
    if not reliable.empty:
        worst_vr = reliable.sort_values("vr", ascending=True).iloc[0]
        findings.append(
            f"Variance-retention collapse is strongest under `{worst_vr['scheme_name']}:{worst_vr['regime']}` for `{worst_vr['model']}` at h={int(worst_vr['horizon'])} (VR={worst_vr['vr']:.1f}%)."
        )

    decoupling = reliable[(reliable["skill_rmse"] > 0) & (reliable["skill_vp"].notna())].copy()
    if not decoupling.empty:
        decoupling["gap"] = decoupling["skill_rmse"] - decoupling["skill_vp"]
        strongest_gap = decoupling.sort_values("gap", ascending=False).iloc[0]
        findings.append(
            f"RMSE skill remains positive but Skill_VP weakens most under `{strongest_gap['scheme_name']}:{strongest_gap['regime']}` for `{strongest_gap['model']}` at h={int(strongest_gap['horizon'])} (skill={strongest_gap['skill_rmse']:.3f}, Skill_VP={strongest_gap['skill_vp']:.3f})."
        )

    event_view = regime_events[
        (~regime_events["insufficient_samples"])
        & (~regime_events["insufficient_events"])
        & (regime_events["scheme_type"] != "overall")
        & (regime_events["event_recall"].notna())
    ].copy()
    if not event_view.empty:
        weakest_recall = event_view.sort_values("event_recall", ascending=True).iloc[0]
        findings.append(
            f"Event recall degradation is concentrated in `{weakest_recall['scheme_name']}:{weakest_recall['regime']}` for `{weakest_recall['model']}` at h={int(weakest_recall['horizon'])} (recall={weakest_recall['event_recall']:.3f})."
        )

    seasonal_view = seasonal_skill[
        (~seasonal_skill["insufficient_samples"])
        & (seasonal_skill["scheme_name"] == "warm_cold")
        & (seasonal_skill["skill_vp"].notna())
    ].copy()
    if not seasonal_view.empty:
        seasonal_view["rank"] = seasonal_view.groupby(["model", "horizon"])["skill_vp"].rank(ascending=True, method="first")
        weakest = seasonal_view.sort_values("skill_vp", ascending=True).iloc[0]
        findings.append(
            f"Seasonal robustness is weakest in `{weakest['regime']}` for `{weakest['model']}` at h={int(weakest['horizon'])} (Skill_VP={weakest['skill_vp']:.3f})."
        )

    if not findings:
        findings.append("No reliable regime-specific contrasts were available under the current minimum sample and event-count rules.")
    return findings


def _build_cautions(discovery_summary: dict[str, Any], regime_skill: pd.DataFrame) -> list[str]:
    cautions: list[str] = []
    if discovery_summary["fallback_used"]:
        cautions.append("No in-repo meteorology fields were detected; the analysis falls back to temporal proxies unless an external aligned source is available.")
    if not discovery_summary["sources_used"]:
        cautions.append("No external meteorology source was loaded, so clustered regimes are proxy-based rather than fully meteorological.")
    unreliable = regime_skill[(regime_skill["insufficient_samples"]) | (regime_skill["insufficient_events"])]
    if not unreliable.empty:
        cautions.append(f"{len(unreliable)} regime/model/horizon cells were flagged for low support and their event metrics were suppressed.")
    cautions.append("This layer is additive; legacy P33 outputs remain unchanged and may not match the stricter target alignment used here.")
    return cautions


def _sample_size_markdown(regime_skill: pd.DataFrame) -> str:
    view = regime_skill[
        regime_skill["scheme_type"].isin(["physical", "clustered"])
    ][["scheme_name", "regime", "horizon", "sample_count", "insufficient_samples"]].copy()
    if view.empty:
        return "No regime-conditioned rows were generated."
    view["flag"] = np.where(view["insufficient_samples"], "low_support", "ok")
    grouped = (
        view.groupby(["scheme_name", "regime", "horizon"], as_index=False)
        .agg(sample_count=("sample_count", "max"), support_flag=("flag", "max"))
        .sort_values(["scheme_name", "regime", "horizon"])
    )
    headers = ["scheme_name", "regime", "horizon", "sample_count", "support_flag"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[header]) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _month_to_season(timestamp: pd.Timestamp) -> str:
    month = int(timestamp.month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"
