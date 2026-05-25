"""Build unified variance-retention table for all stations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES_DIR  = Path("outputs/tables")
OUTPUT_PATH = Path("outputs/tables/variance_retention_all_stations.csv")

STATION_META = {
    # PUNTO_MUESTREO: (name, province, station_type, lat, lon, altitude_m, dem_code)
    "03014002_10_M": ("Elx-Agroalimentari", "Alicante", "Suburban/Industrial", 38.24222, -0.68278, 44, "ES1624A"),
    "46250043_10_M": ("Valencia-Vivers", "Valencia", "Urban/Residential", 39.47806, -0.36833, 11, "ES1619A"),
    "46263999_10_M": ("Zarra-EMEP", "Valencia", "Rural Remote EMEP", 39.08278, -1.10083, 855, "ES0012R"),
    "08019052_10_M": ("Barcelona El Port Vell", "Barcelona", "Suburban/Industrial", 41.37494, 2.18684, 1, "ES1870A"),
    "50008001_10_M": ("Alagon", "Zaragoza", "Suburban/Traffic", 41.76280, -1.14330, 235, "ES1418A"),
    "44216001_10_M": ("Teruel", "Teruel", "Urban/Background", 40.33639, -1.10667, 915, "ES1421A"),
    "45153999_10_M": ("San Pablo de los Montes", "Toledo", "Rural Remote/Background", 39.54633, -4.35042, 923, "ES0001R"),
    "08019043_10_M": ("Barcelona L'Eixample", "Barcelona", "Urban/Traffic", 41.38530, 2.15380, 26, "ES1438A"),
    "08019045_10_M": ("Barcelona Zona Universitaria", "Barcelona", "Urban/Background", 41.38490, 2.11960, 24, "ES0567A"),
    "22125001_10_M": ("Huesca", "Huesca", "Urban/Traffic", 42.13611, -0.40389, 488, "ES1417A"),
    "08019004_10_M": ("Barcelona El Poblenou", "Barcelona", "Urban/Background", 41.40388, 2.20452, 3, "ES0691A"),
    "08019028_10_M": ("Barcelona Pl. Universitat", "Barcelona", "Urban/Traffic", 41.38740, 2.16490, 24, "ES0559A"),
    "08019054_10_M": ("Barcelona Vall d'Hebron", "Barcelona", "Urban/Background", 41.42608, 2.14799, 136, "ES1856A"),
    "44013007_10_M": ("Alcaniz Capuchinos", "Teruel", "Urban/Industrial", 41.05417, -0.13889, 340, "ES1966A"),
    "08263007_10_M": ("Sant Vicenc dels Horts Alaba", "Barcelona", "Suburban/Industrial", 41.40077, 1.99963, 65, "ES2011A"),
    "43004006_10_M": ("Alcanar Montecarlo", "Tarragona", "Rural Remote/Industrial", 40.57987, 0.55352, 6, "ES2091A"),
    "43004005_10_M": ("Alcanar Llar de Jubilats", "Tarragona", "Rural Near-city/Industrial", 40.55282, 0.52998, 7, "ES2017A"),
}

FILENAME_STATION_OVERRIDES = {
    "summary": "03014002_10_M",
    "valencia_vivers": "46250043_10_M",
    "zarra_emep": "46263999_10_M",
}

SKIP_STEMS = {"all_stations", "daily", "test"}


def _station_id_from_path(path: Path) -> str | None:
    raw = path.stem.replace("variance_retention_", "")
    if raw in SKIP_STEMS:
        return None
    return FILENAME_STATION_OVERRIDES.get(raw, raw)


def _station_class(station_type: str) -> str:
    low = station_type.lower()
    if "rural" in low:
        return "rural"
    if "industrial" in low:
        return "industrial"
    return "urban"


def main() -> None:
    frames = []
    for f in TABLES_DIR.glob("variance_retention_*.csv"):
        station_id = _station_id_from_path(f)
        if station_id is None:
            continue
        df = pd.read_csv(f)
        meta = STATION_META.get(
            station_id,
            (station_id, "Unknown", "Unknown", float("nan"), float("nan"), float("nan"), "Unknown"),
        )
        df["station_id"] = station_id
        df["station_name"] = meta[0]
        df["province"] = meta[1]
        df["station_type"] = meta[2]
        df["station_class"] = _station_class(meta[2])
        df["lat"] = meta[3]
        df["lon"] = meta[4]
        df["altitude_m"] = meta[5]
        df["dem_code"] = meta[6]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No variance_retention_*.csv files found in {TABLES_DIR}")

    unified = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(unified)}")
    print(f"Stations:   {unified['station_id'].nunique()}")
    print(f"Collapse rate (alpha<0.5): {unified['collapse_flag'].mean():.1%}")
    print("\nCollapse by station_type:")
    print(unified.groupby("station_type")["collapse_flag"].mean().round(3))
    print("\nMean alpha by model:")
    print(unified.groupby("model")["alpha"].mean().round(3))


if __name__ == "__main__":
    main()
