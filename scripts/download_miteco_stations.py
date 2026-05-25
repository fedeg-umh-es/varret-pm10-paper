"""
Download MITECO PM10 daily data for years 2017-2024 and extract
candidate stations with good coverage across Spain.
"""
import io, zipfile, argparse
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

YEARS = list(range(2017, 2025))
MAGNITUD_PM10 = 10
DAY_COLS = [f"D{i:02d}" for i in range(1, 32)]
MIN_COVERAGE = 0.75
MIN_YEARS = 5

EXCLUDED_PUNTOS = {
    "03014002_10_M",  # Elx-Agroalimentari (already in paper)
    "46250043_10_M",  # Valencia-Vivers (already in paper)
    "46263999_10_M",  # Zarra EMEP (already in paper)
}

URL_PATTERNS = [
    "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/calidad-del-aire/evaluacion-datos/datos/PM10_DD_{year}.zip",
    "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/calidad-del-aire/evaluacion-datos/datos/Anio{year}.zip",
]

def parse_csv(raw):
    for enc in ("latin1", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=";", encoding=enc, low_memory=False)
            df.columns = df.columns.str.strip().str.upper()
            return df
        except Exception:
            continue
    return pd.DataFrame()

def wide_to_long(df):
    df = df[df["MAGNITUD"] == MAGNITUD_PM10].copy()
    present = [c for c in DAY_COLS if c in df.columns]
    rows = []
    for _, row in df.iterrows():
        punto = str(row["PUNTO_MUESTREO"]).strip()
        for dc in present:
            day = int(dc[1:])
            try:
                date = pd.Timestamp(year=int(row["ANNO"]), month=int(row["MES"]), day=day)
            except ValueError:
                continue
            try:
                val = float(str(row[dc]).replace(",", "."))
                val = float("nan") if val < 0 else val
            except (ValueError, TypeError):
                val = float("nan")
            rows.append({"date": date, "pm10": val, "punto": punto})
    return pd.DataFrame(rows)

def download_year(year, cache_dir):
    cache = cache_dir / f"PM10_DD_{year}.zip"
    if cache.exists():
        return cache.read_bytes()
    for pat in URL_PATTERNS:
        try:
            r = requests.get(pat.format(year=year), timeout=90)
            if r.status_code == 200:
                cache.write_bytes(r.content)
                print(f"  {year}: OK")
                return r.content
        except Exception:
            continue
    print(f"  {year}: FAILED — add ZIP manually to {cache_dir}")
    return None

def load_year(year, cache_dir):
    raw = download_year(year, cache_dir)
    if raw is None:
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            frames = [parse_csv(zf.read(n)) for n in zf.namelist()
                      if n.lower().endswith(".csv")]
            df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    except zipfile.BadZipFile:
        df = parse_csv(raw)
    if df.empty:
        return pd.DataFrame()
    return wide_to_long(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/miteco_cache", type=Path)
    parser.add_argument("--out-dir",   default="data/raw",          type=Path)
    parser.add_argument("--n-stations", type=int, default=14)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading years ===")
    frames = []
    for y in tqdm(YEARS):
        df = load_year(y, args.cache_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        print("ERROR: no data loaded.")
        return
    all_data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(all_data):,} rows, {all_data['punto'].nunique()} stations")

    print("=== Building inventory ===")
    rows = []
    for punto, grp in all_data.groupby("punto"):
        grp = grp.sort_values("date").drop_duplicates("date")
        idx = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        cov = grp["pm10"].notna().sum() / len(idx) if len(idx) > 0 else 0
        rows.append({"punto": punto, "n_days": len(idx),
                     "valid_days": int(grp["pm10"].notna().sum()),
                     "coverage": round(cov, 3),
                     "n_years": grp["date"].dt.year.nunique(),
                     "date_min": grp["date"].min().date(),
                     "date_max": grp["date"].max().date()})
    inv = pd.DataFrame(rows).sort_values("coverage", ascending=False)
    inv.to_csv(args.out_dir / "../miteco_inventory.csv", index=False)

    good = inv[
        (inv["coverage"] >= MIN_COVERAGE) &
        (inv["n_years"]  >= MIN_YEARS) &
        (~inv["punto"].isin(EXCLUDED_PUNTOS))
    ]
    print(f"Eligible stations: {len(good)}")

    selected = good.head(args.n_stations)
    print("\n=== Selected stations ===")
    print(selected[["punto","coverage","n_years","date_min","date_max"]].to_string(index=False))

    print("\n=== Writing CSVs ===")
    written = []
    for _, row in selected.iterrows():
        punto = row["punto"]
        sdata = (all_data[all_data["punto"] == punto][["date","pm10"]]
                 .sort_values("date").drop_duplicates("date"))
        safe  = punto.replace("/","_").replace(" ","_")
        path  = args.out_dir / f"pm10_{safe}.csv"
        sdata.to_csv(path, index=False)
        written.append({"punto": punto, "file": str(path),
                        "rows": len(sdata), "coverage": row["coverage"]})
        print(f"  {path}  ({len(sdata)} rows)")
    pd.DataFrame(written).to_csv(args.out_dir / "../miteco_selected.csv", index=False)
    print(f"\nDone. {len(written)} station CSVs in {args.out_dir.resolve()}/")

if __name__ == "__main__":
    main()
