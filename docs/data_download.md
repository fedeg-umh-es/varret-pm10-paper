# Data Download Instructions

## PM10 Casa de Campo, Madrid — EEA Dataset

The canonical dataset for P33 is **daily mean PM10** from station **ES0118A (Casa de Campo, Madrid)**, year **2023**.

---

### Option A — EEA Discomap web download (recommended)

1. Go to: <https://eeadmz1-downloads-webapp.azurewebsites.net/>
2. Select: *Download by station*
3. Station code: **ES0118A**
4. Pollutant: **PM10 (particulate matter < 10 µm)**
5. Averaging time: **Day**
6. Year range: **2023–2023**
7. Download CSV
8. Rename/reformat to `data/raw/pm10_daily.csv` with two columns: `date` and `y`

---

### Option B — EEA API (bulk download)

```bash
# Download raw station data (aggregated daily)
curl -L "https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm?\
CountryCode=ES&CityName=&Pollutant=PM10&Year_from=2023&Year_to=2023\
&Station=ES0118A&Samplingpoint=&AveragingTime=day\
&Output=TEXT&TimeCoverage=Year" \
  -o data/raw/pm10_raw_eea.csv
```

Then reformat to the required schema:

```python
import pandas as pd
df = pd.read_csv("data/raw/pm10_raw_eea.csv")
# EEA columns: DatetimeBegin, Concentration, ...
out = df[["DatetimeBegin", "Concentration"]].rename(
    columns={"DatetimeBegin": "date", "Concentration": "y"}
)
out["date"] = pd.to_datetime(out["date"]).dt.date
out.to_csv("data/raw/pm10_daily.csv", index=False)
```

---

### Expected format of `data/raw/pm10_daily.csv`

```
date,y
2023-01-01,18.4
2023-01-02,22.1
...
2023-12-31,15.3
```

| Column | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Calendar date |
| `y` | float | Daily mean PM10 concentration (µg/m³) |

Missing values (`NaN`, empty) are allowed — they will be forward-filled during processing.

---

### Station metadata

| Attribute | Value |
|---|---|
| Station code | ES0118A |
| Station name | Casa de Campo |
| City | Madrid, Spain |
| Latitude | 40.419° N |
| Longitude | 3.718° W |
| Altitude | 669 m |
| Network | Red de Vigilancia de la Calidad del Aire de Madrid |

---

### Data integrity check

After downloading and running `scripts/02_build_processed_datasets.py`, verify:

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/processed/pm10_daily.parquet")
print(f"Rows:    {len(df)}")
print(f"Range:   {df['date'].min()} → {df['date'].max()}")
print(f"Missing: {df['y'].isna().sum()}")
print(f"Mean:    {df['y'].mean():.1f} µg/m³")
EOF
```

Expected: 365 rows, 2023-01-01 → 2023-12-31, ≤ 10 missing values.

Record the checksum after your first successful run:

```bash
sha256sum data/processed/pm10_daily.parquet
# Paste result here:
# SHA256: <hash>
```
