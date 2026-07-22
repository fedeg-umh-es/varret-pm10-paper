"""Acquire and parse validated hourly PM10 observations from Madrid Open Data."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


MADRID_2023_URL = (
    "https://datos.madrid.es/dataset/201200-0-calidad-aire-horario/"
    "resource/201200-3-calidad-aire-horario-zip/download/"
    "201200-3-calidad-aire-horario-zip.zip"
)
MADRID_2023_SHA256 = "b3ee481e0a787239dd07b33e93b2da97e31e6b5123d3c659f49e14549fb62b2e"
CASA_DE_CAMPO_STATION = 24
PM10_MAGNITUDE = 10


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_2023_archive(destination: Path) -> Path:
    """Download the immutable 2023 annual archive and verify its checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MADRID_2023_URL, timeout=120) as response:
        payload = response.read()
    digest = sha256_bytes(payload)
    if digest != MADRID_2023_SHA256:
        raise ValueError(
            f"Unexpected Madrid 2023 archive checksum: {digest}; "
            f"expected {MADRID_2023_SHA256}"
        )
    destination.write_bytes(payload)
    return destination


def parse_casa_de_campo_pm10(archive: Path | bytes) -> pd.DataFrame:
    """Return a regular 2023 hourly grid; invalid measurements remain missing.

    Madrid files encode one row per day and pairs H01/V01 ... H24/V24.
    Only observations marked ``V`` are accepted. Missing days and invalid hours
    are represented as NaN, never filled from future observations.
    """
    source = io.BytesIO(archive) if isinstance(archive, bytes) else archive
    daily = []
    with zipfile.ZipFile(source) as bundle:
        for member in sorted(bundle.namelist()):
            if member.lower().endswith(".csv"):
                daily.append(pd.read_csv(bundle.open(member), sep=";"))
    if not daily:
        raise ValueError("No CSV members found in Madrid archive")

    data = pd.concat(daily, ignore_index=True)
    data = data.loc[
        (data["ESTACION"] == CASA_DE_CAMPO_STATION)
        & (data["MAGNITUD"] == PM10_MAGNITUDE)
    ]
    if data.empty:
        raise ValueError("Casa de Campo PM10 records not found")

    rows: list[dict[str, object]] = []
    for record in data.itertuples(index=False):
        day = pd.Timestamp(int(record.ANO), int(record.MES), int(record.DIA))
        for hour in range(1, 25):
            valid = getattr(record, f"V{hour:02d}") == "V"
            rows.append(
                {
                    "timestamp": day + pd.Timedelta(hours=hour - 1),
                    "pm10": float(getattr(record, f"H{hour:02d}")) if valid else pd.NA,
                    "is_valid": valid,
                }
            )

    observed = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if observed.index.duplicated().any():
        raise ValueError("Duplicate hourly records in Madrid archive")
    grid = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", name="timestamp")
    observed = observed.reindex(grid)
    observed["pm10"] = pd.to_numeric(observed["pm10"], errors="coerce")
    observed["is_valid"] = observed["is_valid"].eq(True)
    return observed.reset_index()
