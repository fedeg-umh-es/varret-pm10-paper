from pathlib import Path

import pandas as pd

from src.data.madrid_hourly import parse_casa_de_campo_pm10


def test_parser_preserves_invalid_values_as_missing(tmp_path: Path) -> None:
    import zipfile

    columns = [
        "PROVINCIA", "MUNICIPIO", "ESTACION", "MAGNITUD", "PUNTO_MUESTREO",
        "ANO", "MES", "DIA",
    ]
    values: list[object] = [28, 79, 24, 10, "28079024_10_47", 2023, 1, 1]
    for hour in range(1, 25):
        columns.extend([f"H{hour:02d}", f"V{hour:02d}"])
        values.extend([hour, "N" if hour == 2 else "V"])
    csv = pd.DataFrame([values], columns=columns).to_csv(index=False, sep=";")
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ene_mo23.csv", csv)

    parsed = parse_casa_de_campo_pm10(archive)

    assert len(parsed) == 8760
    assert parsed.loc[0, "pm10"] == 1
    assert pd.isna(parsed.loc[1, "pm10"])
    assert not bool(parsed.loc[1, "is_valid"])
    assert pd.isna(parsed.loc[24, "pm10"])
