from __future__ import annotations

from pathlib import Path

from src.data.loaders import load_dataset


def test_load_dataset_returns_expected_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "pm10_example.csv"
    csv_path.write_text(
        "date,PM10\n2024-01-01,10.0\n2024-01-02,12.5\n2024-01-03,11.0\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "configs" / "datasets" / "pm10_example.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "name: pm10_example",
                "target_column: PM10",
                "datetime_column: date",
                "frequency: daily",
                "horizons: [1, 3, 7]",
                "features:",
                "  endogenous:",
                "    - PM10",
                "  exogenous: []",
                "paths:",
                "  raw: data/raw/pm10_example.csv",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(config_path)

    assert dataset["name"] == "pm10_example"
    assert dataset["target_column"] == "PM10"
    assert dataset["datetime_column"] == "date"
    assert dataset["frequency"] == "daily"
    assert dataset["horizons"] == [1, 3, 7]
    assert list(dataset["data"].columns) == ["date", "PM10"]
    assert list(dataset["X"].columns) == ["PM10"]
    assert dataset["y"].tolist() == [10.0, 12.5, 11.0]
    assert dataset["time_index"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert dataset["metadata"]["features"] == {
        "endogenous": ["PM10"],
        "exogenous": [],
    }
