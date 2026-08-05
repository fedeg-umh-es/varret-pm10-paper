import pandas as pd
import pytest

from src.evaluation.p4_exceedance.schema_adapter import adapt_schema, SchemaAdapterError


def _daily_df():
    return pd.DataFrame(
        {
            "origin_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "date": ["2020-01-03", "2020-01-04", "2020-01-05"],
            "horizon": [2, 2, 2],
        }
    )


def _hourly_df():
    return pd.DataFrame(
        {
            "origin_time": ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"],
            "target_time": ["2020-01-01 03:00", "2020-01-01 04:00", "2020-01-01 05:00"],
            "horizon": [3, 3, 3],
        }
    )


class TestScenario1DateToTargetDate:
    def test_date_column_adapted_to_target_date(self):
        df = _daily_df()
        adapted, report = adapt_schema(df)
        assert "target_date" in adapted.columns
        assert (adapted["target_date"] == pd.to_datetime(df["date"])).all()
        assert report.target_date_source == "date"
        assert any("interpreted as the forecast target time" in n for n in report.notes)

    def test_target_date_used_directly_when_present(self):
        df = _daily_df().rename(columns={"date": "target_date"})
        adapted, report = adapt_schema(df)
        assert report.target_date_source == "target_date"
        assert report.notes == []


class TestScenario2TargetNotAfterOrigin:
    def test_rejects_target_date_equal_to_origin(self):
        df = pd.DataFrame({"origin_date": ["2020-01-02"], "date": ["2020-01-02"], "horizon": [0]})
        with pytest.raises(SchemaAdapterError):
            adapt_schema(df)

    def test_rejects_target_date_before_origin(self):
        df = pd.DataFrame({"origin_date": ["2020-01-05"], "date": ["2020-01-01"], "horizon": [1]})
        with pytest.raises(SchemaAdapterError, match="target_date must be strictly after"):
            adapt_schema(df)

    def test_rejects_non_positive_horizon(self):
        df = pd.DataFrame({"origin_date": ["2020-01-01"], "date": ["2020-01-05"], "horizon": [0]})
        with pytest.raises(SchemaAdapterError, match="strictly positive"):
            adapt_schema(df)


class TestScenario3DailyCoherence:
    def test_daily_resolution_inferred_and_validated(self):
        df = _daily_df()
        adapted, report = adapt_schema(df)
        assert report.resolution == "daily"
        assert (adapted["target_date"] - adapted["origin_date"] == pd.Timedelta(days=2)).all()

    def test_daily_incoherence_rejected_when_declared(self):
        df = _daily_df()
        df.loc[0, "date"] = "2020-01-30"  # breaks origin + 2 days
        with pytest.raises(SchemaAdapterError):
            adapt_schema(df, resolution="daily")


class TestScenario4HourlyCoherence:
    def test_hourly_resolution_inferred_and_validated(self):
        df = _hourly_df()
        adapted, report = adapt_schema(df)
        assert report.resolution == "hourly"
        assert (adapted["target_date"] - adapted["origin_date"] == pd.Timedelta(hours=3)).all()

    def test_hourly_declared_and_coherent(self):
        df = _hourly_df()
        adapted, report = adapt_schema(df, resolution="hourly")
        assert report.resolution == "hourly"


class TestScenario5AmbiguousResolution:
    def test_neither_daily_nor_hourly_hypothesis_holds(self):
        df = pd.DataFrame(
            {
                "origin_date": ["2020-01-01 00:00"],
                "date": ["2020-01-01 07:30"],
                "horizon": [3],
            }
        )
        with pytest.raises(SchemaAdapterError, match="resolution could not be determined"):
            adapt_schema(df)

    def test_declaring_resolution_explicitly_bypasses_inference(self):
        df = pd.DataFrame(
            {
                "origin_date": ["2020-01-01 00:00"],
                "date": ["2020-01-01 07:30"],
                "horizon": [3],
            }
        )
        with pytest.raises(SchemaAdapterError, match="not coherent"):
            adapt_schema(df, resolution="hourly")

    def test_invalid_resolution_argument_rejected(self):
        df = _daily_df()
        with pytest.raises(SchemaAdapterError, match="Unsupported resolution"):
            adapt_schema(df, resolution="weekly")


class TestMissingColumns:
    def test_missing_origin_column_raises(self):
        df = pd.DataFrame({"date": ["2020-01-02"], "horizon": [1]})
        with pytest.raises(SchemaAdapterError, match="origin timestamp"):
            adapt_schema(df)

    def test_missing_target_column_raises(self):
        df = pd.DataFrame({"origin_date": ["2020-01-01"], "horizon": [1]})
        with pytest.raises(SchemaAdapterError, match="target timestamp"):
            adapt_schema(df)

    def test_missing_horizon_column_raises(self):
        df = pd.DataFrame({"origin_date": ["2020-01-01"], "date": ["2020-01-02"]})
        with pytest.raises(SchemaAdapterError, match="horizon"):
            adapt_schema(df)

    def test_source_csv_not_mutated(self):
        df = _daily_df()
        original = df.copy(deep=True)
        adapt_schema(df)
        pd.testing.assert_frame_equal(df, original)
