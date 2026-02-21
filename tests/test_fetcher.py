"""
Tests for data.fetcher module.
"""

import io

import pandas as pd
import pytest

from data.fetcher import load_csv, get_data
from data.sample_data import generate_sample_data, generate_fx_sample_data, INDICATORS, SAMPLE_COUNTRIES


# ---------------------------------------------------------------------------
# generate_sample_data
# ---------------------------------------------------------------------------

class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_sample_data()
        assert {"year", "country", "indicator_code", "indicator_name", "value"}.issubset(df.columns)

    def test_all_countries_present(self):
        df = generate_sample_data()
        assert set(SAMPLE_COUNTRIES) == set(df["country"].unique())

    def test_all_indicators_present(self):
        df = generate_sample_data()
        assert set(INDICATORS.keys()) == set(df["indicator_code"].unique())

    def test_year_range(self):
        df = generate_sample_data(start_year=2010, end_year=2015)
        assert df["year"].min() == 2010
        assert df["year"].max() == 2015

    def test_no_null_values(self):
        df = generate_sample_data()
        assert not df["value"].isnull().any()


class TestGenerateFxSampleData:
    def test_returns_dataframe(self):
        df = generate_fx_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_fx_sample_data()
        assert {"year", "country", "fx_rate"}.issubset(df.columns)

    def test_positive_rates(self):
        df = generate_fx_sample_data()
        assert (df["fx_rate"] > 0).all()


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def _make_csv(self, content: str) -> io.BytesIO:
        return io.BytesIO(content.encode("utf-8"))

    def test_loads_valid_csv_with_code(self):
        content = "year,country,indicator_code,value\n2020,TestLand,FP.CPI.TOTL.ZG,5.5\n"
        df = load_csv(self._make_csv(content))
        assert len(df) == 1
        assert df.iloc[0]["value"] == 5.5

    def test_loads_valid_csv_with_name(self):
        content = "year,country,indicator_name,value\n2020,TestLand,Inflation,5.5\n"
        df = load_csv(self._make_csv(content))
        assert len(df) == 1
        assert df.iloc[0]["indicator_code"] == "Inflation"

    def test_raises_on_missing_columns(self):
        content = "year,country\n2020,TestLand\n"
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(self._make_csv(content))

    def test_raises_on_missing_indicator_column(self):
        content = "year,country,value\n2020,TestLand,5.5\n"
        with pytest.raises(ValueError, match="indicator"):
            load_csv(self._make_csv(content))

    def test_drops_rows_with_null_values(self):
        content = "year,country,indicator_code,value\n2020,A,FP.CPI.TOTL.ZG,5.5\n2021,B,FP.CPI.TOTL.ZG,\n"
        df = load_csv(self._make_csv(content))
        assert len(df) == 1

    def test_standard_columns_returned(self):
        content = "year,country,indicator_code,value\n2020,X,FP.CPI.TOTL.ZG,3.0\n"
        df = load_csv(self._make_csv(content))
        expected = {"year", "country", "indicator_code", "indicator_name", "value"}
        assert expected == set(df.columns)


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------

class TestGetData:
    def test_sample_source(self):
        ind_df, fx_df = get_data(source="sample")
        assert not ind_df.empty
        assert not fx_df.empty

    def test_csv_source(self):
        content = (
            "year,country,indicator_code,value\n"
            "2020,TestLand,FP.CPI.TOTL.ZG,5.5\n"
            "2021,TestLand,FP.CPI.TOTL.ZG,6.0\n"
        )
        csv_file = io.BytesIO(content.encode("utf-8"))
        ind_df, fx_df = get_data(source="csv", csv_file=csv_file)
        assert len(ind_df) == 2
        assert not fx_df.empty

    def test_csv_source_raises_without_file(self):
        with pytest.raises(ValueError, match="csv_file"):
            get_data(source="csv", csv_file=None)
