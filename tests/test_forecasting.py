"""
Tests for analytics.forecasting module.
"""

import pandas as pd
import pytest

from analytics.forecasting import forecast_indicator
from data.sample_data import generate_sample_data


@pytest.fixture
def full_df():
    return generate_sample_data(start_year=2000, end_year=2023)


class TestForecastIndicator:
    def test_returns_dataframe(self, full_df):
        result = forecast_indicator(full_df, "United States", "FP.CPI.TOTL.ZG", n_periods=5)
        assert isinstance(result, pd.DataFrame)

    def test_forecast_length(self, full_df):
        n = 5
        result = forecast_indicator(full_df, "United States", "FP.CPI.TOTL.ZG", n_periods=n)
        assert len(result[result["type"] == "forecast"]) == n

    def test_historical_years_preserved(self, full_df):
        result = forecast_indicator(full_df, "Germany", "NY.GDP.MKTP.KD.ZG", n_periods=3)
        hist = result[result["type"] == "historical"]
        assert len(hist) == len(
            full_df[
                (full_df["country"] == "Germany")
                & (full_df["indicator_code"] == "NY.GDP.MKTP.KD.ZG")
            ]
        )

    def test_forecast_years_are_future(self, full_df):
        result = forecast_indicator(full_df, "India", "SL.UEM.TOTL.ZS", n_periods=4)
        hist_max = result[result["type"] == "historical"]["year"].max()
        fc_min = result[result["type"] == "forecast"]["year"].min()
        assert fc_min == hist_max + 1

    def test_returns_none_for_insufficient_data(self):
        tiny_df = pd.DataFrame(
            [
                {"year": 2020, "country": "Tiny", "indicator_code": "FP.CPI.TOTL.ZG",
                 "indicator_name": "Inflation", "value": 3.0},
                {"year": 2021, "country": "Tiny", "indicator_code": "FP.CPI.TOTL.ZG",
                 "indicator_name": "Inflation", "value": 4.0},
            ]
        )
        result = forecast_indicator(tiny_df, "Tiny", "FP.CPI.TOTL.ZG", n_periods=3)
        assert result is None

    def test_linear_method(self, full_df):
        result = forecast_indicator(
            full_df, "Brazil", "FP.CPI.TOTL.ZG", n_periods=3, method="linear"
        )
        assert result is not None
        assert len(result[result["type"] == "forecast"]) == 3
