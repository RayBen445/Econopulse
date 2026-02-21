"""
Tests for analytics.premium module – no mocks, all real computations.
"""

import pandas as pd
import pytest

from analytics.premium import (
    cluster_countries,
    compute_economic_health_score,
    compute_moving_average,
    compute_risk_scores,
    compute_volatility,
    compute_yoy_heatmap,
    detect_recessions,
    multi_country_forecast,
    rank_countries,
    run_custom_threshold_alerts,
)
from analytics.signals import alerts_to_dataframe, run_all_detectors
from data.sample_data import generate_sample_data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_df():
    return generate_sample_data(start_year=2000, end_year=2023)


@pytest.fixture(scope="module")
def alerts_df(full_df):
    return alerts_to_dataframe(run_all_detectors(full_df))


# ---------------------------------------------------------------------------
# 1. compute_economic_health_score
# ---------------------------------------------------------------------------

class TestComputeEconomicHealthScore:
    def test_returns_dataframe(self, full_df):
        result = compute_economic_health_score(full_df)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, full_df):
        result = compute_economic_health_score(full_df)
        assert {"year", "country", "health_score"}.issubset(result.columns)

    def test_scores_in_range(self, full_df):
        result = compute_economic_health_score(full_df)
        assert (result["health_score"] >= 0).all()
        assert (result["health_score"] <= 100).all()

    def test_covers_all_countries(self, full_df):
        result = compute_economic_health_score(full_df)
        assert set(full_df["country"].unique()) == set(result["country"].unique())

    def test_empty_df_returns_empty(self):
        result = compute_economic_health_score(pd.DataFrame(
            columns=["year", "country", "indicator_code", "value"]
        ))
        assert result.empty


# ---------------------------------------------------------------------------
# 2. multi_country_forecast
# ---------------------------------------------------------------------------

class TestMultiCountryForecast:
    def test_returns_dataframe(self, full_df):
        result = multi_country_forecast(
            full_df, ["United States", "Germany"], "FP.CPI.TOTL.ZG", n_periods=3
        )
        assert isinstance(result, pd.DataFrame)

    def test_contains_forecast_rows_for_each_country(self, full_df):
        countries = ["United States", "Germany"]
        result = multi_country_forecast(full_df, countries, "FP.CPI.TOTL.ZG", n_periods=3)
        fc = result[result["type"] == "forecast"]
        assert set(fc["country"].unique()) == set(countries)

    def test_forecast_horizon_per_country(self, full_df):
        n = 4
        result = multi_country_forecast(
            full_df, ["Brazil"], "NY.GDP.MKTP.KD.ZG", n_periods=n
        )
        fc = result[(result["type"] == "forecast") & (result["country"] == "Brazil")]
        assert len(fc) == n

    def test_empty_result_for_unknown_country(self, full_df):
        result = multi_country_forecast(
            full_df, ["NoSuchCountry"], "FP.CPI.TOTL.ZG", n_periods=3
        )
        assert result.empty


# ---------------------------------------------------------------------------
# 3. compute_yoy_heatmap
# ---------------------------------------------------------------------------

class TestComputeYoyHeatmap:
    def test_returns_dataframe(self, full_df):
        result = compute_yoy_heatmap(full_df, "FP.CPI.TOTL.ZG")
        assert isinstance(result, pd.DataFrame)

    def test_index_is_countries(self, full_df):
        result = compute_yoy_heatmap(full_df, "FP.CPI.TOTL.ZG")
        assert set(result.index) == set(full_df["country"].unique())

    def test_country_filter_applied(self, full_df):
        countries = ["United States", "Germany"]
        result = compute_yoy_heatmap(full_df, "FP.CPI.TOTL.ZG", countries=countries)
        assert set(result.index) == set(countries)

    def test_columns_are_years(self, full_df):
        result = compute_yoy_heatmap(full_df, "FP.CPI.TOTL.ZG")
        assert all(isinstance(c, int) for c in result.columns)


# ---------------------------------------------------------------------------
# 4. compute_risk_scores
# ---------------------------------------------------------------------------

class TestComputeRiskScores:
    def test_returns_dataframe(self, alerts_df):
        result = compute_risk_scores(alerts_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_values_non_negative(self, alerts_df):
        result = compute_risk_scores(alerts_df)
        assert (result.values >= 0).all()

    def test_empty_alerts_returns_empty(self):
        result = compute_risk_scores(pd.DataFrame(
            columns=["country", "year", "severity", "signal_type", "indicator", "value", "description"]
        ))
        assert result.empty

    def test_country_filter(self, alerts_df):
        countries = ["United States"]
        result = compute_risk_scores(alerts_df, countries=countries)
        assert set(result.index).issubset(set(countries))


# ---------------------------------------------------------------------------
# 5. detect_recessions
# ---------------------------------------------------------------------------

class TestDetectRecessions:
    def test_returns_list(self, full_df):
        result = detect_recessions(full_df, "United States")
        assert isinstance(result, list)

    def test_recession_dict_keys(self, full_df):
        # Use a hand-crafted df with a known recession
        df = pd.DataFrame([
            {"year": 2008, "country": "X", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": -2.0},
            {"year": 2009, "country": "X", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": -1.5},
            {"year": 2010, "country": "X", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": 2.5},
        ])
        result = detect_recessions(df, "X", min_consecutive=2)
        assert len(result) == 1
        assert result[0]["start_year"] == 2008
        assert result[0]["end_year"] == 2009
        assert {"country", "start_year", "end_year", "depth"} == set(result[0].keys())

    def test_single_negative_year_not_flagged(self):
        df = pd.DataFrame([
            {"year": 2008, "country": "Y", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": -1.0},
            {"year": 2009, "country": "Y", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": 1.5},
        ])
        result = detect_recessions(df, "Y", min_consecutive=2)
        assert result == []

    def test_no_recession_for_positive_growth(self):
        df = pd.DataFrame([
            {"year": y, "country": "Z", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": 2.0 + y * 0.01}
            for y in range(2000, 2010)
        ])
        result = detect_recessions(df, "Z")
        assert result == []


# ---------------------------------------------------------------------------
# 6. rank_countries
# ---------------------------------------------------------------------------

class TestRankCountries:
    def test_returns_dataframe(self, full_df):
        result = rank_countries(full_df, "FP.CPI.TOTL.ZG")
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, full_df):
        result = rank_countries(full_df, "FP.CPI.TOTL.ZG")
        assert {"country", "year", "value", "rank", "percentile"}.issubset(result.columns)

    def test_rank_starts_at_one(self, full_df):
        result = rank_countries(full_df, "FP.CPI.TOTL.ZG")
        assert result["rank"].min() == 1

    def test_percentile_range(self, full_df):
        result = rank_countries(full_df, "FP.CPI.TOTL.ZG")
        assert (result["percentile"] >= 0).all()
        assert (result["percentile"] <= 100).all()

    def test_custom_year(self, full_df):
        result = rank_countries(full_df, "FP.CPI.TOTL.ZG", year=2010)
        assert (result["year"] == 2010).all()


# ---------------------------------------------------------------------------
# 7. compute_moving_average
# ---------------------------------------------------------------------------

class TestComputeMovingAverage:
    def test_returns_dataframe(self, full_df):
        result = compute_moving_average(full_df, "United States", "FP.CPI.TOTL.ZG")
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, full_df):
        result = compute_moving_average(full_df, "Germany", "NY.GDP.MKTP.KD.ZG")
        assert {"year", "value", "moving_avg"}.issubset(result.columns)

    def test_moving_avg_not_null(self, full_df):
        result = compute_moving_average(full_df, "India", "SL.UEM.TOTL.ZS", window=3)
        assert not result["moving_avg"].isnull().any()

    def test_window_smoothing(self, full_df):
        result = compute_moving_average(full_df, "Brazil", "FP.CPI.TOTL.ZG", window=5)
        # Moving average variance should be less than or equal to raw variance
        assert result["moving_avg"].std() <= result["value"].std() + 1e-3


# ---------------------------------------------------------------------------
# 8. run_custom_threshold_alerts
# ---------------------------------------------------------------------------

class TestRunCustomThresholdAlerts:
    def test_returns_dataframe(self, full_df):
        result = run_custom_threshold_alerts(
            full_df, {"FP.CPI.TOTL.ZG": 5.0}
        )
        assert isinstance(result, pd.DataFrame)

    def test_schema_matches_standard_alerts(self, full_df):
        result = run_custom_threshold_alerts(
            full_df, {"FP.CPI.TOTL.ZG": 5.0}
        )
        expected_cols = {"country", "year", "value", "signal_type", "severity", "description", "indicator"}
        assert expected_cols.issubset(set(result.columns))

    def test_threshold_respected(self, full_df):
        threshold = 5.0
        code = "FP.CPI.TOTL.ZG"
        result = run_custom_threshold_alerts(full_df, {code: threshold})
        if not result.empty:
            actual_values = (
                full_df[full_df["indicator_code"] == code]
                .set_index(["country", "year"])["value"]
            )
            for _, row in result.iterrows():
                assert actual_values.loc[(row["country"], row["year"])] > threshold

    def test_gdp_below_threshold(self):
        df = pd.DataFrame([
            {"year": 2009, "country": "TestLand", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": -2.5},
            {"year": 2010, "country": "TestLand", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth", "value": 2.0},
        ])
        result = run_custom_threshold_alerts(df, {"NY.GDP.MKTP.KD.ZG": 0.0})
        assert len(result) == 1
        assert result.iloc[0]["year"] == 2009

    def test_empty_thresholds_returns_empty_df(self, full_df):
        result = run_custom_threshold_alerts(full_df, {})
        assert result.empty


# ---------------------------------------------------------------------------
# 9. compute_volatility
# ---------------------------------------------------------------------------

class TestComputeVolatility:
    def test_returns_dataframe(self, full_df):
        result = compute_volatility(full_df, "FP.CPI.TOTL.ZG")
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, full_df):
        result = compute_volatility(full_df, "FP.CPI.TOTL.ZG")
        assert {"year", "country", "value", "rolling_std"}.issubset(result.columns)

    def test_rolling_std_non_negative(self, full_df):
        result = compute_volatility(full_df, "NY.GDP.MKTP.KD.ZG")
        non_null = result["rolling_std"].dropna()
        assert (non_null >= 0).all()

    def test_country_filter(self, full_df):
        countries = ["United States", "Germany"]
        result = compute_volatility(full_df, "FP.CPI.TOTL.ZG", countries=countries)
        assert set(result["country"].unique()) == set(countries)


# ---------------------------------------------------------------------------
# 10. cluster_countries
# ---------------------------------------------------------------------------

class TestClusterCountries:
    def test_returns_dataframe(self, full_df):
        result = cluster_countries(full_df, n_clusters=3)
        assert isinstance(result, pd.DataFrame)

    def test_cluster_column_present(self, full_df):
        result = cluster_countries(full_df, n_clusters=3)
        assert "cluster" in result.columns

    def test_number_of_clusters(self, full_df):
        n = 3
        result = cluster_countries(full_df, n_clusters=n)
        assert result["cluster"].nunique() <= n

    def test_all_countries_present(self, full_df):
        result = cluster_countries(full_df, n_clusters=2)
        assert set(result["country"].unique()) == set(full_df["country"].unique())

    def test_custom_year(self, full_df):
        result = cluster_countries(full_df, year=2010, n_clusters=2)
        assert (result["year"] == 2010).all()
