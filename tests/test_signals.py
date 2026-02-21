"""
Tests for analytics.signals module.
"""

import pandas as pd
import pytest

from analytics.signals import (
    SignalAlert,
    alerts_to_dataframe,
    detect_level_signals,
    detect_yoy_signals,
    run_all_detectors,
)
from data.sample_data import generate_sample_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_df():
    """Tiny hand-crafted DataFrame with known stress signals."""
    return pd.DataFrame(
        [
            # High inflation
            {"year": 2010, "country": "TestLand", "indicator_code": "FP.CPI.TOTL.ZG",
             "indicator_name": "Inflation (CPI, % annual)", "value": 25.0},
            {"year": 2011, "country": "TestLand", "indicator_code": "FP.CPI.TOTL.ZG",
             "indicator_name": "Inflation (CPI, % annual)", "value": 15.0},
            # GDP contraction
            {"year": 2012, "country": "TestLand", "indicator_code": "NY.GDP.MKTP.KD.ZG",
             "indicator_name": "GDP Growth (% annual)", "value": -2.5},
            # Normal unemployment
            {"year": 2012, "country": "TestLand", "indicator_code": "SL.UEM.TOTL.ZS",
             "indicator_name": "Unemployment Rate (%)", "value": 5.0},
        ]
    )


@pytest.fixture
def sample_df():
    return generate_sample_data(start_year=2000, end_year=2010)


# ---------------------------------------------------------------------------
# detect_level_signals
# ---------------------------------------------------------------------------

class TestDetectLevelSignals:
    def test_flags_high_inflation(self, minimal_df):
        alerts = detect_level_signals(minimal_df)
        types = [a.signal_type for a in alerts]
        assert any("Inflation" in t for t in types)

    def test_flags_gdp_contraction(self, minimal_df):
        alerts = detect_level_signals(minimal_df)
        types = [a.signal_type for a in alerts]
        assert any("Contraction" in t for t in types)

    def test_no_high_unemployment_alert_for_normal_value(self, minimal_df):
        alerts = detect_level_signals(minimal_df)
        unemp_alerts = [
            a for a in alerts
            if "Unemployment" in a.signal_type and a.country == "TestLand"
        ]
        assert len(unemp_alerts) == 0

    def test_returns_list_of_signal_alerts(self, minimal_df):
        alerts = detect_level_signals(minimal_df)
        assert isinstance(alerts, list)
        assert all(isinstance(a, SignalAlert) for a in alerts)


# ---------------------------------------------------------------------------
# detect_yoy_signals
# ---------------------------------------------------------------------------

class TestDetectYoySignals:
    def test_flags_inflation_spike(self):
        df = pd.DataFrame(
            [
                {"year": 2009, "country": "Alpha", "indicator_code": "FP.CPI.TOTL.ZG",
                 "indicator_name": "Inflation (CPI, % annual)", "value": 3.0},
                {"year": 2010, "country": "Alpha", "indicator_code": "FP.CPI.TOTL.ZG",
                 "indicator_name": "Inflation (CPI, % annual)", "value": 9.5},
            ]
        )
        alerts = detect_yoy_signals(df)
        assert any("Spike" in a.signal_type for a in alerts)

    def test_no_alert_for_stable_series(self):
        df = pd.DataFrame(
            [
                {"year": y, "country": "Stable", "indicator_code": "FP.CPI.TOTL.ZG",
                 "indicator_name": "Inflation (CPI, % annual)", "value": 2.0 + y * 0.01}
                for y in range(2000, 2010)
            ]
        )
        alerts = detect_yoy_signals(df)
        assert len(alerts) == 0

    def test_insufficient_data_returns_empty(self):
        df = pd.DataFrame(
            [
                {"year": 2010, "country": "OnlyOne", "indicator_code": "FP.CPI.TOTL.ZG",
                 "indicator_name": "Inflation", "value": 15.0},
            ]
        )
        alerts = detect_yoy_signals(df)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# run_all_detectors
# ---------------------------------------------------------------------------

class TestRunAllDetectors:
    def test_deduplicates_alerts(self, minimal_df):
        alerts1 = run_all_detectors(minimal_df)
        alerts2 = run_all_detectors(minimal_df)
        keys1 = {(a.country, a.indicator, a.year, a.signal_type) for a in alerts1}
        keys2 = {(a.country, a.indicator, a.year, a.signal_type) for a in alerts2}
        assert keys1 == keys2
        assert len(alerts1) == len(keys1)

    def test_sample_data_produces_some_alerts(self, sample_df):
        alerts = run_all_detectors(sample_df)
        assert len(alerts) > 0


# ---------------------------------------------------------------------------
# alerts_to_dataframe
# ---------------------------------------------------------------------------

class TestAlertsToDataframe:
    def test_empty_input_returns_empty_df(self):
        df = alerts_to_dataframe([])
        assert df.empty
        assert set(df.columns) == {
            "country", "indicator", "year", "value",
            "signal_type", "severity", "description",
        }

    def test_columns_present(self, minimal_df):
        alerts = run_all_detectors(minimal_df)
        df = alerts_to_dataframe(alerts)
        expected_cols = {"country", "indicator", "year", "value",
                         "signal_type", "severity", "description"}
        assert expected_cols.issubset(set(df.columns))

    def test_length_matches(self, minimal_df):
        alerts = run_all_detectors(minimal_df)
        df = alerts_to_dataframe(alerts)
        assert len(df) == len(alerts)
