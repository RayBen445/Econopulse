"""
Economic stress signal detection for EconoPulse.

Each detector returns a list of SignalAlert dataclasses.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SignalAlert:
    """Represents a detected economic stress signal."""

    country: str
    indicator: str
    year: int
    value: float
    signal_type: str
    severity: str          # "low", "medium", "high"
    description: str


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "inflation_high": {"indicator": "FP.CPI.TOTL.ZG", "threshold": 10.0, "severity": "high",
                       "label": "High Inflation"},
    "inflation_very_high": {"indicator": "FP.CPI.TOTL.ZG", "threshold": 20.0, "severity": "high",
                            "label": "Very High / Hyperinflation Risk"},
    "inflation_spike": {"indicator": "FP.CPI.TOTL.ZG", "yoy_change": 5.0, "severity": "medium",
                        "label": "Inflation Spike (YoY)"},
    "gdp_contraction": {"indicator": "NY.GDP.MKTP.KD.ZG", "threshold": 0.0, "severity": "medium",
                        "label": "GDP Contraction / Recession"},
    "gdp_sharp_drop": {"indicator": "NY.GDP.MKTP.KD.ZG", "yoy_change": -3.0, "severity": "high",
                       "label": "Sharp GDP Drop (YoY)"},
    "high_unemployment": {"indicator": "SL.UEM.TOTL.ZS", "threshold": 15.0, "severity": "high",
                          "label": "High Unemployment"},
    "unemployment_surge": {"indicator": "SL.UEM.TOTL.ZS", "yoy_change": 3.0, "severity": "medium",
                           "label": "Unemployment Surge (YoY)"},
}


def _get_series(df: pd.DataFrame, country: str, indicator_code: str) -> pd.DataFrame:
    mask = (df["country"] == country) & (df["indicator_code"] == indicator_code)
    return df[mask].sort_values("year").copy()


def detect_level_signals(df: pd.DataFrame) -> list[SignalAlert]:
    """Flag observations that breach absolute threshold levels."""
    alerts: list[SignalAlert] = []

    level_checks = [
        ("FP.CPI.TOTL.ZG", 10.0, "high", "High Inflation (CPI > 10%)"),
        ("FP.CPI.TOTL.ZG", 20.0, "high", "Very High Inflation / Hyperinflation Risk (CPI > 20%)"),
        ("NY.GDP.MKTP.KD.ZG", 0.0, "medium", "GDP Contraction (growth < 0%)"),
        ("SL.UEM.TOTL.ZS", 15.0, "high", "High Unemployment (> 15%)"),
    ]

    for code, threshold, severity, label in level_checks:
        subset = df[df["indicator_code"] == code]
        for _, row in subset.iterrows():
            triggered = (
                (code == "NY.GDP.MKTP.KD.ZG" and row["value"] < threshold)
                or (code != "NY.GDP.MKTP.KD.ZG" and row["value"] > threshold)
            )
            if triggered:
                alerts.append(
                    SignalAlert(
                        country=row["country"],
                        indicator=row["indicator_name"],
                        year=int(row["year"]),
                        value=round(float(row["value"]), 2),
                        signal_type=label,
                        severity=severity,
                        description=(
                            f"{row['country']} recorded {row['indicator_name']} "
                            f"of {row['value']:.2f} in {int(row['year'])}."
                        ),
                    )
                )
    return alerts


def detect_yoy_signals(df: pd.DataFrame) -> list[SignalAlert]:
    """Flag year-over-year changes that exceed threshold magnitudes."""
    alerts: list[SignalAlert] = []

    yoy_checks = [
        ("FP.CPI.TOTL.ZG", 5.0, "medium", "Inflation Spike (YoY increase > 5 pp)"),
        ("NY.GDP.MKTP.KD.ZG", -3.0, "high", "Sharp GDP Drop (YoY decline > 3 pp)"),
        ("SL.UEM.TOTL.ZS", 3.0, "medium", "Unemployment Surge (YoY increase > 3 pp)"),
    ]

    countries = df["country"].unique()
    for country in countries:
        for code, change_threshold, severity, label in yoy_checks:
            series = _get_series(df, country, code)
            if len(series) < 2:
                continue
            series["yoy_change"] = series["value"].diff()
            triggered_rows = (
                series[series["yoy_change"] > change_threshold]
                if change_threshold > 0
                else series[series["yoy_change"] < change_threshold]
            )
            for _, row in triggered_rows.iterrows():
                if pd.isna(row["yoy_change"]):
                    continue
                alerts.append(
                    SignalAlert(
                        country=country,
                        indicator=row["indicator_name"],
                        year=int(row["year"]),
                        value=round(float(row["value"]), 2),
                        signal_type=label,
                        severity=severity,
                        description=(
                            f"{country}: {row['indicator_name']} changed by "
                            f"{row['yoy_change']:+.2f} pp in {int(row['year'])} "
                            f"(value: {row['value']:.2f})."
                        ),
                    )
                )
    return alerts


def run_all_detectors(df: pd.DataFrame) -> list[SignalAlert]:
    """Run all signal detectors and return a deduplicated list of alerts."""
    alerts = detect_level_signals(df) + detect_yoy_signals(df)
    # Deduplicate on (country, indicator, year, signal_type)
    seen: set[tuple] = set()
    unique: list[SignalAlert] = []
    for a in alerts:
        key = (a.country, a.indicator, a.year, a.signal_type)
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def alerts_to_dataframe(alerts: list[SignalAlert]) -> pd.DataFrame:
    """Convert a list of SignalAlert objects to a tidy DataFrame."""
    if not alerts:
        return pd.DataFrame(
            columns=["country", "indicator", "year", "value",
                     "signal_type", "severity", "description"]
        )
    return pd.DataFrame(
        [
            {
                "country": a.country,
                "indicator": a.indicator,
                "year": a.year,
                "value": a.value,
                "signal_type": a.signal_type,
                "severity": a.severity,
                "description": a.description,
            }
            for a in alerts
        ]
    )
