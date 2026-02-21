"""
Sample data generator for EconoPulse.
Produces realistic-looking time-series economic data for offline / demo use.
"""

import numpy as np
import pandas as pd

# Countries available in the sample dataset
SAMPLE_COUNTRIES = ["United States", "Germany", "Brazil", "India", "China", "Nigeria"]

# World Bank indicator codes → display names
INDICATORS = {
    "FP.CPI.TOTL.ZG": "Inflation (CPI, % annual)",
    "NY.GDP.MKTP.KD.ZG": "GDP Growth (% annual)",
    "SL.UEM.TOTL.ZS": "Unemployment Rate (%)",
    "NY.GDP.PCAP.CD": "GDP per Capita (USD)",
}

# Rough "baseline" values per country to make the data plausible
_BASELINES: dict[str, dict[str, float]] = {
    "United States": {
        "FP.CPI.TOTL.ZG": 2.5,
        "NY.GDP.MKTP.KD.ZG": 2.3,
        "SL.UEM.TOTL.ZS": 4.0,
        "NY.GDP.PCAP.CD": 63_000,
    },
    "Germany": {
        "FP.CPI.TOTL.ZG": 2.0,
        "NY.GDP.MKTP.KD.ZG": 1.5,
        "SL.UEM.TOTL.ZS": 5.0,
        "NY.GDP.PCAP.CD": 48_000,
    },
    "Brazil": {
        "FP.CPI.TOTL.ZG": 6.5,
        "NY.GDP.MKTP.KD.ZG": 1.8,
        "SL.UEM.TOTL.ZS": 12.0,
        "NY.GDP.PCAP.CD": 8_500,
    },
    "India": {
        "FP.CPI.TOTL.ZG": 5.5,
        "NY.GDP.MKTP.KD.ZG": 6.5,
        "SL.UEM.TOTL.ZS": 7.0,
        "NY.GDP.PCAP.CD": 2_200,
    },
    "China": {
        "FP.CPI.TOTL.ZG": 2.8,
        "NY.GDP.MKTP.KD.ZG": 6.0,
        "SL.UEM.TOTL.ZS": 5.5,
        "NY.GDP.PCAP.CD": 11_500,
    },
    "Nigeria": {
        "FP.CPI.TOTL.ZG": 16.0,
        "NY.GDP.MKTP.KD.ZG": 2.5,
        "SL.UEM.TOTL.ZS": 23.0,
        "NY.GDP.PCAP.CD": 2_100,
    },
}


def generate_sample_data(
    start_year: int = 2000,
    end_year: int = 2023,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with annual economic indicators for all sample countries.

    Columns: year, country, indicator_code, indicator_name, value
    """
    rng = np.random.default_rng(seed)
    years = list(range(start_year, end_year + 1))
    records = []

    for country in SAMPLE_COUNTRIES:
        baselines = _BASELINES[country]
        for code, name in INDICATORS.items():
            base = baselines[code]
            # Random walk around the baseline
            if code == "NY.GDP.PCAP.CD":
                # Growing series (roughly 2-4 % annual growth)
                values = [base]
                for _ in range(1, len(years)):
                    growth = rng.normal(0.03, 0.02)
                    values.append(max(0.0, values[-1] * (1 + growth)))
            else:
                noise_scale = max(abs(base) * 0.15, 0.5)
                walk = np.cumsum(rng.normal(0, noise_scale * 0.4, len(years)))
                values = (base + walk).tolist()
                # Clamp unemployment & inflation to sensible ranges
                if code in ("SL.UEM.TOTL.ZS", "FP.CPI.TOTL.ZG"):
                    values = [max(0.1, v) for v in values]

            for year, value in zip(years, values):
                records.append(
                    {
                        "year": year,
                        "country": country,
                        "indicator_code": code,
                        "indicator_name": name,
                        "value": round(value, 2),
                    }
                )

    return pd.DataFrame(records)


def generate_fx_sample_data(
    start_year: int = 2000,
    end_year: int = 2023,
    seed: int = 42,
) -> pd.DataFrame:
    """Return simulated annual average FX rates (local currency per USD)."""
    rng = np.random.default_rng(seed + 1)
    years = list(range(start_year, end_year + 1))

    fx_baselines = {
        "United States": 1.00,
        "Germany": 0.85,   # EUR/USD
        "Brazil": 3.50,
        "India": 67.0,
        "China": 6.80,
        "Nigeria": 360.0,
    }

    records = []
    for country, base_rate in fx_baselines.items():
        rate = base_rate
        for year in years:
            drift = rng.normal(0, base_rate * 0.04)
            rate = max(0.01, rate + drift)
            records.append({"year": year, "country": country, "fx_rate": round(rate, 4)})

    return pd.DataFrame(records)
