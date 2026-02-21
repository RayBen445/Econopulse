"""
Data fetcher for EconoPulse.

Supports two sources:
  1. World Bank Open Data API (online)
  2. User-uploaded CSV file
  3. Built-in sample data (offline / demo)
"""

import io
import logging
from typing import Optional

import pandas as pd
import requests

from data.sample_data import (
    INDICATORS,
    SAMPLE_COUNTRIES,
    generate_fx_sample_data,
    generate_sample_data,
)

logger = logging.getLogger(__name__)

WB_BASE_URL = "https://api.worldbank.org/v2"
WB_DATE_RANGE = "2000:2023"
WB_PER_PAGE = 500


# ---------------------------------------------------------------------------
# World Bank API
# ---------------------------------------------------------------------------

def _wb_fetch(indicator: str, country_codes: list[str]) -> pd.DataFrame:
    """Fetch a single indicator for a list of ISO-2 country codes from World Bank."""
    codes = ";".join(country_codes)
    url = (
        f"{WB_BASE_URL}/country/{codes}/indicator/{indicator}"
        f"?format=json&date={WB_DATE_RANGE}&per_page={WB_PER_PAGE}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if len(payload) < 2 or not payload[1]:
            return pd.DataFrame()
        rows = []
        for entry in payload[1]:
            if entry.get("value") is None:
                continue
            rows.append(
                {
                    "year": int(entry["date"]),
                    "country": entry["country"]["value"],
                    "indicator_code": indicator,
                    "indicator_name": INDICATORS.get(indicator, indicator),
                    "value": float(entry["value"]),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("World Bank API request failed for %s: %s", indicator, exc)
        return pd.DataFrame()


# ISO-2 codes for the default countries
_COUNTRY_CODES = {
    "United States": "US",
    "Germany": "DE",
    "Brazil": "BR",
    "India": "IN",
    "China": "CN",
    "Nigeria": "NG",
}


def fetch_world_bank_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all indicators from the World Bank API.

    Returns
    -------
    (indicators_df, fx_df)
        Both may be empty DataFrames if the request fails.
    """
    codes = list(_COUNTRY_CODES.values())
    frames = []
    # Build a lookup: World Bank country name → EconoPulse display name
    wb_name_to_display = {name: display for display, name in _COUNTRY_CODES.items()}
    wb_name_to_display.update({display: display for display in _COUNTRY_CODES})

    for indicator in INDICATORS:
        df = _wb_fetch(indicator, codes)
        if not df.empty:
            df["country"] = df["country"].map(
                lambda c: wb_name_to_display.get(c, c)
            )
            frames.append(df)

    indicators_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # World Bank doesn't offer FX via the standard indicators endpoint with this approach;
    # return empty FX frame for online mode (caller falls back to sample FX).
    return indicators_df, pd.DataFrame()


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------

def load_csv(file_obj) -> pd.DataFrame:
    """Load user-supplied CSV into the standard long-format DataFrame.

    Expected columns (flexible):
        year, country, indicator_code or indicator_name, value
    """
    if hasattr(file_obj, "read"):
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
    else:
        df = pd.read_csv(file_obj)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"year", "country", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "indicator_code" not in df.columns and "indicator_name" not in df.columns:
        raise ValueError("CSV must contain either 'indicator_code' or 'indicator_name' column.")

    if "indicator_code" not in df.columns:
        df["indicator_code"] = df["indicator_name"]
    if "indicator_name" not in df.columns:
        df["indicator_name"] = df["indicator_code"].map(INDICATORS).fillna(df["indicator_code"])

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"])
    return df[["year", "country", "indicator_code", "indicator_name", "value"]]


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def get_data(
    source: str = "sample",
    csv_file=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (indicators_df, fx_df) from the requested source.

    Parameters
    ----------
    source : "sample" | "worldbank" | "csv"
    csv_file : file-like object (required when source=="csv")
    """
    if source == "worldbank":
        ind_df, fx_df = fetch_world_bank_data()
        if ind_df.empty:
            logger.info("World Bank fetch returned no data; falling back to sample data.")
            ind_df = generate_sample_data()
        if fx_df.empty:
            fx_df = generate_fx_sample_data()
        return ind_df, fx_df

    if source == "csv":
        if csv_file is None:
            raise ValueError("csv_file must be provided when source='csv'.")
        ind_df = load_csv(csv_file)
        fx_df = generate_fx_sample_data()
        return ind_df, fx_df

    # Default: sample
    return generate_sample_data(), generate_fx_sample_data()
