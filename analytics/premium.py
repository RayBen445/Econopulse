"""
Premium analytics for EconoPulse.

Ten premium analytical functions:
  1. compute_economic_health_score  – composite 0-100 health index per country/year
  2. multi_country_forecast          – side-by-side forecast for multiple countries
  3. compute_yoy_heatmap             – year-over-year change matrix (pivot)
  4. compute_risk_scores             – signal-based risk score heatmap matrix
  5. detect_recessions               – consecutive-negative-GDP period detection
  6. rank_countries                  – percentile ranking of countries per indicator
  7. compute_moving_average          – rolling mean overlay for an indicator series
  8. run_custom_threshold_alerts     – signal detection with user-supplied thresholds
  9. compute_volatility              – rolling standard deviation of an indicator
 10. cluster_countries               – k-means clustering of countries by economic profile
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from analytics.forecasting import forecast_indicator
from analytics.signals import SignalAlert, alerts_to_dataframe

# Decimal places used when rounding output values
_DISPLAY_PRECISION = 2


# ---------------------------------------------------------------------------
# 1. Economic Health Score
# ---------------------------------------------------------------------------

# (weight, direction)  direction +1 = higher is better, -1 = lower is better
_HEALTH_CONFIG: dict[str, tuple[float, int]] = {
    "NY.GDP.MKTP.KD.ZG": (0.30, +1),   # GDP growth
    "FP.CPI.TOTL.ZG":    (0.25, -1),   # Inflation
    "SL.UEM.TOTL.ZS":    (0.25, -1),   # Unemployment
    "NY.GDP.PCAP.CD":    (0.20, +1),   # GDP per capita
}


def compute_economic_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-country, per-year composite economic health score (0–100).

    Each indicator is min-max normalised across the whole dataset, inverted if
    lower-is-better, then multiplied by its weight and summed.

    Parameters
    ----------
    df : long-format DataFrame with columns year, country, indicator_code, value

    Returns
    -------
    DataFrame with columns: year, country, health_score
    """
    codes = [c for c in _HEALTH_CONFIG if c in df["indicator_code"].unique()]
    if not codes:
        return pd.DataFrame(columns=["year", "country", "health_score"])

    pivot = (
        df[df["indicator_code"].isin(codes)]
        .pivot_table(index=["year", "country"], columns="indicator_code", values="value")
        .reset_index()
    )

    score = pd.Series(0.0, index=pivot.index)
    total_weight = 0.0
    for code in codes:
        if code not in pivot.columns:
            continue
        col = pivot[code].astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max == col_min:
            normalised = pd.Series(0.5, index=col.index)
        else:
            normalised = (col - col_min) / (col_max - col_min)
        weight, direction = _HEALTH_CONFIG[code]
        if direction == -1:
            normalised = 1.0 - normalised
        score += weight * normalised
        total_weight += weight

    if total_weight > 0:
        score = score / total_weight

    result = pivot[["year", "country"]].copy()
    result["health_score"] = (score * 100).round(1)
    return result.sort_values(["country", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Multi-Country Forecast Comparison
# ---------------------------------------------------------------------------

def multi_country_forecast(
    df: pd.DataFrame,
    countries: list[str],
    indicator_code: str,
    n_periods: int = 5,
) -> pd.DataFrame:
    """Forecast an indicator for multiple countries and combine into one DataFrame.

    Parameters
    ----------
    df             : long-format indicators DataFrame
    countries      : list of country names to forecast
    indicator_code : World Bank indicator code
    n_periods      : forecast horizon in years

    Returns
    -------
    DataFrame with columns: year, value, type, country, indicator_code, indicator_name
    Returns an empty DataFrame if no country had sufficient data.
    """
    frames: list[pd.DataFrame] = []
    for country in countries:
        result = forecast_indicator(df, country, indicator_code, n_periods=n_periods)
        if result is not None:
            frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. YoY Change Heatmap
# ---------------------------------------------------------------------------

def compute_yoy_heatmap(
    df: pd.DataFrame,
    indicator_code: str,
    countries: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return a pivot table of year-over-year absolute changes for one indicator.

    Rows = countries, Columns = years, Values = YoY change (percentage points).

    Parameters
    ----------
    df             : long-format DataFrame
    indicator_code : indicator to analyse
    countries      : subset of countries (None = all)

    Returns
    -------
    Pivot DataFrame indexed by country, columns are years.
    """
    subset = df[df["indicator_code"] == indicator_code].copy()
    if countries:
        subset = subset[subset["country"].isin(countries)]

    records: list[dict] = []
    for country, grp in subset.groupby("country"):
        grp = grp.sort_values("year").copy()
        grp["yoy"] = grp["value"].diff()
        for _, row in grp.dropna(subset=["yoy"]).iterrows():
            records.append({"country": country, "year": int(row["year"]), "yoy": round(row["yoy"], _DISPLAY_PRECISION)})

    if not records:
        return pd.DataFrame()

    long = pd.DataFrame(records)
    return long.pivot_table(index="country", columns="year", values="yoy")


# ---------------------------------------------------------------------------
# 4. Economic Risk Score Heatmap
# ---------------------------------------------------------------------------

_SEVERITY_SCORE: dict[str, int] = {"low": 1, "medium": 2, "high": 3}


def compute_risk_scores(
    alerts_df: pd.DataFrame,
    countries: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aggregate signal alerts into an annual risk-score matrix.

    Risk score per country/year = sum of severity weights (low=1, medium=2, high=3).

    Parameters
    ----------
    alerts_df : DataFrame from ``alerts_to_dataframe()``
    countries : optional country filter

    Returns
    -------
    Pivot DataFrame indexed by country, columns are years.
    """
    if alerts_df.empty:
        return pd.DataFrame()

    work = alerts_df.copy()
    if countries:
        work = work[work["country"].isin(countries)]
    if work.empty:
        return pd.DataFrame()

    work["score"] = work["severity"].map(_SEVERITY_SCORE).fillna(0)
    agg = work.groupby(["country", "year"])["score"].sum().reset_index()
    return agg.pivot_table(index="country", columns="year", values="score", fill_value=0)


# ---------------------------------------------------------------------------
# 5. Recession Detection
# ---------------------------------------------------------------------------

def detect_recessions(
    df: pd.DataFrame,
    country: str,
    min_consecutive: int = 2,
) -> list[dict]:
    """Identify recession periods (≥ N consecutive years of negative GDP growth).

    Parameters
    ----------
    df               : long-format DataFrame
    country          : country name
    min_consecutive  : minimum consecutive years of negative GDP growth (default 2)

    Returns
    -------
    List of dicts with keys: country, start_year, end_year, depth
    ``depth`` is the mean GDP growth rate across the recession period.
    """
    gdp_code = "NY.GDP.MKTP.KD.ZG"
    series = (
        df[(df["country"] == country) & (df["indicator_code"] == gdp_code)]
        .sort_values("year")[["year", "value"]]
        .dropna()
    )

    years = series["year"].tolist()
    values = series["value"].tolist()
    recessions: list[dict] = []
    i = 0

    while i < len(values):
        if values[i] < 0:
            j = i
            while j < len(values) and values[j] < 0:
                j += 1
            if j - i >= min_consecutive:
                recessions.append(
                    {
                        "country": country,
                        "start_year": int(years[i]),
                        "end_year": int(years[j - 1]),
                        "depth": round(float(np.mean(values[i:j])), _DISPLAY_PRECISION),
                    }
                )
            i = j
        else:
            i += 1

    return recessions


# ---------------------------------------------------------------------------
# 6. Country Percentile Ranking
# ---------------------------------------------------------------------------

def rank_countries(
    df: pd.DataFrame,
    indicator_code: str,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """Return per-country percentile rankings for one indicator in a given year.

    Parameters
    ----------
    df             : long-format DataFrame
    indicator_code : indicator code
    year           : year to rank (defaults to the latest available year)

    Returns
    -------
    DataFrame with columns: country, year, value, rank, percentile (0–100)
    Sorted descending by value.
    """
    subset = df[df["indicator_code"] == indicator_code]
    if subset.empty:
        return pd.DataFrame(columns=["country", "year", "value", "rank", "percentile"])

    if year is None:
        year = int(subset["year"].max())

    subset = subset[subset["year"] == year][["country", "year", "value"]].dropna().copy()
    if subset.empty:
        return pd.DataFrame(columns=["country", "year", "value", "rank", "percentile"])

    subset = subset.sort_values("value", ascending=False).reset_index(drop=True)
    subset["rank"] = subset.index + 1
    n = len(subset)
    subset["percentile"] = ((n - subset["rank"]) / max(n - 1, 1) * 100).round(1)
    return subset


# ---------------------------------------------------------------------------
# 7. Moving Average Overlay
# ---------------------------------------------------------------------------

def compute_moving_average(
    df: pd.DataFrame,
    country: str,
    indicator_code: str,
    window: int = 3,
) -> pd.DataFrame:
    """Return the indicator series with an additional rolling-mean column.

    Parameters
    ----------
    df             : long-format DataFrame
    country        : country name
    indicator_code : indicator code
    window         : rolling window in years (default 3)

    Returns
    -------
    DataFrame with columns: year, value, moving_avg
    """
    series = (
        df[(df["country"] == country) & (df["indicator_code"] == indicator_code)]
        .sort_values("year")[["year", "value"]]
        .dropna()
        .copy()
    )
    series["moving_avg"] = (
        # min_periods=1 ensures no NaN at the start of the series;
        # early values use whatever observations are available.
        series["value"].rolling(window=window, min_periods=1).mean().round(_DISPLAY_PRECISION)
    )
    return series.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Custom Alert Thresholds
# ---------------------------------------------------------------------------

def run_custom_threshold_alerts(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Run level-based signal detection with caller-supplied thresholds.

    For GDP growth the threshold is a minimum (below triggers alert).
    For all other indicators the threshold is a maximum (above triggers alert).

    Parameters
    ----------
    df         : long-format DataFrame
    thresholds : mapping of indicator_code → threshold value

    Returns
    -------
    DataFrame with same schema as ``alerts_to_dataframe()``.
    """
    gdp_code = "NY.GDP.MKTP.KD.ZG"
    alerts: list[SignalAlert] = []

    for code, threshold in thresholds.items():
        subset = df[df["indicator_code"] == code].dropna(subset=["value"])
        for _, row in subset.iterrows():
            triggered = (
                (code == gdp_code and row["value"] < threshold)
                or (code != gdp_code and row["value"] > threshold)
            )
            if triggered:
                direction = "below" if code == gdp_code else "above"
                alerts.append(
                    SignalAlert(
                        country=row["country"],
                        indicator=row["indicator_name"],
                        year=int(row["year"]),
                        value=round(float(row["value"]), _DISPLAY_PRECISION),
                        signal_type=f"Custom: {row['indicator_name']} {direction} {threshold}",
                        severity="medium",
                        description=(
                            f"{row['country']}: {row['indicator_name']} = {row['value']:.2f}"
                            f" in {int(row['year'])} ({direction} threshold {threshold})."
                        ),
                    )
                )

    return alerts_to_dataframe(alerts)


# ---------------------------------------------------------------------------
# 9. Volatility Analysis
# ---------------------------------------------------------------------------

def compute_volatility(
    df: pd.DataFrame,
    indicator_code: str,
    window: int = 5,
    countries: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return rolling standard deviation of an indicator for each country.

    Parameters
    ----------
    df             : long-format DataFrame
    indicator_code : indicator code
    window         : rolling window in years (default 5)
    countries      : optional country filter

    Returns
    -------
    Long-format DataFrame with columns: year, country, value, rolling_std
    """
    subset = df[df["indicator_code"] == indicator_code].copy()
    if countries:
        subset = subset[subset["country"].isin(countries)]

    frames: list[pd.DataFrame] = []
    for country, grp in subset.groupby("country"):
        grp = grp.sort_values("year").copy()
        grp["rolling_std"] = (
            grp["value"].rolling(window=window, min_periods=2).std().round(3)
        )
        frames.append(grp[["year", "country", "value", "rolling_std"]])

    if not frames:
        return pd.DataFrame(columns=["year", "country", "value", "rolling_std"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 10. Economic Similarity Clustering
# ---------------------------------------------------------------------------

def cluster_countries(
    df: pd.DataFrame,
    year: Optional[int] = None,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Cluster countries by their economic profile in a given year using k-means.

    Features are the available indicator values, standardised before clustering.

    Parameters
    ----------
    df         : long-format DataFrame
    year       : year to use (defaults to the most recent year with data)
    n_clusters : number of k-means clusters

    Returns
    -------
    DataFrame with columns: country, cluster, year, + one column per indicator
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if year is None:
        year = int(df["year"].max())

    codes = sorted(df["indicator_code"].unique())
    year_df = df[df["year"] == year]
    pivot = (
        year_df.pivot_table(index="country", columns="indicator_code", values="value")[codes]
        .dropna()
    )

    if pivot.empty:
        return pd.DataFrame()

    n_clusters = max(1, min(n_clusters, len(pivot)))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pivot.values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)

    result = pivot.reset_index().copy()
    result["cluster"] = labels
    result["year"] = year
    return result
