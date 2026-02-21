"""
Simple time-series forecasting for EconoPulse.

Uses linear regression as a baseline and, when statsmodels is available,
an exponential smoothing (Holt's linear trend) model.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import Holt  # type: ignore

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


def _linear_forecast(values: np.ndarray, n_periods: int) -> np.ndarray:
    """Return n_periods forecasts using OLS linear trend."""
    t = np.arange(len(values), dtype=float)
    coeffs = np.polyfit(t, values, 1)
    future_t = np.arange(len(values), len(values) + n_periods, dtype=float)
    return np.polyval(coeffs, future_t)


def forecast_indicator(
    df: pd.DataFrame,
    country: str,
    indicator_code: str,
    n_periods: int = 5,
    method: str = "auto",
) -> Optional[pd.DataFrame]:
    """Forecast an indicator for a given country.

    Parameters
    ----------
    df : long-format DataFrame (year, country, indicator_code, indicator_name, value)
    country : country name
    indicator_code : World Bank indicator code
    n_periods : number of years to forecast
    method : "linear" | "holt" | "auto"
              "auto" uses Holt when statsmodels is available, else linear.

    Returns
    -------
    DataFrame with columns: year, value, type ("historical" | "forecast")
    or None if insufficient data.
    """
    mask = (df["country"] == country) & (df["indicator_code"] == indicator_code)
    series = df[mask].sort_values("year").dropna(subset=["value"])

    if len(series) < 4:
        return None

    indicator_name = series["indicator_name"].iloc[0]
    years = series["year"].values
    values = series["value"].values.astype(float)
    last_year = int(years[-1])
    future_years = list(range(last_year + 1, last_year + n_periods + 1))

    use_holt = (method == "holt") or (method == "auto" and _STATSMODELS_AVAILABLE)

    if use_holt and _STATSMODELS_AVAILABLE and len(values) >= 6:
        try:
            model = Holt(values, initialization_method="estimated").fit(optimized=True)
            forecast_values = model.forecast(n_periods)
        except Exception:  # noqa: BLE001
            forecast_values = _linear_forecast(values, n_periods)
    else:
        forecast_values = _linear_forecast(values, n_periods)

    historical = pd.DataFrame(
        {
            "year": years.tolist(),
            "value": values.tolist(),
            "type": "historical",
            "country": country,
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
        }
    )
    forecasted = pd.DataFrame(
        {
            "year": future_years,
            "value": forecast_values.tolist(),
            "type": "forecast",
            "country": country,
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
        }
    )
    return pd.concat([historical, forecasted], ignore_index=True)
