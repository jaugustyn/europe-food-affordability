"""ARIMA-based forecasting of annual food-affordability metrics."""
from __future__ import annotations

import warnings

import pandas as pd

DEFAULT_ORDER: tuple[int, int, int] = (1, 1, 1)
MIN_OBS = 8


def forecast_inflation(
    df: pd.DataFrame,
    country_code: str,
    metric: str = "food_inflation_pct",
    horizon: int = 2,
    order: tuple[int, int, int] = DEFAULT_ORDER,
) -> dict:
    """Forecast `metric` for `country_code` over `horizon` years ahead."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    series = (
        df[df.country_code == country_code]
        .sort_values("year")
        .set_index("year")[metric]
        .dropna()
    )
    if len(series) < MIN_OBS:
        return {"error": f"Too few observations ({len(series)}); minimum is {MIN_OBS}."}

    try:
        adf_p = float(adfuller(series.values, autolag="AIC")[1])
    except Exception:
        adf_p = float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series.values, order=order).fit()
            fc = model.get_forecast(steps=horizon)
            mean = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
    except Exception as exc:
        return {"error": f"ARIMA{order} did not converge: {exc}"}

    last_year = int(series.index.max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    forecast_df = pd.DataFrame(
        {
            "year": future_years,
            "forecast": mean,
            "lower": ci[:, 0],
            "upper": ci[:, 1],
        }
    )
    history_df = series.reset_index()

    return {
        "history": history_df,
        "forecast": forecast_df,
        "order": order,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n_obs": len(series),
        "adf_p": adf_p,
    }


def forecast_food_inflation(
    df: pd.DataFrame,
    country_name: str,
    periods: int = 3,
    order: tuple[int, int, int] = DEFAULT_ORDER,
) -> dict:
    """Forecast food inflation for a country selected by display name."""
    countries = df.loc[df["country_name"] == country_name, "country_code"].dropna().unique()
    if len(countries) == 0:
        raise ValueError(f"Country not found in the dataset: {country_name}")

    result = forecast_inflation(
        df=df,
        country_code=str(countries[0]),
        metric="food_inflation_pct",
        horizon=periods,
        order=order,
    )
    if "error" in result:
        raise ValueError(result["error"])
    return result
