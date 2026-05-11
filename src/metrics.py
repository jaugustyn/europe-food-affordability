"""Metric dictionary and KPI helper utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column name -> metadata used by the dashboard.
METRICS: dict[str, dict] = {
    "fpi": {
        "label": "Food Pressure Index",
        "unit": "",
        "fmt": "{:+.2f}",
        "desc": "Synthetic ratio: food inflation (%) / median income growth (%). "
                "Values above 1 mean food prices grew faster than income.",
        "color_scale": "Reds",
        "diverging": False,
    },
    "food_inflation_pct": {
        "label": "Food Inflation (HICP CP01)",
        "unit": "%",
        "fmt": "{:+.1f}%",
        "desc": "Annual average rate of change for the harmonised food and non-alcoholic beverages price index.",
        "color_scale": "Reds",
        "diverging": False,
    },
    "headline_inflation_pct": {
        "label": "Headline Inflation (HICP CP00)",
        "unit": "%",
        "fmt": "{:+.1f}%",
        "desc": "Annual average rate of change for the all-items harmonised price index.",
        "color_scale": "Oranges",
        "diverging": False,
    },
    "food_share_budget_pct": {
        "label": "Food Share of Household Spending",
        "unit": "%",
        "fmt": "{:.1f}%",
        "desc": "Food and non-alcoholic beverages as a share of household final consumption expenditure.",
        "color_scale": "YlOrBr",
        "diverging": False,
    },
    "median_income_eur": {
        "label": "Median Equivalised Income",
        "unit": "EUR/year",
        "fmt": "€{:,.0f}",
        "desc": "Annual median equivalised net income (Eurostat, ilc_di03).",
        "color_scale": "Viridis",
        "diverging": False,
    },
    "income_growth_pct": {
        "label": "Median Income Growth",
        "unit": "%",
        "fmt": "{:+.1f}%",
        "desc": "Country-level annual growth rate of median equivalised income.",
        "color_scale": "RdYlGn",
        "diverging": True,
    },
    "minimum_wage_eur_month": {
        "label": "Minimum Wage",
        "unit": "EUR/month",
        "fmt": "€{:,.0f}",
        "desc": "Monthly minimum wage in EUR; semi-annual data averaged to years (Eurostat, earn_mw_cur).",
        "color_scale": "Blues",
        "diverging": False,
    },
    "food_price_level_index": {
        "label": "Food Price Level (EU=100)",
        "unit": "",
        "fmt": "{:.0f}",
        "desc": "Price level index for food and non-alcoholic beverages relative to EU27_2020=100.",
        "color_scale": "Plasma",
        "diverging": False,
    },
    "meal_deprivation_pct": {
        "label": "Unable to Afford a Proper Meal",
        "unit": "%",
        "fmt": "{:.1f}%",
        "desc": "Share of people unable to afford a meal with meat, fish, or a vegetarian equivalent every second day.",
        "color_scale": "Magma",
        "diverging": False,
    },
    "food_affordability_gap_pct": {
        "label": "Food Affordability Gap",
        "unit": "pp",
        "fmt": "{:+.1f} pp",
        "desc": "Food inflation minus median income growth. Positive values imply deteriorating affordability.",
        "color_scale": "RdBu_r",
        "diverging": True,
    },
    "food_inflation_index_2020": {
        "label": "Cumulative Food Price Index (2020=100)",
        "unit": "",
        "fmt": "{:.1f}",
        "desc": "Cumulative index built from HICP CP01 annual rates with 2020 as the base year.",
        "color_scale": "Reds",
        "diverging": False,
    },
}

KEY_METRICS = [
    "fpi", "food_inflation_pct", "headline_inflation_pct",
    "median_income_eur", "income_growth_pct", "food_share_budget_pct",
    "food_price_level_index", "meal_deprivation_pct", "food_affordability_gap_pct",
]


def fmt(value: float, metric: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    return METRICS[metric]["fmt"].format(value)


def yoy_delta(df: pd.DataFrame, metric: str, country_code: str, year: int) -> float | None:
    """Year-over-year change of `metric` for a given country."""
    cur = df[(df.country_code == country_code) & (df.year == year)][metric]
    prev = df[(df.country_code == country_code) & (df.year == year - 1)][metric]
    if cur.empty or prev.empty:
        return None
    a, b = cur.iloc[0], prev.iloc[0]
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a - b)


def bootstrap_ci(
    values: np.ndarray, n_iter: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap the mean and return (mean, ci_low, ci_high) at 1−alpha."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_iter, arr.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(low), float(high)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "country_code",
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float, int]:
    """Cluster bootstrap CI for a mean, resampling whole countries.

    Row-level bootstrap treats every country-year as independent. For panel data
    this is too optimistic because observations from the same country are
    correlated over time. Cluster bootstrap samples countries with replacement
    and keeps all selected years for each sampled country.
    """
    sub = df[[cluster_col, value_col]].dropna().copy()
    if sub.empty:
        return (np.nan, np.nan, np.nan, 0)

    clusters = sub[cluster_col].dropna().unique()
    n_clusters = len(clusters)
    observed_mean = float(sub[value_col].mean())
    if n_clusters <= 1:
        return (observed_mean, observed_mean, observed_mean, n_clusters)

    grouped = {
        cluster: np.asarray(sub.loc[sub[cluster_col] == cluster, value_col], dtype=float)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_values = np.concatenate([grouped[cluster] for cluster in sampled_clusters])
        means[i] = sampled_values.mean()

    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed_mean, float(low), float(high), n_clusters


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi)
