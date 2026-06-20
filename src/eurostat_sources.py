"""Eurostat source adapters used by the ETL pipeline."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EUROSTAT_TO_ISO2, FOOD_CATEGORIES, RAW_DIR, YEAR_MAX, YEAR_MIN

log = logging.getLogger("etl")


def fetch_eurostat(dataset: str, filter_pars: dict, cache_name: str) -> pd.DataFrame:
    """Fetch a Eurostat dataset and cache the raw table as parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{cache_name}.parquet"
    if cache.exists():
        log.info("Cache HIT  Eurostat: %s", cache.name)
        return pd.read_parquet(cache)

    import eurostat  # lazy import; slow and only needed for ETL

    log.info("Cache MISS Eurostat: %s (filter=%s)", dataset, filter_pars)
    df = eurostat.get_data_df(dataset, filter_pars=filter_pars)
    if df is None or df.empty:
        raise RuntimeError(f"Eurostat returned an empty dataset: {dataset}")
    df.to_parquet(cache, index=False)
    return df


def _time_columns(df: pd.DataFrame) -> list[str | int]:
    """Return annual and semester columns from Eurostat wide tables."""
    out: list[str | int] = []
    for col in df.columns:
        if isinstance(col, (int, np.integer)):
            out.append(col)
            continue
        if isinstance(col, str) and re.fullmatch(r"\d{4}(-S[12])?", col):
            out.append(col)
    return out


def _country_column(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if str(c).lower().startswith("geo"))


def reshape_eurostat_long(
    df: pd.DataFrame,
    value_col: str,
    *,
    aggregate_semesters: bool = False,
) -> pd.DataFrame:
    """Reshape Eurostat wide output to country_code, year, value_col."""
    geo_col = _country_column(df)
    long = df.melt(
        id_vars=[geo_col],
        value_vars=_time_columns(df),
        var_name="period",
        value_name=value_col,
    ).rename(columns={geo_col: "country_code"})

    period = long["period"].astype(str)
    long["year"] = period.str.extract(r"(\d{4})").astype(int)
    long["country_code"] = long["country_code"].replace(EUROSTAT_TO_ISO2)
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")
    long = long[(long["year"] >= YEAR_MIN) & (long["year"] <= YEAR_MAX)]

    if aggregate_semesters:
        return (
            long.groupby(["country_code", "year"], as_index=False)[value_col]
            .mean()
            .sort_values(["country_code", "year"])
        )

    return long[["country_code", "year", value_col]].sort_values(
        ["country_code", "year"]
    )


def _first_working_dataset(specs: Iterable[tuple[str, dict, str]], value_col: str) -> pd.DataFrame:
    """Try compatible Eurostat dataset/filter variants until one works."""
    errors: list[str] = []
    for dataset, filters, cache_name in specs:
        try:
            raw = fetch_eurostat(dataset, filters, cache_name)
            return reshape_eurostat_long(raw, value_col)
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
            log.warning("Skipping Eurostat variant %s (%s)", dataset, exc)
    raise RuntimeError("; ".join(errors))


def combine_ppp_sources(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Prefer a valid new PPP value, otherwise fall back to the historical source."""
    combined = pd.concat(parts, ignore_index=True)
    combined["has_value"] = combined["food_price_level_index"].notna()
    combined = combined.sort_values(
        ["country_code", "year", "has_value", "source_priority"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["country_code", "year"], keep="first")
    combined.loc[
        combined["food_price_level_index"].isna(), "food_price_level_source"
    ] = pd.NA
    return combined.drop(columns=["source_priority", "has_value"])


def get_food_inflation() -> pd.DataFrame:
    """HICP CP01: food and non-alcoholic beverages, annual rate of change."""
    raw = fetch_eurostat(
        "prc_hicp_aind",
        {"coicop": ["CP01"], "unit": ["RCH_A_AVG"]},
        "eurostat_food_inflation",
    )
    return reshape_eurostat_long(raw, "food_inflation_pct")


def get_food_category_inflation() -> pd.DataFrame:
    """Full annual HICP country-year-category grid, including missing values."""
    raw = fetch_eurostat(
        "prc_hicp_aind",
        {"coicop": list(FOOD_CATEGORIES), "unit": ["RCH_A_AVG"]},
        "eurostat_food_categories",
    )
    geo_col = _country_column(raw)
    long = raw.melt(
        id_vars=[geo_col, "coicop"],
        value_vars=_time_columns(raw),
        var_name="year",
        value_name="category_food_inflation_pct",
    ).rename(
        columns={geo_col: "country_code", "coicop": "food_category_code"}
    )
    long["year"] = long["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    long["country_code"] = long["country_code"].replace(EUROSTAT_TO_ISO2)
    long["category_food_inflation_pct"] = pd.to_numeric(
        long["category_food_inflation_pct"], errors="coerce"
    )
    long = long[
        long["year"].between(YEAR_MIN, YEAR_MAX)
        & long["food_category_code"].isin(FOOD_CATEGORIES)
    ]
    long["food_category_name"] = long["food_category_code"].map(FOOD_CATEGORIES)
    return long[
        [
            "country_code",
            "year",
            "food_category_code",
            "food_category_name",
            "category_food_inflation_pct",
        ]
    ].sort_values(["country_code", "year", "food_category_code"])


def get_headline_inflation() -> pd.DataFrame:
    """HICP CP00: all-items annual rate of change."""
    raw = fetch_eurostat(
        "prc_hicp_aind",
        {"coicop": ["CP00"], "unit": ["RCH_A_AVG"]},
        "eurostat_headline_inflation",
    )
    return reshape_eurostat_long(raw, "headline_inflation_pct")


def get_median_income() -> pd.DataFrame:
    """Median equivalised net income, total population, euro per year."""
    raw = fetch_eurostat(
        "ilc_di03",
        {
            "age": ["TOTAL"],
            "sex": ["T"],
            "indic_il": ["MED_E"],
            "unit": ["EUR"],
        },
        "eurostat_median_income",
    )
    return reshape_eurostat_long(raw, "median_income_eur")


def get_minimum_wage() -> pd.DataFrame:
    """Monthly minimum wage in euro; biannual values averaged to country-year."""
    raw = fetch_eurostat(
        "earn_mw_cur",
        {"currency": ["EUR"]},
        "eurostat_minimum_wage",
    )
    return reshape_eurostat_long(
        raw,
        "minimum_wage_eur_month",
        aggregate_semesters=True,
    )


def get_food_price_level() -> pd.DataFrame:
    """Food price level index versus EU27_2020=100.

    Eurostat moved PPP product categories to COICOP 2018 in prc_ppp_ind_1.
    The old prc_ppp_ind table remains a fallback for historical years and uses
    the same A0101 food/non-alcoholic beverages category in the legacy dimension.
    """
    parts: list[pd.DataFrame] = []
    specs = [
        (
            "prc_ppp_ind_1",
            {"indic_ppp": ["PLI_EU27_2020"], "ppp_cat18": ["A0101"]},
            "eurostat_food_price_level_coicop18",
        ),
        (
            "prc_ppp_ind",
            {"na_item": ["PLI_EU27_2020"], "ppp_cat": ["A0101"]},
            "eurostat_food_price_level",
        ),
    ]
    for dataset, filters, cache_name in specs:
        try:
            raw = fetch_eurostat(dataset, filters, cache_name)
            part = reshape_eurostat_long(raw, "food_price_level_index")
            part["food_price_level_source"] = dataset
            part["source_priority"] = 0 if dataset == "prc_ppp_ind_1" else 1
            parts.append(part)
        except Exception as exc:
            log.warning("Skipping PPP source %s (%s)", dataset, exc)

    if not parts:
        raise RuntimeError("No usable Eurostat PPP food price dataset returned data")

    return combine_ppp_sources(parts)


def get_food_share_budget() -> pd.DataFrame:
    """Household final consumption share spent on food, percent of total."""
    raw = fetch_eurostat(
        "nama_10_co3_p3",
        {"coicop": ["CP01"], "unit": ["PC_TOT"]},
        "eurostat_food_share",
    )
    return reshape_eurostat_long(raw, "food_share_budget_pct")


def get_meal_deprivation() -> pd.DataFrame:
    """People unable to afford a proper meal every second day, percent."""
    raw = fetch_eurostat(
        "ilc_mdes03",
        {"hhtyp": ["TOTAL"], "incgrp": ["TOTAL"], "unit": ["PC"]},
        "eurostat_meal_deprivation",
    )
    return reshape_eurostat_long(raw, "meal_deprivation_pct")
