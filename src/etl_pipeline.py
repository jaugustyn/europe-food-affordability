"""ETL orchestration for the food-price-pressure dashboard."""
from __future__ import annotations

import logging
import sys

import pandas as pd

from .config import DATA_DIR, ROOT
from .eurostat_sources import (
    get_food_inflation,
    get_food_price_level,
    get_food_share_budget,
    get_headline_inflation,
    get_meal_deprivation,
    get_median_income,
    get_minimum_wage,
)
from .transforms import (
    add_food_pressure_metrics,
    add_income_growth,
    apply_missing_policy,
)

log = logging.getLogger("etl")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "etl.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_regions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "regions.csv")
    log.info("Loaded country mapping: %d entries", len(df))
    return df


def _merge_sources(parts: list[pd.DataFrame]) -> pd.DataFrame:
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["country_code", "year"], how="outer")
    return out


def run_pipeline() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_regions()
    country_whitelist = set(regions["country_code"])

    log.info("=== Eurostat sources ===")
    df = _merge_sources(
        [
            get_food_inflation(),
            get_headline_inflation(),
            get_median_income(),
            get_minimum_wage(),
            get_food_price_level(),
            get_food_share_budget(),
            get_meal_deprivation(),
        ]
    )
    df = df[df["country_code"].isin(country_whitelist)]

    strict_cols = [
        "food_inflation_pct",
        "headline_inflation_pct",
        "median_income_eur",
    ]
    soft_cols = [
        "minimum_wage_eur_month",
        "food_price_level_index",
        "food_share_budget_pct",
        "meal_deprivation_pct",
    ]

    log.info("=== Transform ===")
    df, exclusions = apply_missing_policy(df, strict_cols, soft_cols)
    df = add_income_growth(df)
    df = add_food_pressure_metrics(df)

    df = df.merge(
        regions[["country_code", "country_name", "iso3", "region"]],
        on="country_code",
        how="left",
    )

    ordered_cols = [
        "country_code",
        "country_name",
        "iso3",
        "region",
        "year",
        "food_inflation_pct",
        "headline_inflation_pct",
        "median_income_eur",
        "income_growth_pct",
        "minimum_wage_eur_month",
        "food_price_level_index",
        "food_share_budget_pct",
        "meal_deprivation_pct",
        "fpi",
        "food_affordability_gap_pct",
        "food_inflation_index_2020",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    out_path = DATA_DIR / "merged.parquet"
    excl_path = DATA_DIR / "exclusions.csv"
    df.to_parquet(out_path, index=False)
    exclusions.to_csv(excl_path, index=False)

    log.info(
        "Wrote: %s (%d rows, %d countries, years %d-%d)",
        out_path,
        len(df),
        df["country_code"].nunique(),
        int(df["year"].min()),
        int(df["year"].max()),
    )
    log.info("Exclusions: %s (%d)", excl_path, len(exclusions))
    log.info("Columns: %s", list(df.columns))
    return df


def main() -> None:
    configure_logging()
    run_pipeline()
