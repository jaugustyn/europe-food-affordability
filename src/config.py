"""Shared project configuration."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

YEAR_MIN = 2010
YEAR_MAX = 2024

# Eurostat uses EL for Greece and sometimes UK for the United Kingdom.
EUROSTAT_TO_ISO2 = {"EL": "GR", "UK": "GB"}

REQUIRED_ANALYTIC_COLUMNS = [
    "country_code",
    "country_name",
    "iso3",
    "region",
    "year",
    "food_inflation_pct",
    "headline_inflation_pct",
    "median_income_eur",
    "income_growth_pct",
    "food_price_level_index",
    "food_share_budget_pct",
    "meal_deprivation_pct",
    "minimum_wage_eur_month",
    "fpi",
    "food_affordability_gap_pct",
    "food_inflation_index_2020",
]
