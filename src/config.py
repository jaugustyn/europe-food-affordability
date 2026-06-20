"""Shared project configuration."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

YEAR_MIN = 2010
YEAR_MAX = 2024

FOOD_CATEGORIES = {
    "CP0111": "Pieczywo i produkty zbożowe",
    "CP0112": "Mięso",
    "CP0113": "Ryby i owoce morza",
    "CP0114": "Mleko, sery i jaja",
    "CP0115": "Oleje i tłuszcze",
    "CP0116": "Owoce",
    "CP0117": "Warzywa",
    "CP0118": "Cukier, dżem, miód, czekolada i wyroby cukiernicze",
    "CP0119": "Pozostałe produkty żywnościowe",
    "CP012": "Napoje bezalkoholowe",
}

# Eurostat uses EL for Greece and sometimes UK for the United Kingdom.
EUROSTAT_TO_ISO2 = {"EL": "GR", "UK": "GB"}

REQUIRED_ANALYTIC_COLUMNS = [
    "country_code",
    "country_name",
    "iso3",
    "region",
    "year",
    "food_inflation_pct",
    "food_inflation_pct_imputed",
    "headline_inflation_pct",
    "headline_inflation_pct_imputed",
    "median_income_eur",
    "median_income_eur_imputed",
    "income_growth_pct",
    "food_price_level_index",
    "food_price_level_index_imputed",
    "food_share_budget_pct",
    "food_share_budget_pct_imputed",
    "meal_deprivation_pct",
    "meal_deprivation_pct_imputed",
    "minimum_wage_eur_month",
    "minimum_wage_eur_month_imputed",
    "food_affordability_gap_pct",
    "food_inflation_index_2020",
]
