"""ETL orchestration for the food-price-pressure dashboard."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT, YEAR_MAX, YEAR_MIN
from .eurostat_sources import (
    get_food_inflation,
    get_food_category_inflation,
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


def _quality_profile(df: pd.DataFrame, view: str, stage: str) -> pd.DataFrame:
    """Return an auditable column-level data-quality snapshot."""
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "view": view,
                "stage": stage,
                "column": col,
                "dtype": str(df[col].dtype),
                "row_count": int(len(df)),
                "missing_count": missing,
                "missing_pct": float(missing / len(df) * 100) if len(df) else np.nan,
                "unique_count": int(df[col].nunique(dropna=True)),
                "imputed_count": (
                    int(df[f"{col}_imputed"].fillna(False).astype(bool).sum())
                    if not col.endswith("_imputed") and f"{col}_imputed" in df.columns
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_output(
    df: pd.DataFrame,
    *,
    key: list[str],
    name: str,
    min_rows: int = 1,
    min_numeric: int = 5,
) -> None:
    if len(df) < min_rows:
        raise ValueError(f"{name}: expected at least {min_rows} rows, got {len(df)}")
    duplicates = int(df.duplicated(key).sum())
    if duplicates:
        raise ValueError(f"{name}: found {duplicates} duplicate keys: {key}")
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) < min_numeric:
        raise ValueError(
            f"{name}: expected at least {min_numeric} numeric columns, got {len(numeric.columns)}"
        )
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError(f"{name}: infinite numeric values detected")
    if not df["year"].between(YEAR_MIN, YEAR_MAX).all():
        raise ValueError(
            f"{name}: year outside the configured {YEAR_MIN}-{YEAR_MAX} range"
        )


def run_pipeline() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    regions = load_regions()
    country_whitelist = set(regions["country_code"])
    category_inflation = get_food_category_inflation()
    category_inflation = category_inflation[
        category_inflation["country_code"].isin(country_whitelist)
    ].copy()
    category_quality_before = _quality_profile(
        category_inflation, "country_year_category", "raw_grid_before_etl"
    )
    observed_category_inflation = category_inflation.dropna(
        subset=["category_food_inflation_pct"]
    ).copy()
    log.info(
        "Category HICP grid: %d rows, %d missing, %d observed",
        len(category_inflation),
        int(category_inflation["category_food_inflation_pct"].isna().sum()),
        len(observed_category_inflation),
    )

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
    quality_before = _quality_profile(df, "country_year", "before_missing_policy")

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
    if "food_price_level_source" in df.columns:
        imputed_ppp = df["food_price_level_index"].notna() & df["food_price_level_source"].isna()
        df.loc[imputed_ppp, "food_price_level_source"] = "interpolated"
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
        "food_inflation_pct_imputed",
        "headline_inflation_pct",
        "headline_inflation_pct_imputed",
        "median_income_eur",
        "median_income_eur_imputed",
        "income_growth_pct",
        "minimum_wage_eur_month",
        "minimum_wage_eur_month_imputed",
        "food_price_level_index",
        "food_price_level_index_imputed",
        "food_price_level_source",
        "food_share_budget_pct",
        "food_share_budget_pct_imputed",
        "meal_deprivation_pct",
        "meal_deprivation_pct_imputed",
        "food_affordability_gap_pct",
        "food_inflation_index_2020",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    context_cols = [
        "country_code",
        "country_name",
        "iso3",
        "region",
        "year",
        "headline_inflation_pct",
        "headline_inflation_pct_imputed",
        "median_income_eur",
        "median_income_eur_imputed",
        "income_growth_pct",
        "food_price_level_index",
        "food_price_level_index_imputed",
        "food_price_level_source",
        "food_share_budget_pct",
        "food_share_budget_pct_imputed",
        "meal_deprivation_pct",
        "meal_deprivation_pct_imputed",
    ]
    category_df = observed_category_inflation.merge(
        df[context_cols], on=["country_code", "year"], how="inner", validate="many_to_one"
    )
    category_df["category_affordability_gap_pct"] = (
        category_df["category_food_inflation_pct"] - category_df["income_growth_pct"]
    )
    category_df = category_df.sort_values(
        ["country_code", "year", "food_category_code"]
    ).reset_index(drop=True)

    _validate_output(df, key=["country_code", "year"], name="country_year")
    _validate_output(
        category_df,
        key=["country_code", "year", "food_category_code"],
        name="country_year_category",
        min_rows=1000,
    )

    quality_after = _quality_profile(df, "country_year", "after_etl")
    category_quality = _quality_profile(
        category_df, "country_year_category", "after_etl"
    )
    quality = pd.concat(
        [quality_before, quality_after, category_quality_before, category_quality],
        ignore_index=True,
    )

    out_path = DATA_DIR / "merged.parquet"
    category_path = DATA_DIR / "food_categories.parquet"
    quality_path = DATA_DIR / "data_quality.csv"
    excl_path = DATA_DIR / "exclusions.csv"
    df.to_parquet(out_path, index=False)
    category_df.to_parquet(category_path, index=False)
    quality.to_csv(quality_path, index=False)
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
    log.info("Category view: %s (%d rows)", category_path, len(category_df))
    log.info("Data quality report: %s", quality_path)
    log.info("Columns: %s", list(df.columns))
    return df


def main() -> None:
    configure_logging()
    run_pipeline()
