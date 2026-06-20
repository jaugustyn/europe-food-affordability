"""Data-quality policy and analytic metric construction."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("etl")


def apply_missing_policy(
    df: pd.DataFrame,
    strict_cols: list[str],
    soft_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate short gaps, mark filled cells and log incomplete strict rows."""
    df = df.sort_values(["country_code", "year"]).copy()
    exclusions = []

    for col in strict_cols:
        if col not in df.columns:
            continue
        missing_before = df[col].isna()
        df[col] = df.groupby("country_code")[col].transform(
            lambda s: s.interpolate(method="linear", limit=2, limit_area="inside")
        )
        df[f"{col}_imputed"] = missing_before & df[col].notna()

    for col in soft_cols:
        if col not in df.columns:
            continue
        missing_before = df[col].isna()
        df[col] = df.groupby("country_code")[col].transform(
            lambda s: (
                s.interpolate(method="linear", limit=3, limit_area="inside")
                .ffill(limit=3)
                .bfill(limit=3)
            )
        )
        df[f"{col}_imputed"] = missing_before & df[col].notna()

    present_strict = [c for c in strict_cols if c in df.columns]
    mask_bad = df[present_strict].isna().any(axis=1) if present_strict else False
    for _, row in df[mask_bad].iterrows():
        missing = [c for c in present_strict if pd.isna(row[c])]
        exclusions.append(
            {
                "country_code": row["country_code"],
                "year": int(row["year"]),
                "missing_columns": ";".join(missing),
            }
        )

    clean = df.loc[~mask_bad].copy()
    excl = pd.DataFrame(exclusions, columns=["country_code", "year", "missing_columns"])
    log.info("Missing-data policy: removed %d rows (out of %d)", int(mask_bad.sum()), len(df))
    return clean, excl


def add_income_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year median income growth by country."""
    df = df.sort_values(["country_code", "year"]).copy()
    df["income_growth_pct"] = (
        df.groupby("country_code")["median_income_eur"]
        .pct_change(fill_method=None)
        .mul(100)
    )
    return df


def add_food_pressure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add affordability gap and 2020-based cumulative price index."""
    df = df.copy()
    df["food_affordability_gap_pct"] = (
        df["food_inflation_pct"].astype(float) - df["income_growth_pct"].astype(float)
    )

    base_year = 2020
    df = df.sort_values(["country_code", "year"])
    df["food_inflation_index_2020"] = np.nan
    for _, group in df.groupby("country_code", sort=False):
        rates = group["food_inflation_pct"].astype(float).fillna(0)
        chained_index = (1 + rates / 100).cumprod()
        base_value = chained_index.loc[group["year"] == base_year]
        if not base_value.empty and base_value.iloc[0] != 0:
            df.loc[group.index, "food_inflation_index_2020"] = (
                chained_index / float(base_value.iloc[0]) * 100
            )
    return df
