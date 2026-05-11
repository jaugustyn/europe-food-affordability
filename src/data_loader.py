"""Cached data loaders for the Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .config import REQUIRED_ANALYTIC_COLUMNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "merged.parquet"
EXCL_PATH = Path(__file__).resolve().parent.parent / "data" / "exclusions.csv"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run `python etl.py` first.")

    df = pd.read_parquet(DATA_PATH)
    missing = [c for c in REQUIRED_ANALYTIC_COLUMNS if c not in df.columns]
    if missing:
        raise FileNotFoundError(
            "data/merged.parquet uses an outdated schema. Run `python etl.py` "
            f"again. Missing columns: {', '.join(missing)}"
        )
    return df


@st.cache_data(show_spinner=False)
def load_exclusions() -> pd.DataFrame:
    if not EXCL_PATH.exists():
        return pd.DataFrame(columns=["country_code", "year", "missing_columns"])
    return pd.read_csv(EXCL_PATH)


def filter_df(
    df: pd.DataFrame,
    years: tuple[int, int] | None = None,
    countries: list[str] | None = None,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    out = df
    if years is not None:
        y0, y1 = years
        out = out[(out["year"] >= y0) & (out["year"] <= y1)]
    if countries:
        out = out[out["country_code"].isin(countries)]
    if regions:
        out = out[out["region"].isin(regions)]
    return out
