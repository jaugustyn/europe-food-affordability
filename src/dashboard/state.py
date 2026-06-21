"""Loading, filtering and state construction for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.dashboard.support import REGION_LABELS_PL, region_label
from src.data_loader import load_category_data, load_data, load_data_quality, load_exclusions


@dataclass(frozen=True)
class DashboardContext:
    df: pd.DataFrame
    category_df: pd.DataFrame
    data_quality: pd.DataFrame
    exclusions: pd.DataFrame
    min_year: int
    max_year: int
    year_range: tuple[int, int]
    reference_year: int
    regions: list[str]
    countries: list[str]
    country_options: list[str]
    df_filtered: pd.DataFrame
    category_filtered: pd.DataFrame
    scope_df: pd.DataFrame
    latest: pd.DataFrame


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner=False)
def get_exclusions() -> pd.DataFrame:
    return load_exclusions()


@st.cache_data(show_spinner=False)
def get_category_data() -> pd.DataFrame:
    return load_category_data()


@st.cache_data(show_spinner=False)
def get_data_quality() -> pd.DataFrame:
    return load_data_quality()


def build_dashboard_context() -> DashboardContext:
    try:
        df = get_data()
        category_df = get_category_data()
        data_quality = get_data_quality()
    except FileNotFoundError:
        st.error("Brakuje aktualnych plików danych. Uruchom `python etl.py`, aby odtworzyć oba widoki i raport jakości.")
        st.stop()
        raise RuntimeError("Streamlit failed to stop after a data-loading error")

    exclusions = get_exclusions()
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    st.sidebar.title("Filtry globalne")
    year_range = st.sidebar.slider(
        "Zakres lat", min_year, max_year, (max(min_year, max_year - 9), max_year)
    )
    if year_range[0] == year_range[1]:
        reference_year = year_range[0]
        st.sidebar.caption(f"Rok referencyjny: **{reference_year}**")
    else:
        reference_year = st.sidebar.slider(
            "Rok referencyjny", year_range[0], year_range[1], year_range[1]
        )
    regions = st.sidebar.multiselect(
        "Regiony",
        options=sorted(df["region"].dropna().unique()),
        default=sorted(df["region"].dropna().unique()),
        format_func=region_label,
    )
    country_options = (
        df[df["region"].isin(regions)]["country_name"].drop_duplicates().sort_values().tolist()
        if regions
        else df["country_name"].drop_duplicates().sort_values().tolist()
    )
    countries = st.sidebar.multiselect("Kraje", options=country_options, default=country_options)
    df_filtered = df[
        df["year"].between(year_range[0], year_range[1])
        & df["region"].isin(regions)
        & df["country_name"].isin(countries)
    ].copy()
    category_filtered = category_df[
        category_df["year"].between(year_range[0], year_range[1])
        & category_df["region"].isin(regions)
        & category_df["country_name"].isin(countries)
    ].copy()
    scope_df = df[df["region"].isin(regions) & df["country_name"].isin(countries)].copy()
    latest = df_filtered[df_filtered["year"] == reference_year].copy()
    return DashboardContext(
        df=df,
        category_df=category_df,
        data_quality=data_quality,
        exclusions=exclusions,
        min_year=min_year,
        max_year=max_year,
        year_range=year_range,
        reference_year=reference_year,
        regions=regions,
        countries=countries,
        country_options=country_options,
        df_filtered=df_filtered,
        category_filtered=category_filtered,
        scope_df=scope_df,
        latest=latest,
    )
