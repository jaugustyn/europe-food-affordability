"""Streamlit dashboard section renderer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.state import DashboardContext
from src.dashboard.support import (
    EFFECT_SIZE_LABELS_PL,
    MAP_METRICS,
    METRIC_REFERENCE,
    REGION_LABELS_PL,
    SECTION_HELP_PL,
    cumulative_pressure_summary,
    descriptive_statistics,
    display_columns,
    fmt_p,
    fmt_value,
    interpret_cumulative_pressure,
    interpret_current_situation,
    metric_color_range,
    metric_description,
    metric_direction,
    metric_help,
    metric_label,
    metric_unit,
    pressure_driver_notes,
    section_anchor,
)
from src.metrics import KEY_METRICS, METRICS, fmt
from src.pca_analysis import fit_pca
from src.stats_tests import (
    bootstrap_region_means,
    correlation_matrix,
    iqr_outliers,
    mask_imputed_values,
    pvalue_matrix,
    sample_size_matrix,
    select_region_test,
)
from src.viz import (
    bar_with_ci,
    boxplot_region,
    choropleth,
    category_ranking_bar,
    cumulative_gap_bar,
    driver_bar,
    heatmap_corr,
    heatmap_country_year,
    histogram,
    line_trend,
    pca_biplot,
    pca_scree_plot,
    scatter_income,
)


def render_overview(context: DashboardContext) -> None:
    df = context.df
    category_df = context.category_df
    data_quality = context.data_quality
    exclusions = context.exclusions
    min_year = context.min_year
    max_year = context.max_year
    year_range = context.year_range
    reference_year = context.reference_year
    regions = context.regions
    countries = context.countries
    country_options = context.country_options
    df_filtered = context.df_filtered
    category_filtered = context.category_filtered
    scope_df = context.scope_df
    latest = context.latest
    section_anchor(
        "sec-kpi",
        "1. KPI",
        "Najważniejsze wskaźniki dostępności żywności dla roku referencyjnego.",
        help_text=SECTION_HELP_PL["kpi"],
    )
    kpi_cols = st.columns(4)

    top_gap = latest.dropna(subset=["food_affordability_gap_pct"]).sort_values("food_affordability_gap_pct", ascending=False).head(1)
    avg_food_infl = latest["food_inflation_pct"].mean()
    avg_headline = latest["headline_inflation_pct"].mean()
    top_share = latest.dropna(subset=["food_share_budget_pct"]).sort_values("food_share_budget_pct", ascending=False).head(1)
    median_gap = latest["food_affordability_gap_pct"].median()

    with kpi_cols[0]:
        if top_gap.empty:
            st.metric("Największa luka dostępności", "n/a")
        else:
            row = top_gap.iloc[0]
            st.metric(
                "Największa luka dostępności",
                row["country_name"],
                fmt(row["food_affordability_gap_pct"], "food_affordability_gap_pct"),
                help=metric_help("food_affordability_gap_pct"),
            )
    with kpi_cols[1]:
        st.metric(
            "Średnia inflacja żywności",
            fmt(avg_food_infl, "food_inflation_pct"),
            delta=fmt(avg_food_infl - avg_headline, "headline_inflation_pct"),
        )
    with kpi_cols[2]:
        if top_share.empty:
            st.metric("Najwyższy udział żywności w wydatkach", "n/a")
        else:
            row = top_share.iloc[0]
            st.metric(
                "Najwyższy udział żywności w wydatkach",
                row["country_name"],
                fmt(row["food_share_budget_pct"], "food_share_budget_pct"),
            )
    with kpi_cols[3]:
        st.metric(
            "Mediana luki dostępności",
            fmt(median_gap, "food_affordability_gap_pct"),
            help=metric_help("food_affordability_gap_pct"),
        )

    with st.expander("Interpretacja metryk", expanded=True):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metryka": metric_label(key),
                        "Jednostka": metric_unit(key),
                        "Opis": metric_description(key),
                        "Interpretacja": metric_direction(key),
                    }
                    for key in METRIC_REFERENCE
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    section_anchor(
        "sec-quality",
        "2. Struktura i jakość danych",
        "Dwa ziarna danych, kompletność ETL i statystyki opisowe.",
        help_text=SECTION_HELP_PL["quality"],
    )
    quality_cols = st.columns(4)
    quality_cols[0].metric("Rekordy kraj–rok", f"{len(df_filtered):,}".replace(",", " "))
    quality_cols[1].metric("Rekordy kraj–rok–kategoria", f"{len(category_filtered):,}".replace(",", " "))
    quality_cols[2].metric("Kraje", df_filtered["country_code"].nunique())
    quality_cols[3].metric("Kategorie", category_filtered["food_category_code"].nunique())
    st.caption(
        "Klucz widoku głównego: `country_code + year`. Klucz widoku kategorii: "
        "`country_code + year + food_category_code`. Statystyki poniżej reagują na filtry globalne."
    )
    raw_category_quality = data_quality[
        (data_quality["view"] == "country_year_category")
        & (data_quality["stage"] == "raw_grid_before_etl")
        & (data_quality["column"] == "category_food_inflation_pct")
    ]
    if not raw_category_quality.empty:
        raw_category_row = raw_category_quality.iloc[0]
        raw_category_count = int(raw_category_row["row_count"])
        raw_category_missing = int(raw_category_row["missing_count"])
        st.caption(
            f"Przepływ kategorii: surowa siatka **{raw_category_count:,}** rekordów → "
            f"**{raw_category_count - raw_category_missing:,}** zaobserwowanych HICP "
            f"({raw_category_missing} braków odrzuconych bez imputacji) → "
            f"**{len(category_df):,}** rekordów w końcowym widoku po złączeniu z kontekstem kraj–rok."
        )

    tab_quality, tab_country_stats, tab_category_stats = st.tabs(
        ["Braki i typy", "Statystyki kraj–rok", "Statystyki kategorii"]
    )
    with tab_quality:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Typy kolumn: kraj–rok**")
            st.dataframe(
                pd.DataFrame({"Kolumna": df_filtered.columns, "Typ": df_filtered.dtypes.astype(str)}),
                width="stretch",
                hide_index=True,
            )
        with col_b:
            st.markdown("**Typy kolumn: kraj–rok–kategoria**")
            st.dataframe(
                pd.DataFrame({"Kolumna": category_filtered.columns, "Typ": category_filtered.dtypes.astype(str)}),
                width="stretch",
                hide_index=True,
            )
        quality_view = data_quality.copy()
        quality_view["missing_pct"] = quality_view["missing_pct"].round(2)
        st.markdown("**Globalny audyt ETL: braki i imputacje**")
        st.caption(
            "Ta tabela opisuje pełne artefakty ETL i celowo nie reaguje na filtry dashboardu. "
            "Etap `raw_grid_before_etl` zachowuje również brakujące wartości HICP kategorii."
        )
        st.dataframe(quality_view, width="stretch", hide_index=True)
    with tab_country_stats:
        st.dataframe(descriptive_statistics(df_filtered), width="stretch", hide_index=True)
    with tab_category_stats:
        category_stat_cols = [
            "category_food_inflation_pct",
            "category_affordability_gap_pct",
        ]
        st.dataframe(
            descriptive_statistics(category_filtered[category_stat_cols]),
            width="stretch",
            hide_index=True,
        )

    section_anchor(
        "sec-drivers",
        "3. Diagnoza kraju",
        "Co konkretnie napędza presję cen żywności w wybranym kraju.",
        help_text=SECTION_HELP_PL["drivers"],
    )
    driver_options = latest["country_name"].drop_duplicates().sort_values().tolist()
    default_driver = top_gap.iloc[0]["country_name"] if not top_gap.empty else driver_options[0]
    driver_country = str(st.selectbox(
        "Kraj do diagnozy",
        options=driver_options,
        index=driver_options.index(default_driver) if default_driver in driver_options else 0,
    ))
    driver_row = latest[latest["country_name"] == driver_country].iloc[0]
    driver_cols = st.columns(5)
    driver_metrics = [
        ("Inflacja żywności", "food_inflation_pct"),
        ("Wzrost dochodu", "income_growth_pct"),
        ("Luka dostępności", "food_affordability_gap_pct"),
        ("Udział żywności w wydatkach", "food_share_budget_pct"),
        ("Brak pełnowartościowego posiłku", "meal_deprivation_pct"),
    ]
    for col, (label, metric) in zip(driver_cols, driver_metrics):
        with col:
            st.metric(label, fmt_value(driver_row.get(metric), metric), help=metric_help(metric))

    driver_chart, driver_notes_col = st.columns([1.15, 0.85])
    with driver_chart:
        st.plotly_chart(driver_bar(driver_row, driver_country, reference_year), width="stretch")
    with driver_notes_col:
        st.markdown("**Interpretacja**")
        st.markdown("\n".join(f"- {note}" for note in pressure_driver_notes(driver_row, latest)))

    section_anchor(
        "sec-map",
        "4. Mapa presji cenowej",
        "Mapa porównuje kraje w wybranym roku. Braki danych są wyszarzone.",
        help_text=SECTION_HELP_PL["map"],
    )

    map_controls, map_view = st.columns([0.26, 0.74])
    with map_controls:
        if year_range[0] == year_range[1]:
            map_year = reference_year
            st.caption(f"Rok mapy: **{map_year}**")
        else:
            map_year = st.slider(
                "Rok na mapie",
                year_range[0],
                year_range[1],
                reference_year,
                key="map_year",
                help="Zmienia tylko rok mapy. Pozostałe sekcje nadal używają globalnego roku referencyjnego.",
            )
        map_metric = st.selectbox(
            "Metryka na mapie",
            MAP_METRICS,
            format_func=metric_label,
            help="Kolor mapy może przedstawiać presję cenową, inflację, udział żywności w budżecie, poziom cen albo deprywację posiłku.",
        )
        st.caption(f"{metric_description(map_metric)} **Interpretacja:** {metric_direction(map_metric)}")
        use_full_map_range = st.toggle(
            "Pełny zakres min-max",
            value=False,
            help=(
                "Domyślnie mapa używa stałej skali 5-95 percentyl, żeby pojedyncze skrajne wartości "
                "nie spłaszczały kolorów. Włączenie opcji obejmuje wszystkie wartości w skali min-max."
            ),
        )

    scale_frame = df[df["region"].isin(regions) & df["country_name"].isin(countries)]
    map_color_range = metric_color_range(scale_frame, map_metric, robust=not use_full_map_range)
    with map_view:
        st.plotly_chart(
            choropleth(
                df_filtered,
                map_metric,
                map_year,
                metric_label(map_metric),
                color_scale=METRICS[map_metric]["color_scale"],
                color_range=map_color_range,
            ),
            width="stretch",
        )

    if map_color_range is not None:
        scale_note = (
            "Zakres 5-95 percentyl ogranicza wpływ obserwacji skrajnych wyłącznie na kolorystykę."
            if not use_full_map_range
            else "Zakres min-max obejmuje wszystkie obserwacje po aktualnych filtrach krajów i regionów."
        )
        st.caption(
            f"Stała skala kolorów: od {fmt(map_color_range[0], map_metric)} do "
            f"{fmt(map_color_range[1], map_metric)}. Ten sam odcień oznacza tę samą wartość przy zmianie roku. "
            f"{scale_note}"
        )
