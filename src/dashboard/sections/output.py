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


def render_output(context: DashboardContext) -> None:
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
    has_full_cumulative_period = year_range[0] <= 2020 and year_range[1] >= 2024
    cumulative_2020_2024 = (
        cumulative_pressure_summary(scope_df) if has_full_cumulative_period else pd.DataFrame()
    )
    section_anchor(
        "sec-conclusions",
        "12. Wnioski i ograniczenia",
        "Automatycznie generowane obserwacje dla aktualnych filtrów oraz ograniczenia interpretacji.",
        help_text=SECTION_HELP_PL["conclusions"],
    )
    current_year = int(df_filtered["year"].max())
    current_latest = df_filtered[df_filtered["year"] == current_year].copy()
    current_notes = interpret_current_situation(current_latest, current_year)
    selected_notes = interpret_current_situation(latest, reference_year)
    cumulative_notes = interpret_cumulative_pressure(cumulative_2020_2024)

    current_complete = current_latest.dropna(subset=["food_affordability_gap_pct"])
    current_positive_count = int((current_complete["food_affordability_gap_pct"] > 0).sum()) if not current_complete.empty else 0
    current_n = len(current_complete)
    cumulative_median = (
        cumulative_2020_2024["cumulative_affordability_gap_pct"].median()
        if not cumulative_2020_2024.empty
        else np.nan
    )
    if current_n and current_positive_count <= max(3, int(current_n * 0.25)) and pd.notna(cumulative_median) and cumulative_median > 0:
        synthesis = (
            "Najważniejszy obraz analityczny jest dwuwarstwowy: bieżąca inflacja żywności w większości krajów przestała "
            "pogarszać lukę względem dochodów, ale część krajów nadal niesie skumulowany koszt szoku z lat 2020-2024."
        )
    elif current_n and current_positive_count <= max(3, int(current_n * 0.25)):
        synthesis = (
            "W ostatnim roku danych problem przesuwa się z samego tempa inflacji żywności na odporność gospodarstw: "
            "kluczowe stają się poziom dochodów, udział żywności w budżecie i deprywacja posiłku."
        )
    else:
        synthesis = (
            "W aktualnym przekroju presja pozostaje szeroka, dlatego główna interpretacja powinna łączyć bieżącą lukę "
            "cen-dochód z wrażliwością budżetową gospodarstw."
        )

    st.markdown("**Aktualna sytuacja w danych**")
    st.caption(
        f"Najnowszy rok w aktualnym filtrze to {current_year}. To najnowszy punkt dostępny dla wybranego zakresu, "
        "nie bieżący odczyt miesięczny ani prognoza dla obecnego roku."
    )
    st.markdown("\n".join(f"- {note}" for note in current_notes))

    if reference_year != current_year:
        st.markdown(f"**Wybrany rok referencyjny: {reference_year}**")
        st.markdown("\n".join(f"- {note}" for note in selected_notes))

    st.markdown("**Perspektywa po szoku 2020-2024**")
    st.markdown("\n".join(f"- {note}" for note in cumulative_notes))

    st.markdown("**Wniosek syntetyczny**")
    st.info(synthesis)

    with st.expander("Ograniczenia interpretacji", expanded=False):
        st.markdown(
            """
    - Dashboard pokazuje zależności opisowe, a nie dowodzi przyczynowości.
    - Dane są zagregowane do poziomu kraju, więc nie pokazują różnic między gospodarstwami domowymi wewnątrz kraju.
    - Część braków danych jest uzupełniana interpolacją, dlatego pojedyncze wartości należy traktować jako przybliżenia.
    - Porównanie 2020-2024 zależy od dostępności danych dla obu lat i nie opisuje pełnej ścieżki zmian między nimi.
    - Dane PPP o poziomie cen żywności mają słabsze pokrycie historyczne niż dane HICP, dochodowe i wydatkowe.
    """
        )

    if exclusions is not None and not exclusions.empty:
        with st.expander("Wykluczenia danych", expanded=False):
            st.dataframe(display_columns(exclusions), width="stretch", hide_index=True)

    section_anchor(
        "sec-export",
        "13. Eksport",
        "Dwa przefiltrowane ziarna danych do dalszej analizy.",
        help_text=SECTION_HELP_PL["export"],
    )
    export_cols = [
        "country_name",
        "country_code",
        "region",
        "year",
        *KEY_METRICS,
        *[col for col in df_filtered.columns if col.endswith("_imputed")],
    ]
    export_df = display_columns(df_filtered[export_cols].sort_values(["year", "country_name"]))
    st.download_button(
        "Pobierz CSV kraj–rok",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"europe_food_affordability_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
    )
    st.dataframe(export_df, width="stretch", hide_index=True)

    category_export_cols = [
        "country_name",
        "country_code",
        "region",
        "year",
        "food_category_code",
        "food_category_name",
        "category_food_inflation_pct",
        "category_affordability_gap_pct",
        "category_affordability_gap_pct_imputed",
        "income_growth_pct",
        "income_growth_pct_imputed",
        "headline_inflation_pct_imputed",
        "median_income_eur_imputed",
        "food_price_level_index_imputed",
        "food_share_budget_pct_imputed",
        "meal_deprivation_pct_imputed",
    ]
    category_export = display_columns(
        category_filtered[category_export_cols].sort_values(
            ["year", "food_category_code", "country_name"]
        )
    )
    st.download_button(
        "Pobierz CSV kraj–rok–kategoria",
        category_export.to_csv(index=False).encode("utf-8"),
        file_name=f"europe_food_categories_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
    )
    st.dataframe(category_export.head(500), width="stretch", hide_index=True)
    st.caption("Podgląd widoku kategorii jest ograniczony do 500 rekordów; eksport zawiera cały przefiltrowany zbiór.")
