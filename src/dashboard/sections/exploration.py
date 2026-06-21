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


def render_exploration(context: DashboardContext) -> None:
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
        "sec-cumulative",
        "5. Presja skumulowana 2020-2024",
        "Porównanie łącznego wzrostu cen żywności z łącznym wzrostem dochodów.",
        help_text=SECTION_HELP_PL["cumulative"],
    )
    if not has_full_cumulative_period:
        st.info(
            "Analiza wymaga, aby globalny filtr lat obejmował jednocześnie 2020 i 2024. "
            "Rozszerz zakres lat, aby policzyć porównanie skumulowane."
        )
    elif cumulative_2020_2024.empty:
        st.info("Brak kompletnych danych dla porównania 2020-2024 w aktualnie wybranych krajach.")
    else:
        top_cumulative = cumulative_2020_2024.iloc[0]
        cum_cols = st.columns(3)
        with cum_cols[0]:
            st.metric(
                "Największa skumulowana luka",
                top_cumulative["country_name"],
                f"{top_cumulative['cumulative_affordability_gap_pct']:+.1f} p.p.",
                help=metric_help("cumulative_affordability_gap_pct"),
            )
        with cum_cols[1]:
            st.metric(
                "Mediana wzrostu cen żywności",
                f"{cumulative_2020_2024['food_price_growth_2020_2024_pct'].median():+.1f}%",
            )
        with cum_cols[2]:
            st.metric(
                "Mediana wzrostu dochodu",
                f"{cumulative_2020_2024['income_growth_2020_2024_pct'].median():+.1f}%",
            )

        cum_chart, cum_table = st.columns([1.15, 0.85])
        with cum_chart:
            st.plotly_chart(cumulative_gap_bar(cumulative_2020_2024), width="stretch")
        with cum_table:
            cum_view = cumulative_2020_2024[
                [
                    "country_name",
                    "region",
                    "food_price_growth_2020_2024_pct",
                    "income_growth_2020_2024_pct",
                    "cumulative_affordability_gap_pct",
                ]
            ].head(12)
            st.dataframe(display_columns(cum_view), width="stretch", hide_index=True)
            st.caption("Dodatnia luka oznacza, że ceny żywności od 2020 r. wzrosły bardziej niż mediana dochodu.")

    section_anchor(
        "sec-categories",
        "6. Szczegółowe kategorie żywności",
        "Rzeczywiste obserwacje HICP w ziarnie kraj–rok–kategoria.",
        help_text=SECTION_HELP_PL["categories"],
    )
    category_options = sorted(category_filtered["food_category_name"].dropna().unique())
    if not category_options:
        st.info("Brak danych kategorii dla aktualnych filtrów.")
    else:
        selected_category = str(st.selectbox("Kategoria żywności", category_options))
        selected_category_df = category_filtered[
            category_filtered["food_category_name"] == selected_category
        ].copy()
        category_latest = selected_category_df[selected_category_df["year"] == reference_year]
        default_category_countries = (
            category_latest.nlargest(6, "category_food_inflation_pct")["country_name"].tolist()
        )
        category_countries = st.multiselect(
            "Kraje na trendzie kategorii",
            options=sorted(selected_category_df["country_name"].unique()),
            default=default_category_countries,
        )
        cat_a, cat_b = st.columns(2)
        with cat_a:
            st.plotly_chart(
                category_ranking_bar(category_filtered, reference_year, selected_category),
                width="stretch",
            )
        with cat_b:
            st.plotly_chart(
                histogram(
                    selected_category_df,
                    "category_food_inflation_pct",
                    reference_year,
                    f"Inflacja: {selected_category}",
                ),
                width="stretch",
            )
        if category_countries:
            st.plotly_chart(
                line_trend(
                    selected_category_df,
                    "category_food_inflation_pct",
                    category_countries,
                    f"Trend kategorii: {selected_category}",
                    "Inflacja kategorii (%)",
                ),
                width="stretch",
            )
        st.dataframe(
            display_columns(
                category_latest[
                    [
                        "country_name",
                        "region",
                        "food_category_code",
                        "category_food_inflation_pct",
                        "category_affordability_gap_pct",
                    ]
                ].sort_values("category_food_inflation_pct", ascending=False)
            ),
            width="stretch",
            hide_index=True,
        )

    section_anchor(
        "sec-trends",
        "7. Trendy inflacji żywności",
        "Szeregi czasowe dla wybranych krajów.",
        help_text=SECTION_HELP_PL["trends"],
    )
    default_trend = latest.dropna(subset=["food_inflation_pct"]).nlargest(6, "food_inflation_pct")["country_name"].tolist()
    trend_countries = st.multiselect(
        "Kraje na wykresie trendu",
        options=country_options,
        default=default_trend or country_options[:6],
    )
    if trend_countries:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                line_trend(df_filtered, "food_inflation_pct", trend_countries, "Trend inflacji żywności", "Inflacja żywności (%)"),
                width="stretch",
            )
        with col_b:
            st.plotly_chart(
                line_trend(
                    df_filtered,
                    "food_affordability_gap_pct",
                    trend_countries,
                    "Trend luki dostępności żywności",
                    "Luka dostępności żywności (p.p.)",
                ),
                width="stretch",
            )

    section_anchor(
        "sec-income",
        "8. Dochody i ceny żywności",
        "Relacja między poziomem dochodów a inflacją żywności.",
        help_text=SECTION_HELP_PL["income"],
    )
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.plotly_chart(scatter_income(df_filtered, reference_year), width="stretch")
    with col_b:
        selected_cols = [
            "country_name",
            "region",
            "food_inflation_pct",
            "headline_inflation_pct",
            "median_income_eur",
            "income_growth_pct",
            "food_affordability_gap_pct",
            "food_share_budget_pct",
        ]
        st.dataframe(
            display_columns(latest[selected_cols].sort_values("food_affordability_gap_pct", ascending=False)),
            width="stretch",
            hide_index=True,
        )

    section_anchor(
        "sec-distributions",
        "9. Rozkłady i obserwacje odstające",
        "Zróżnicowanie regionalne oraz anomalie IQR w roku referencyjnym.",
        help_text=SECTION_HELP_PL["distributions"],
    )
    dist_metric = st.selectbox(
        "Metryka rozkładu",
        ["food_affordability_gap_pct", "food_inflation_pct", "food_share_budget_pct", "meal_deprivation_pct"],
        format_func=metric_label,
    )
    st.caption(f"{metric_description(dist_metric)} **Interpretacja:** {metric_direction(dist_metric)}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(boxplot_region(df_filtered, dist_metric, reference_year, metric_label(dist_metric)), width="stretch")
    with col_b:
        st.plotly_chart(histogram(df_filtered, dist_metric, reference_year, metric_label(dist_metric)), width="stretch")

    outliers = iqr_outliers(latest, metric=dist_metric, n=10)
    if outliers.empty:
        st.info("Reguła 1,5×IQR nie wykryła obserwacji odstających w roku referencyjnym.")
    else:
        st.dataframe(
            display_columns(
                outliers[
                    [
                        "country_name",
                        "region",
                        "year",
                        dist_metric,
                        "iqr_distance",
                        "iqr_lower",
                        "iqr_upper",
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    st.caption("Anomalia IQR nie oznacza błędu danych; wskazuje wartość poza typowym zakresem przekroju krajów.")
