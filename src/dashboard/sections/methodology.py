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


def render_methodology(context: DashboardContext) -> None:
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
        "sec-correlations",
        "10. Korelacje i PCA",
        "Korelacje przekrojowe dla roku referencyjnego oraz redukcja wymiarowości.",
        help_text=SECTION_HELP_PL["correlations"],
    )
    corr_metrics = [
        "food_inflation_pct",
        "headline_inflation_pct",
        "median_income_eur",
        "income_growth_pct",
        "food_share_budget_pct",
        "food_price_level_index",
        "meal_deprivation_pct",
    ]

    corr_method = st.selectbox(
        "Metoda korelacji",
        ["spearman", "pearson"],
        format_func=lambda method: "Pearson - zależność liniowa" if method == "pearson" else "Spearman - zgodność rang",
        help=(
            "Pearson sprawdza prostą zależność liniową między wartościami. "
            "Spearman sprawdza, czy kraje układają się podobnie w rankingach, nawet gdy zależność nie jest liniowa."
        ),
    )
    corr_method_label = "Pearson" if corr_method == "pearson" else "Spearman"
    st.caption(f"Aktywna metoda korelacji: **{corr_method_label}**.")
    corr_input, corr_imputed_counts = mask_imputed_values(latest, corr_metrics)

    col_a, col_b = st.columns(2)
    with col_a:
        corr = correlation_matrix(corr_input, corr_metrics, method=corr_method)
        corr.index = [metric_label(idx) for idx in corr.index]
        corr.columns = [metric_label(col) for col in corr.columns]
        st.plotly_chart(heatmap_corr(corr, f"Korelacje metryk ({corr_method_label})"), width="stretch")
    with col_b:
        pvals = pvalue_matrix(corr_input, corr_metrics, method=corr_method)
        pvals.index = [metric_label(idx) for idx in pvals.index]
        pvals.columns = [metric_label(col) for col in pvals.columns]
        st.dataframe(pvals.map(fmt_p), width="stretch")
        pair_n = sample_size_matrix(corr_input, corr_metrics)
        pair_n.index = [metric_label(idx) for idx in pair_n.index]
        pair_n.columns = [metric_label(col) for col in pair_n.columns]
        with st.expander("Liczebności par korelacyjnych", expanded=False):
            st.dataframe(pair_n, width="stretch")
        st.caption(
            "Macierz korelacji i p-value są liczone parami, na wspólnych kompletnych obserwacjach dla danej pary zmiennych. "
            "Tabela pokazuje p-value po korekcie Holma. Wartości poniżej 0,05 oznaczają, że zależność jest mało "
            "prawdopodobna jako czysty przypadek przy wielu porównaniach."
        )
    st.caption(
        f"Korelacje wykorzystują wyłącznie przekrój {reference_year}: każdy kraj występuje raz. "
        "Wartości imputowane są traktowane jak braki i nie uczestniczą we wnioskowaniu. "
        "Korelacja nie oznacza przyczynowości."
    )
    masked_corr_metrics = {
        metric_label(metric): count
        for metric, count in corr_imputed_counts.items()
        if count > 0
    }
    if masked_corr_metrics:
        masked_description = ", ".join(
            f"{label}: {count}" for label, count in masked_corr_metrics.items()
        )
        st.info(
            "Z korelacji wyłączono imputowane komórki w roku referencyjnym — "
            f"{masked_description}. Liczebności każdej pary są widoczne w tabeli N."
        )

    st.plotly_chart(heatmap_country_year(df_filtered, "food_share_budget_pct", "Udział żywności w wydatkach według kraju i roku"), width="stretch")

    try:
        pca_result = fit_pca(df_filtered)
        pca_col_a, pca_col_b = st.columns([1.35, 0.65])
        with pca_col_a:
            st.plotly_chart(
                pca_biplot(pca_result["scores"], pca_result["loadings"], pca_result["explained_variance"]),
                width="stretch",
            )
        with pca_col_b:
            st.plotly_chart(pca_scree_plot(pca_result["explained_variance_full"]), width="stretch")
        st.caption(
            f"PCA wykorzystuje {pca_result['n']} kompletnych obserwacji. "
            f"Wykluczono {pca_result['imputed_rows_excluded']} rekordów zawierających "
            "co najmniej jedną imputowaną cechę. "
            f"PC1 i PC2 wyjaśniają łącznie {sum(pca_result['explained_variance']) * 100:.1f}% wariancji. "
            f"Wybrano {pca_result['selected_k']} komponent(y), ponieważ jest to najmniejsze k osiągające "
            f"co najmniej {pca_result['variance_threshold'] * 100:.0f}% skumulowanej wariancji. "
            "Liczba obserwacji PCA może być niższa od pełnego zbioru, ponieważ wszystkie cechy użyte w PCA muszą być kompletne."
        )
        with st.expander("Dlaczego nie dodano t-SNE, UMAP, LDA ani autoenkoderów?", expanded=False):
            st.markdown(
                """
    - t-SNE i UMAP są efektowne wizualnie, ale przy tym zbiorze mogłyby sugerować niestabilne klastry.
    - LDA wymaga zewnętrznej etykiety klasy i celu klasyfikacyjnego, którego ta analiza nie definiuje.
    - Autoenkodery są nieadekwatne do małego, tablicowego zbioru makroekonomicznego.
    """
            )
    except ValueError as exc:
        st.info(str(exc))

    section_anchor(
        "sec-tests",
        "11. Testy statystyczne",
        "Jeden test główny wybierany na podstawie formalnej diagnostyki założeń.",
        help_text=SECTION_HELP_PL["tests"],
    )
    test_metric = st.selectbox(
        "Metryka testowana",
        ["food_affordability_gap_pct", "food_inflation_pct", "food_share_budget_pct", "meal_deprivation_pct"],
        format_func=metric_label,
    )
    st.caption(f"{metric_description(test_metric)} **Interpretacja:** {metric_direction(test_metric)}")
    test_df_raw = df_filtered[df_filtered["year"] == reference_year].copy()
    test_df, test_imputed_counts = mask_imputed_values(test_df_raw, [test_metric])
    test_imputed_count = test_imputed_counts[test_metric]
    st.caption(
        "Wnioskowanie wykorzystuje wyłącznie wartości zaobserwowane: "
        f"wykluczono {test_imputed_count} imputowanych obserwacji metryki."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        try:
            stats_result = select_region_test(test_df, test_metric, alpha=0.05)
            method_label = stats_result["method_label"]
            effect_label = EFFECT_SIZE_LABELS_PL.get(stats_result["effect_label"], stats_result["effect_label"])
            st.markdown(
                f"**H₀:** dla `{metric_label(test_metric)}` {stats_result['hypothesis_null']}.  \n"
                f"**H₁:** dla `{metric_label(test_metric)}` {stats_result['hypothesis_alternative']}."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Test główny": method_label, "Statystyka": stats_result["statistic"], "p-value": fmt_p(stats_result["p_value"])},
                        {"Test główny": "Levene (diagnostyka)", "Statystyka": stats_result["levene_stat"], "p-value": fmt_p(stats_result["levene_p"])},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            selection_reason = (
                "ANOVA: we wszystkich grupach brak podstaw do odrzucenia normalności i test Levene’a nie odrzuca równości wariancji."
                if stats_result["method"] == "anova"
                else "Kruskal–Wallis: co najmniej jedno założenie ANOVA nie zostało spełnione."
            )
            st.caption(selection_reason)
            decision = (
                f"Odrzucamy H₀ przy α=0,05 (p={stats_result['p_value']:.4f})."
                if stats_result["reject_h0"]
                else f"Brak podstaw do odrzucenia H₀ przy α=0,05 (p={stats_result['p_value']:.4f})."
            )
            st.success(decision) if stats_result["reject_h0"] else st.info(decision)
            st.write(
                f"Wielkość efektu ({stats_result['effect_name']}): "
                f"**{stats_result['effect_size']:.3f}** — efekt {effect_label}."
            )

            shapiro_diag = stats_result["diagnostics"]
            if not shapiro_diag.empty:
                shapiro_view = shapiro_diag.copy()
                shapiro_view["region"] = shapiro_view["region"].map(REGION_LABELS_PL).fillna(shapiro_view["region"])
                shapiro_view["p_value"] = shapiro_view["p_value"].map(fmt_p)
                shapiro_view["normal_05"] = shapiro_view["normal_05"].map({True: "tak", False: "nie"}).fillna("brak danych")
                shapiro_view = shapiro_view.rename(
                    columns={
                        "region": "Region",
                        "n": "N",
                        "w": "Statystyka W",
                        "p_value": "p-value",
                        "normal_05": "Brak podstaw do odrzucenia normalności (0,05)",
                        "note": "Uwagi",
                    }
                )
                st.dataframe(shapiro_view, width="stretch", hide_index=True)
            posthoc = stats_result["posthoc"].copy()
            if stats_result["reject_h0"] and not posthoc.empty:
                for col in ["group_A", "group_B", "group1", "group2"]:
                    if col in posthoc.columns:
                        posthoc[col] = posthoc[col].map(REGION_LABELS_PL).fillna(posthoc[col])
                for p_col in ["p_adj", "p-adj"]:
                    if p_col in posthoc.columns:
                        posthoc[p_col] = posthoc[p_col].map(fmt_p)
                if "effect_size" in posthoc.columns:
                    posthoc["effect_size"] = posthoc["effect_size"].map(EFFECT_SIZE_LABELS_PL).fillna(posthoc["effect_size"])
                st.markdown("**Post-hoc wykonany po istotnym teście globalnym**")
                st.dataframe(posthoc, width="stretch", hide_index=True)
            elif not stats_result["reject_h0"]:
                st.caption("Post-hoc pominięto, ponieważ test globalny nie był istotny.")

            group_summary = stats_result["group_summary"]
            if len(group_summary) >= 2:
                high, low = group_summary.index[0], group_summary.index[-1]
                summary_name = stats_result["group_statistic_name"]
                st.caption(
                    f"Praktycznie: najwyższa {summary_name} występuje w regionie "
                    f"{REGION_LABELS_PL.get(high, high)}, a najniższa w regionie "
                    f"{REGION_LABELS_PL.get(low, low)}. "
                    f"{stats_result['interpretation']}"
                )
        except ValueError as exc:
            st.info(str(exc))

    with col_b:
        ci = bootstrap_region_means(test_df, test_metric)
        if ci.empty:
            st.info(
                "Przedziały bootstrapowe pominięto: żaden region nie ma co najmniej "
                "3 obserwowanych wartości tej metryki."
            )
        else:
            st.plotly_chart(
                bar_with_ci(ci, f"Średnia regionalna i 95% CI · {metric_label(test_metric)}"),
                width="stretch",
            )
