from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_data, load_exclusions
from src.forecast import forecast_food_inflation
from src.metrics import KEY_METRICS, METRICS, fmt
from src.pca_analysis import fit_pca
from src.regression import DEFAULT_FEATURES, fit_fpi_regression, fit_panel_fixed_effects
from src.stats_tests import (
    bootstrap_region_means,
    chi_square_high_pressure,
    correlation_matrix,
    pvalue_matrix,
    region_anova,
    top_outliers,
)
from src.viz import (
    bar_with_ci,
    boxplot_region,
    choropleth,
    forecast_plot,
    heatmap_corr,
    heatmap_country_year,
    histogram,
    line_trend,
    pca_biplot,
    residuals_plot,
    scatter_income,
)


st.set_page_config(
    page_title="Europe Food Affordability",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLUMN_LABELS = {
    "country_name": "Country",
    "country_code": "Country code",
    "iso3": "ISO-3",
    "region": "Region",
    "year": "Year",
    **{key: meta.label for key, meta in METRICS.items()},
}


def display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={k: v for k, v in COLUMN_LABELS.items() if k in frame.columns})


def metric_label(metric: str) -> str:
    return METRICS.get(metric).label if metric in METRICS else metric


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def section_anchor(anchor: str, title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<h2 id='{anchor}'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner=False)
def get_exclusions() -> pd.DataFrame:
    return load_exclusions()


try:
    df = get_data()
except FileNotFoundError:
    st.error("Processed data file was not found. Run `python etl.py` first.")
    st.stop()

exclusions = get_exclusions()
min_year, max_year = int(df["year"].min()), int(df["year"].max())

st.sidebar.title("Filters")
year_range = st.sidebar.slider("Year range", min_year, max_year, (max(min_year, max_year - 9), max_year))
map_year = st.sidebar.slider("Reference year", year_range[0], year_range[1], year_range[1])
regions = st.sidebar.multiselect(
    "Regions",
    options=sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique()),
)

country_options = (
    df[df["region"].isin(regions)]["country_name"].drop_duplicates().sort_values().tolist()
    if regions
    else df["country_name"].drop_duplicates().sort_values().tolist()
)
countries = st.sidebar.multiselect(
    "Countries",
    options=country_options,
    default=country_options,
)

df_filtered = df[
    df["year"].between(year_range[0], year_range[1])
    & df["region"].isin(regions)
    & df["country_name"].isin(countries)
].copy()
latest = df_filtered[df_filtered["year"] == map_year].copy()

st.title("Europe Food Affordability")
st.caption(
    "Dashboard for comparing food inflation pressure, household income and food affordability "
    "across European countries using Eurostat data."
)

st.markdown(
    """
<div class="toc">
<a href="#sec-kpi">1. KPI</a> ·
<a href="#sec-map">2. Map</a> ·
<a href="#sec-trends">3. Trends</a> ·
<a href="#sec-income">4. Income</a> ·
<a href="#sec-distributions">5. Distribution</a> ·
<a href="#sec-correlations">6. Correlations</a> ·
<a href="#sec-tests">7. Tests</a> ·
<a href="#sec-prediction">8. Prediction</a> ·
<a href="#sec-conclusions">9. Conclusions</a> ·
<a href="#sec-export">10. Export</a>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
    h2 {margin-top: 2rem;}
    .toc {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 1rem 0 1.2rem 0;
        line-height: 1.8;
    }
    .toc a {text-decoration: none; color: #0f172a; font-weight: 600;}
    div[data-testid="stMetricValue"] {font-size: 1.75rem;}
</style>
""",
    unsafe_allow_html=True,
)

if df_filtered.empty or latest.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

section_anchor("sec-kpi", "1. KPI", "Headline affordability indicators for the selected reference year.")
kpi_cols = st.columns(4)

top_fpi = latest.dropna(subset=["fpi"]).sort_values("fpi", ascending=False).head(1)
avg_food_infl = latest["food_inflation_pct"].mean()
avg_headline = latest["headline_inflation_pct"].mean()
top_share = latest.dropna(subset=["food_share_budget_pct"]).sort_values("food_share_budget_pct", ascending=False).head(1)
median_fpi = latest["fpi"].median()

with kpi_cols[0]:
    if top_fpi.empty:
        st.metric("Highest pressure", "n/a")
    else:
        row = top_fpi.iloc[0]
        st.metric("Highest pressure", row["country_name"], fmt(row["fpi"], "fpi"))
with kpi_cols[1]:
    st.metric("Average food inflation", fmt(avg_food_infl, "food_inflation_pct"), delta=fmt(avg_food_infl - avg_headline, "headline_inflation_pct"))
with kpi_cols[2]:
    if top_share.empty:
        st.metric("Highest food spending share", "n/a")
    else:
        row = top_share.iloc[0]
        st.metric("Highest food spending share", row["country_name"], fmt(row["food_share_budget_pct"], "food_share_budget_pct"))
with kpi_cols[3]:
    st.metric("Median FPI", fmt(median_fpi, "fpi"), help=METRICS["fpi"].description)

with st.expander("Metric definitions", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": meta.label, "Unit": meta.unit, "Description": meta.description}
                for key, meta in METRICS.items()
                if key in KEY_METRICS
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

section_anchor("sec-map", "2. Spatial Pattern", "Country-level map of food affordability pressure.")
map_metric = st.selectbox(
    "Map metric",
    ["fpi", "food_inflation_pct", "food_share_budget_pct", "food_price_level_index", "meal_deprivation_pct"],
    format_func=metric_label,
)
st.plotly_chart(
    choropleth(df_filtered, map_metric, map_year, metric_label(map_metric)),
    use_container_width=True,
)

section_anchor("sec-trends", "3. Food Inflation Trends", "Time series for selected countries.")
default_trend = latest.dropna(subset=["food_inflation_pct"]).nlargest(6, "food_inflation_pct")["country_name"].tolist()
trend_countries = st.multiselect(
    "Countries in trend chart",
    options=country_options,
    default=default_trend or country_options[:6],
)
if trend_countries:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            line_trend(df_filtered, "food_inflation_pct", trend_countries, "Food inflation trend", "Food inflation (%)"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            line_trend(df_filtered, "fpi", trend_countries, "Food Pressure Index trend", "Food Pressure Index"),
            use_container_width=True,
        )

section_anchor("sec-income", "4. Income and Food Prices", "Relationship between income levels and food inflation.")
col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.plotly_chart(scatter_income(df_filtered, map_year), use_container_width=True)
with col_b:
    selected_cols = [
        "country_name",
        "region",
        "food_inflation_pct",
        "headline_inflation_pct",
        "median_income_eur",
        "income_growth_pct",
        "food_share_budget_pct",
        "fpi",
    ]
    st.dataframe(
        display_columns(latest[selected_cols].sort_values("fpi", ascending=False)),
        use_container_width=True,
        hide_index=True,
    )

section_anchor("sec-distributions", "5. Distributions and Outliers", "Regional spread and extreme country-year observations.")
dist_metric = st.selectbox(
    "Distribution metric",
    ["fpi", "food_inflation_pct", "food_share_budget_pct", "meal_deprivation_pct"],
    format_func=metric_label,
)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(boxplot_region(df_filtered, dist_metric, map_year, metric_label(dist_metric)), use_container_width=True)
with col_b:
    st.plotly_chart(histogram(df_filtered, dist_metric, map_year, metric_label(dist_metric)), use_container_width=True)

outliers = top_outliers(df_filtered, metric=dist_metric, n=10)
st.dataframe(
    display_columns(outliers[["country_name", "region", "year", dist_metric, "zscore"]]),
    use_container_width=True,
    hide_index=True,
)

section_anchor("sec-correlations", "6. Correlations and Structure", "Metric relationships, p-values, heatmap and PCA.")
corr_metrics = [
    "fpi",
    "food_inflation_pct",
    "headline_inflation_pct",
    "median_income_eur",
    "income_growth_pct",
    "food_share_budget_pct",
    "food_price_level_index",
    "meal_deprivation_pct",
]

col_a, col_b = st.columns(2)
with col_a:
    corr = correlation_matrix(df_filtered, corr_metrics)
    corr.index = [metric_label(idx) for idx in corr.index]
    corr.columns = [metric_label(col) for col in corr.columns]
    st.plotly_chart(heatmap_corr(corr), use_container_width=True)
with col_b:
    pvals = pvalue_matrix(df_filtered, corr_metrics)
    pvals.index = [metric_label(idx) for idx in pvals.index]
    pvals.columns = [metric_label(col) for col in pvals.columns]
    st.dataframe(pvals.map(fmt_p), use_container_width=True)

st.plotly_chart(heatmap_country_year(df_filtered, "food_share_budget_pct", "Food Spending Share by Country and Year"), use_container_width=True)

try:
    pca_result = fit_pca(df_filtered)
    st.plotly_chart(
        pca_biplot(pca_result["scores"], pca_result["loadings"], pca_result["explained_variance"]),
        use_container_width=True,
    )
    st.caption(
        f"PCA uses {pca_result['n']} complete observations. "
        f"PC1 and PC2 explain {sum(pca_result['explained_variance']) * 100:.1f}% of variance."
    )
except ValueError as exc:
    st.info(str(exc))

section_anchor("sec-tests", "7. Statistical Tests", "Regional differences and uncertainty estimates.")
test_metric = st.selectbox("Test metric", ["fpi", "food_inflation_pct", "food_share_budget_pct"], format_func=metric_label)
test_df = df_filtered[df_filtered["year"] == map_year].copy()

col_a, col_b = st.columns(2)
with col_a:
    try:
        stats_result = region_anova(test_df, test_metric)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Test": "One-way ANOVA", "Statistic": stats_result.anova_stat, "p-value": fmt_p(stats_result.anova_p)},
                    {"Test": "Kruskal-Wallis", "Statistic": stats_result.kruskal_stat, "p-value": fmt_p(stats_result.kruskal_p)},
                    {"Test": "Levene variance test", "Statistic": stats_result.levene_stat, "p-value": fmt_p(stats_result.levene_p)},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if not stats_result.pairwise.empty:
            pairwise = stats_result.pairwise.copy()
            pairwise["p_adj"] = pairwise["p_adj"].map(fmt_p)
            st.dataframe(pairwise, use_container_width=True, hide_index=True)
    except ValueError as exc:
        st.info(str(exc))

with col_b:
    ci = bootstrap_region_means(test_df, test_metric)
    st.plotly_chart(bar_with_ci(ci, f"Regional Mean and 95% CI · {metric_label(test_metric)}"), use_container_width=True)

chi2 = chi_square_high_pressure(test_df, value_col=test_metric)
if chi2:
    st.caption(f"Chi-square test for high-pressure countries by region: p-value {fmt_p(chi2['p_value'])}.")
    st.dataframe(display_columns(chi2["table"].reset_index()), use_container_width=True, hide_index=True)

section_anchor("sec-prediction", "8. Prediction", "Regression, panel fixed effects and short-term forecasting.")
tab_reg, tab_panel, tab_forecast = st.tabs(["Regression", "Panel fixed effects", "Forecast"])

with tab_reg:
    model_type = st.radio("Model", ["ridge", "random_forest"], format_func=lambda v: "Ridge" if v == "ridge" else "Random Forest", horizontal=True)
    features = st.multiselect(
        "Predictors",
        options=[
            "median_income_eur",
            "food_share_budget_pct",
            "headline_inflation_pct",
            "meal_deprivation_pct",
            "food_price_level_index",
        ],
        default=DEFAULT_FEATURES,
        format_func=metric_label,
    )
    if "income_growth_pct" in features:
        st.warning("Income growth is not available as a same-year predictor because it is part of the FPI formula.")
    try:
        reg = fit_fpi_regression(df_filtered, features=features, model_type=model_type)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Test R²", f"{reg['r2_test']:.3f}")
            st.metric("Test MAE", f"{reg['mae_test']:.2f}")
            imp = reg["feature_importance"].copy()
            imp["feature"] = imp["feature"].map(metric_label)
            st.dataframe(imp, use_container_width=True, hide_index=True)
        with col_b:
            predictions = reg["predictions"].merge(df[["country_code", "year", "region"]], on=["country_code", "year"], how="left")
            st.plotly_chart(residuals_plot(predictions), use_container_width=True)
            st.dataframe(display_columns(predictions.head(15)), use_container_width=True, hide_index=True)
    except ValueError as exc:
        st.info(str(exc))

with tab_panel:
    try:
        panel = fit_panel_fixed_effects(df_filtered, features=DEFAULT_FEATURES)
        st.metric("Adjusted R²", f"{panel['adj_r2']:.3f}", help=panel["formula"])
        coef = panel["coefficients"].copy()
        coef["term"] = coef["term"].map(metric_label)
        coef["p_value"] = coef["p_value"].map(fmt_p)
        st.dataframe(coef, use_container_width=True, hide_index=True)
        with st.expander("Full statsmodels output", expanded=False):
            st.code(panel["summary_text"], language="text")
    except ValueError as exc:
        st.info(str(exc))

with tab_forecast:
    forecast_country = st.selectbox("Country", options=country_options, index=country_options.index("Poland") if "Poland" in country_options else 0)
    periods = st.slider("Forecast horizon", 1, 5, 3)
    try:
        forecast_result = forecast_food_inflation(df, forecast_country, periods=periods)
        st.plotly_chart(
            forecast_plot(forecast_result["history"], forecast_result["forecast"], forecast_country),
            use_container_width=True,
        )
        st.dataframe(forecast_result["forecast"], use_container_width=True, hide_index=True)
    except ValueError as exc:
        st.info(str(exc))

section_anchor("sec-conclusions", "9. Conclusions", "Automatically generated analytical takeaways for the current selection.")
top_year = latest.dropna(subset=["fpi"]).sort_values("fpi", ascending=False).head(5)
top_year_str = ", ".join(top_year["country_name"].tolist()) if not top_year.empty else "no countries"
avg_gap = latest["food_inflation_pct"].mean() - latest["headline_inflation_pct"].mean()
income_corr = df_filtered[["median_income_eur", "food_inflation_pct"]].corr().iloc[0, 1]
share_corr = df_filtered[["food_share_budget_pct", "fpi"]].corr().iloc[0, 1]

st.markdown(
    f"""
- In {map_year}, the highest Food Pressure Index values are observed in: **{top_year_str}**.
- In the selected countries, average food inflation is **{avg_gap:.2f} percentage points** above headline inflation.
- The correlation between median income and food inflation is **{income_corr:.2f}**, which indicates whether lower-income economies face stronger food price pressure.
- The correlation between the food spending share and FPI is **{share_corr:.2f}**, showing how budget structure amplifies affordability pressure.
"""
)

if exclusions is not None and not exclusions.empty:
    with st.expander("Data exclusions and caveats", expanded=False):
        st.dataframe(display_columns(exclusions), use_container_width=True, hide_index=True)

section_anchor("sec-export", "10. Export", "Filtered dataset for further analysis.")
export_cols = ["country_name", "country_code", "region", "year", *KEY_METRICS]
export_df = display_columns(df_filtered[export_cols].sort_values(["year", "country_name"]))
st.download_button(
    "Download filtered CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"europe_food_affordability_{year_range[0]}_{year_range[1]}.csv",
    mime="text/csv",
)
st.dataframe(export_df, use_container_width=True, hide_index=True)
