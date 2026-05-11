from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REGION_ORDER = ["Northern Europe", "Western Europe", "Southern Europe", "Central Europe", "Eastern Europe"]
REGION_COLORS = {
    "Northern Europe": "#3b82f6",
    "Western Europe": "#10b981",
    "Southern Europe": "#f97316",
    "Central Europe": "#8b5cf6",
    "Eastern Europe": "#ef4444",
}


def _apply_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55 if title else 25, b=10),
        legend_title_text="Region",
        font=dict(family="Inter, Segoe UI, Arial", size=13),
    )
    return fig


def choropleth(df: pd.DataFrame, value: str, year: int, title: str, color_scale: str = "Reds") -> go.Figure:
    d = df[df["year"] == year].copy()
    fig = px.choropleth(
        d,
        locations="iso3",
        color=value,
        hover_name="country_name",
        hover_data={
            "iso3": False,
            "country_code": True,
            "region": True,
            value: ":.2f",
        },
        color_continuous_scale=color_scale,
        projection="natural earth",
        scope="europe",
        labels={value: title, "region": "Region"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_coloraxes(colorbar_title=title)
    return _apply_layout(fig, f"{title} · {year}")


def line_trend(
    df: pd.DataFrame,
    value: str,
    countries: Iterable[str],
    title: str,
    y_label: str | None = None,
) -> go.Figure:
    d = df[df["country_name"].isin(list(countries))].copy()
    fig = px.line(
        d,
        x="year",
        y=value,
        color="country_name",
        markers=True,
        labels={"year": "Year", value: y_label or value, "country_name": "Country"},
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    return _apply_layout(fig, title)


def scatter_income(df: pd.DataFrame, year: int) -> go.Figure:
    d = df[df["year"] == year].copy()
    d["food_spending_share"] = d["food_share_budget_pct"].round(1)
    fig = px.scatter(
        d,
        x="median_income_eur",
        y="food_inflation_pct",
        size="food_spending_share",
        color="region",
        hover_name="country_name",
        hover_data={
            "median_income_eur": ":,.0f",
            "food_inflation_pct": ":.2f",
            "headline_inflation_pct": ":.2f",
            "fpi": ":.2f",
            "food_spending_share": ":.1f",
        },
        color_discrete_map=REGION_COLORS,
        category_orders={"region": REGION_ORDER},
        labels={
            "median_income_eur": "Median equivalised income (EUR/year)",
            "food_inflation_pct": "Food inflation (%)",
            "region": "Region",
            "food_spending_share": "Food spending share (%)",
        },
    )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=0.7, color="white")))
    return _apply_layout(fig, f"Household Income vs Food Inflation · {year}")


def boxplot_region(df: pd.DataFrame, value: str, year: int, title: str) -> go.Figure:
    d = df[df["year"] == year].copy()
    fig = px.box(
        d,
        x="region",
        y=value,
        color="region",
        points="all",
        category_orders={"region": REGION_ORDER},
        color_discrete_map=REGION_COLORS,
        hover_name="country_name",
        labels={"region": "Region", value: title},
    )
    fig.update_layout(showlegend=False)
    return _apply_layout(fig, f"{title} by Region · {year}")


def histogram(df: pd.DataFrame, value: str, year: int, title: str) -> go.Figure:
    d = df[df["year"] == year].copy()
    fig = px.histogram(
        d,
        x=value,
        color="region",
        nbins=20,
        marginal="box",
        color_discrete_map=REGION_COLORS,
        category_orders={"region": REGION_ORDER},
        labels={value: title, "region": "Region"},
    )
    fig.update_yaxes(title="Observation count")
    return _apply_layout(fig, f"Distribution of {title} · {year}")


def heatmap_corr(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    return _apply_layout(fig, "Metric Correlations")


def bar_with_ci(ci_df: pd.DataFrame, title: str, x: str = "region", y: str = "mean") -> go.Figure:
    if ci_df.empty:
        return go.Figure()
    d = ci_df.copy()
    err_plus = d["ci_high"] - d[y]
    err_minus = d[y] - d["ci_low"]
    fig = go.Figure()
    fig.add_bar(
        x=d[x],
        y=d[y],
        error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
        marker_color=[REGION_COLORS.get(v, "#64748b") for v in d[x]],
        customdata=np.stack([d["ci_low"], d["ci_high"], d["n"]], axis=-1),
        hovertemplate=(
            "%{x}<br>Mean: %{y:.2f}<br>95% CI: %{customdata[0]:.2f} - "
            "%{customdata[1]:.2f}<br>Countries in cluster: %{customdata[2]}<extra></extra>"
        ),
    )
    return _apply_layout(fig, title)


def heatmap_country_year(df: pd.DataFrame, value: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index="country_name", columns="year", values=value, aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(x="Year", y="Country", color=title),
    )
    return _apply_layout(fig, title)


def residuals_plot(predictions: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        predictions,
        x="predicted_fpi",
        y="residual",
        color="region" if "region" in predictions.columns else None,
        hover_name="country_name",
        color_discrete_map=REGION_COLORS,
        labels={"predicted_fpi": "Predicted FPI", "residual": "Residual (FPI - predicted)", "region": "Region"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#334155")
    return _apply_layout(fig, "Model Residuals")


def pca_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, explained: list[float]) -> go.Figure:
    fig = px.scatter(
        scores,
        x="PC1",
        y="PC2",
        color="region",
        hover_name="country_name",
        hover_data={"year": True, "PC1": ":.2f", "PC2": ":.2f"},
        color_discrete_map=REGION_COLORS,
        category_orders={"region": REGION_ORDER},
        labels={
            "PC1": f"PC1 ({explained[0] * 100:.1f}% variance)",
            "PC2": f"PC2 ({explained[1] * 100:.1f}% variance)",
            "region": "Region",
        },
    )
    for _, row in loadings.iterrows():
        fig.add_annotation(
            x=row["PC1"] * 2.5,
            y=row["PC2"] * 2.5,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=row["feature"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#0f172a",
            font=dict(size=11),
        )
    return _apply_layout(fig, "PCA: Countries in Affordability Space")


def forecast_plot(history: pd.DataFrame, forecast: pd.DataFrame, country_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=history["year"],
        y=history["food_inflation_pct"],
        mode="lines+markers",
        name="History",
        line=dict(color="#2563eb", width=3),
    )
    fig.add_scatter(
        x=forecast["year"],
        y=forecast["forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#ef4444", width=3, dash="dash"),
    )
    fig.add_scatter(
        x=forecast["year"],
        y=forecast["lower"],
        mode="lines",
        name="Lower bound",
        line=dict(color="rgba(239,68,68,0.25)"),
        showlegend=False,
    )
    fig.add_scatter(
        x=forecast["year"],
        y=forecast["upper"],
        mode="lines",
        name="Upper bound",
        fill="tonexty",
        fillcolor="rgba(239,68,68,0.12)",
        line=dict(color="rgba(239,68,68,0.25)"),
        showlegend=False,
    )
    fig.update_yaxes(title="Food inflation (%)")
    fig.update_xaxes(title="Year")
    return _apply_layout(fig, f"ARIMA Forecast - {country_label}")
