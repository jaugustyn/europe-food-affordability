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
REGION_LABELS_PL = {
    "Northern Europe": "Europa Północna",
    "Western Europe": "Europa Zachodnia",
    "Southern Europe": "Europa Południowa",
    "Central Europe": "Europa Centralna",
    "Eastern Europe": "Europa Wschodnia",
}
REGION_COLORS_PL = {REGION_LABELS_PL[key]: value for key, value in REGION_COLORS.items()}
FEATURE_LABELS_PL = {
    "fpi": "Indeks Presji Cen Żywności",
    "food_affordability_gap_pct": "Luka dostępności żywności",
    "food_inflation_pct": "Inflacja żywności",
    "headline_inflation_pct": "Inflacja ogólna",
    "food_share_budget_pct": "Udział żywności w wydatkach",
    "median_income_eur": "Mediana dochodu",
    "income_growth_pct": "Wzrost dochodu",
    "food_price_level_index": "Poziom cen żywności",
    "meal_deprivation_pct": "Brak pełnowartościowego posiłku",
}

SEGMENT_COLORS = {
    "Największe ryzyko dostępności": "#dc2626",
    "Presja cenowa": "#f97316",
    "Wrażliwy budżet": "#eab308",
    "Relatywnie stabilna sytuacja": "#16a34a",
    "Brak klasyfikacji": "#64748b",
}


def _apply_layout(fig: go.Figure, title: str | None = None, legend_title: str = "Region") -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55 if title else 25, b=10),
        legend_title_text=legend_title,
        font=dict(family="Inter, Segoe UI, Arial", size=13),
    )
    return fig


def _empty_figure(title: str, message: str = "Brak danych dla wybranych filtrów") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_layout(fig, title)


def choropleth(
    df: pd.DataFrame,
    value: str,
    year: int,
    title: str,
    color_scale: str = "Reds",
    color_range: tuple[float, float] | None = None,
) -> go.Figure:
    d = df[df["year"] == year].copy()
    d["region_label"] = d["region"].map(REGION_LABELS_PL).fillna(d["region"])
    missing = d[d[value].isna() & d["iso3"].notna()].copy()
    visible = d[d[value].notna()].copy()

    if visible.empty:
        fig = go.Figure()
    else:
        fig = px.choropleth(
            visible,
            locations="iso3",
            color=value,
            hover_name="country_name",
            hover_data={
                "iso3": False,
                "country_code": True,
                "region": False,
                "region_label": True,
                value: ":.2f",
            },
            color_continuous_scale=color_scale,
            range_color=color_range,
            projection="natural earth",
            scope="europe",
            labels={value: title, "region_label": "Region", "country_code": "Kod kraju"},
        )

    if not missing.empty:
        fig.add_trace(
            go.Choropleth(
                locations=missing["iso3"],
                z=[0] * len(missing),
                customdata=np.stack(
                    [missing["country_name"], missing["region_label"], missing["country_code"]],
                    axis=-1,
                ),
                colorscale=[[0, "#d1d5db"], [1, "#d1d5db"]],
                showscale=False,
                showlegend=True,
                name="Brak danych",
                marker_line_color="white",
                marker_line_width=0.6,
                hovertemplate=(
                    "%{customdata[0]}<br>Region: %{customdata[1]}<br>"
                    "Kod kraju: %{customdata[2]}<br>Brak danych dla wybranej metryki i roku<extra></extra>"
                ),
            )
        )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="natural earth",
        scope="europe",
    )
    if visible.empty:
        fig.update_layout(coloraxis_showscale=False)
    else:
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
        labels={"year": "Rok", value: y_label or value, "country_name": "Kraj"},
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    return _apply_layout(fig, title, legend_title="Kraj")


def scatter_income(df: pd.DataFrame, year: int) -> go.Figure:
    title = f"Dochód gospodarstw domowych vs inflacja żywności · {year}"
    d = df[df["year"] == year].copy()
    d = d.dropna(subset=["median_income_eur", "food_inflation_pct"]).copy()
    d = d[np.isfinite(d["median_income_eur"]) & np.isfinite(d["food_inflation_pct"])]
    if d.empty:
        return _empty_figure(title)

    fallback_size = d["food_share_budget_pct"].dropna().median()
    if pd.isna(fallback_size) or fallback_size <= 0:
        fallback_size = 1.0
    d["food_spending_share"] = d["food_share_budget_pct"].fillna(fallback_size).clip(lower=0.1).round(1)
    d["region_label"] = d["region"].map(REGION_LABELS_PL).fillna(d["region"])
    fig = px.scatter(
        d,
        x="median_income_eur",
        y="food_inflation_pct",
        size="food_spending_share",
        color="region_label",
        hover_name="country_name",
        hover_data={
            "median_income_eur": ":,.0f",
            "food_inflation_pct": ":.2f",
            "headline_inflation_pct": ":.2f",
            "fpi": ":.2f",
            "food_spending_share": ":.1f",
        },
        color_discrete_map=REGION_COLORS_PL,
        category_orders={"region_label": [REGION_LABELS_PL[r] for r in REGION_ORDER]},
        labels={
            "median_income_eur": "Mediana dochodu ekwiwalentnego (EUR/rok)",
            "food_inflation_pct": "Inflacja żywności (%)",
            "region_label": "Region",
            "food_spending_share": "Udział wydatków na żywność (%)",
        },
    )
    trend = d[["median_income_eur", "food_inflation_pct"]].dropna().sort_values("median_income_eur")
    if len(trend) >= 2 and trend["median_income_eur"].nunique() >= 2:
        slope, intercept = np.polyfit(trend["median_income_eur"], trend["food_inflation_pct"], 1)
        fig.add_scatter(
            x=trend["median_income_eur"],
            y=slope * trend["median_income_eur"] + intercept,
            mode="lines",
            name="Trend OLS",
            line=dict(color="#0f172a", width=2.4, dash="dash"),
            hovertemplate="Trend OLS<extra></extra>",
        )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=0.7, color="white")), selector=dict(mode="markers"))
    return _apply_layout(fig, title)


def driver_bar(row: pd.Series, country_label: str, year: int) -> go.Figure:
    metrics = [
        ("food_inflation_pct", "Inflacja żywności"),
        ("income_growth_pct", "Wzrost dochodu"),
        ("food_affordability_gap_pct", "Luka cen-dochód"),
        ("food_share_budget_pct", "Udział żywności w wydatkach"),
        ("meal_deprivation_pct", "Brak pełnowartościowego posiłku"),
    ]
    rows = [
        {"key": key, "metric": label, "value": float(row[key])}
        for key, label in metrics
        if key in row.index and pd.notna(row[key])
    ]
    if not rows:
        return go.Figure()

    d = pd.DataFrame(rows)
    colors = []
    for _, item in d.iterrows():
        if item["key"] == "income_growth_pct":
            colors.append("#16a34a" if item["value"] >= 0 else "#dc2626")
        elif item["key"] == "food_affordability_gap_pct":
            colors.append("#dc2626" if item["value"] > 0 else "#16a34a")
        else:
            colors.append("#dc2626")
    fig = go.Figure()
    fig.add_bar(
        x=d["value"],
        y=d["metric"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Wartość: %{x:.2f}<extra></extra>",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
    fig.update_xaxes(title="Wartość (%) lub p.p.")
    fig.update_yaxes(title="")
    return _apply_layout(fig, f"Co napędza presję? · {country_label}, {year}", legend_title="")


def cumulative_gap_bar(summary: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if summary.empty:
        return go.Figure()
    d = summary.nlargest(top_n, "cumulative_affordability_gap_pct").sort_values("cumulative_affordability_gap_pct")
    d["region_label"] = d["region"].map(REGION_LABELS_PL).fillna(d["region"])
    fig = px.bar(
        d,
        x="cumulative_affordability_gap_pct",
        y="country_name",
        color="region_label",
        orientation="h",
        color_discrete_map=REGION_COLORS_PL,
        labels={
            "cumulative_affordability_gap_pct": "Ceny żywności minus wzrost dochodu (p.p.)",
            "country_name": "Kraj",
            "region_label": "Region",
        },
        hover_data={
            "food_price_growth_2020_2024_pct": ":.1f",
            "income_growth_2020_2024_pct": ":.1f",
            "cumulative_affordability_gap_pct": ":.1f",
            "region_label": False,
        },
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
    fig.update_layout(height=max(480, 28 * len(d) + 120))
    return _apply_layout(fig, "Największa skumulowana presja 2020-2024")


def typology_scatter(df: pd.DataFrame, year: int) -> go.Figure:
    title = f"Typologia krajów według presji i dochodu · {year}"
    d = df.dropna(
        subset=[
            "median_income_eur",
            "food_affordability_gap_pct",
            "food_share_budget_pct",
            "pressure_segment",
        ]
    ).copy()
    d = d[np.isfinite(d["median_income_eur"]) & np.isfinite(d["food_affordability_gap_pct"])]
    if d.empty:
        return _empty_figure(title)
    d["food_share_budget_pct"] = d["food_share_budget_pct"].clip(lower=0.1)
    fig = px.scatter(
        d,
        x="median_income_eur",
        y="food_affordability_gap_pct",
        size="food_share_budget_pct",
        color="pressure_segment",
        hover_name="country_name",
        hover_data={
            "food_inflation_pct": ":.2f",
            "income_growth_pct": ":.2f",
            "food_share_budget_pct": ":.1f",
            "meal_deprivation_pct": ":.1f",
            "pressure_segment": False,
        },
        color_discrete_map=SEGMENT_COLORS,
        labels={
            "median_income_eur": "Mediana dochodu (EUR/rok)",
            "food_affordability_gap_pct": "Luka dostępności żywności (p.p.)",
            "food_share_budget_pct": "Udział żywności w wydatkach (%)",
            "pressure_segment": "Typ kraju",
            "food_inflation_pct": "Inflacja żywności (%)",
            "income_growth_pct": "Wzrost dochodu (%)",
            "meal_deprivation_pct": "Brak pełnowartościowego posiłku (%)",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=0.7, color="white")))
    return _apply_layout(fig, title, legend_title="Typ kraju")


def boxplot_region(df: pd.DataFrame, value: str, year: int, title: str) -> go.Figure:
    fig_title = f"{title} według regionu · {year}"
    d = df[df["year"] == year].dropna(subset=[value]).copy()
    d = d[np.isfinite(d[value])]
    if d.empty:
        return _empty_figure(fig_title)
    d["region_label"] = d["region"].map(REGION_LABELS_PL).fillna(d["region"])
    fig = px.box(
        d,
        x="region_label",
        y=value,
        color="region_label",
        points="all",
        category_orders={"region_label": [REGION_LABELS_PL[r] for r in REGION_ORDER]},
        color_discrete_map=REGION_COLORS_PL,
        hover_name="country_name",
        labels={"region_label": "Region", value: title},
    )
    fig.update_layout(showlegend=False)
    return _apply_layout(fig, fig_title)


def histogram(df: pd.DataFrame, value: str, year: int, title: str) -> go.Figure:
    fig_title = f"Rozkład: {title} · {year}"
    d = df[df["year"] == year].dropna(subset=[value]).copy()
    d = d[np.isfinite(d[value])]
    if d.empty:
        return _empty_figure(fig_title)
    d["region_label"] = d["region"].map(REGION_LABELS_PL).fillna(d["region"])
    fig = px.histogram(
        d,
        x=value,
        color="region_label",
        nbins=20,
        marginal="box",
        color_discrete_map=REGION_COLORS_PL,
        category_orders={"region_label": [REGION_LABELS_PL[r] for r in REGION_ORDER]},
        labels={value: title, "region_label": "Region"},
    )
    fig.update_yaxes(title="Liczba obserwacji")
    return _apply_layout(fig, fig_title)


def heatmap_corr(corr: pd.DataFrame, title: str = "Korelacje metryk") -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    return _apply_layout(fig, title)


def bar_with_ci(ci_df: pd.DataFrame, title: str, x: str = "region", y: str = "mean") -> go.Figure:
    if ci_df.empty:
        return go.Figure()
    d = ci_df.copy()
    d[x] = d[x].map(REGION_LABELS_PL).fillna(d[x])
    err_plus = d["ci_high"] - d[y]
    err_minus = d[y] - d["ci_low"]
    fig = go.Figure()
    fig.add_bar(
        x=d[x],
        y=d[y],
        error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
        marker_color=[REGION_COLORS_PL.get(v, "#64748b") for v in d[x]],
        customdata=np.stack([d["ci_low"], d["ci_high"], d["n"]], axis=-1),
        hovertemplate=(
            "%{x}<br>Średnia: %{y:.2f}<br>95% CI: %{customdata[0]:.2f} - "
            "%{customdata[1]:.2f}<br>Liczba krajów: %{customdata[2]}<extra></extra>"
        ),
    )
    return _apply_layout(fig, title)


def heatmap_country_year(df: pd.DataFrame, value: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index="country_name", columns="year", values=value, aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(year) for year in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="YlOrRd",
            colorbar=dict(title=title),
            hovertemplate="Kraj: %{y}<br>Rok: %{x}<br>Wartość: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(560, 22 * len(pivot.index) + 120))
    fig.update_xaxes(title="Rok", tickangle=0)
    fig.update_yaxes(
        title="Kraj",
        tickmode="array",
        tickvals=pivot.index.tolist(),
        ticktext=pivot.index.tolist(),
        tickfont=dict(size=10),
        automargin=True,
    )
    return _apply_layout(fig, title)


def residuals_plot(
    predictions: pd.DataFrame,
    target_label: str = "Wartość docelowa",
) -> go.Figure:
    predictions = predictions.copy()
    if "region" in predictions.columns:
        predictions["region_label"] = predictions["region"].map(REGION_LABELS_PL).fillna(predictions["region"])
    fig = px.scatter(
        predictions,
        x="predicted",
        y="residual",
        color="region_label" if "region_label" in predictions.columns else None,
        hover_name="country_name",
        color_discrete_map=REGION_COLORS_PL,
        labels={
            "predicted": f"Prognozowana wartość: {target_label}",
            "residual": "Reszta (wartość rzeczywista - prognoza)",
            "region_label": "Region",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#334155")
    return _apply_layout(fig, "Reszty modelu")


def pca_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, explained: list[float]) -> go.Figure:
    scores = scores.copy()
    scores["region_label"] = scores["region"].map(REGION_LABELS_PL).fillna(scores["region"])
    fig = px.scatter(
        scores,
        x="PC1",
        y="PC2",
        color="region_label",
        hover_name="country_name",
        hover_data={"year": True, "PC1": ":.2f", "PC2": ":.2f"},
        color_discrete_map=REGION_COLORS_PL,
        category_orders={"region_label": [REGION_LABELS_PL[r] for r in REGION_ORDER]},
        labels={
            "PC1": f"PC1 ({explained[0] * 100:.1f}% wariancji)",
            "PC2": f"PC2 ({explained[1] * 100:.1f}% wariancji)",
            "region_label": "Region",
            "year": "Rok",
        },
    )
    endpoints = loadings.assign(
        label_x=lambda x: x["PC1"] * 2.9,
        label_y=lambda x: x["PC2"] * 2.9,
    ).sort_values("label_y").reset_index(drop=True)
    min_gap = 0.22
    for i in range(1, len(endpoints)):
        if endpoints.loc[i, "label_y"] - endpoints.loc[i - 1, "label_y"] < min_gap:
            endpoints.loc[i, "label_y"] = endpoints.loc[i - 1, "label_y"] + min_gap

    for _, row in endpoints.iterrows():
        end_x = row["PC1"] * 2.35
        end_y = row["PC2"] * 2.35
        label_x = row["label_x"] + (0.2 if row["PC1"] >= 0 else -0.2)
        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=0,
            ay=0,
            text="",
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor="#0f172a",
        )
        fig.add_annotation(
            x=label_x,
            y=row["label_y"],
            text=FEATURE_LABELS_PL.get(row["feature"], row["feature"]),
            showarrow=True,
            arrowhead=0,
            ax=end_x,
            ay=end_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowwidth=0.8,
            arrowcolor="rgba(15,23,42,0.45)",
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="rgba(148,163,184,0.7)",
            borderpad=2,
            font=dict(size=10),
        )
    return _apply_layout(fig, "PCA: kraje w przestrzeni dostępności żywności")


def pca_scree_plot(explained: list[float]) -> go.Figure:
    explained_pct = np.asarray(explained, dtype=float) * 100
    cumulative = np.cumsum(explained_pct)
    components = [f"PC{i + 1}" for i in range(len(explained_pct))]

    fig = go.Figure()
    fig.add_bar(
        x=components,
        y=explained_pct,
        name="Wariancja komponentu",
        marker_color="#2563eb",
        hovertemplate="%{x}<br>Udział: %{y:.1f}%<extra></extra>",
    )
    fig.add_scatter(
        x=components,
        y=cumulative,
        mode="lines+markers",
        name="Wariancja skumulowana",
        line=dict(color="#dc2626", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{x}<br>Skumulowanie: %{y:.1f}%<extra></extra>",
    )
    fig.update_yaxes(title="Wyjaśniona wariancja (%)", range=[0, 105])
    fig.update_xaxes(title="Komponent PCA")
    return _apply_layout(fig, "Scree plot PCA", legend_title="Miara")


def forecast_plot(history: pd.DataFrame, forecast: pd.DataFrame, country_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=history["year"],
        y=history["food_inflation_pct"],
        mode="lines+markers",
        name="Dane historyczne",
        line=dict(color="#2563eb", width=3),
    )
    fig.add_scatter(
        x=forecast["year"],
        y=forecast["forecast"],
        mode="lines+markers",
        name="Prognoza",
        line=dict(color="#ef4444", width=3, dash="dash"),
    )
    if "baseline_last" in forecast.columns:
        fig.add_scatter(
            x=forecast["year"],
            y=forecast["baseline_last"],
            mode="lines+markers",
            name="Baseline: ostatnia wartość",
            line=dict(color="#475569", width=2, dash="dot"),
        )
    if "baseline_recent_mean" in forecast.columns:
        fig.add_scatter(
            x=forecast["year"],
            y=forecast["baseline_recent_mean"],
            mode="lines+markers",
            name="Baseline: średnia z 3 lat",
            line=dict(color="#0f766e", width=2, dash="dot"),
        )
    fig.add_scatter(
        x=forecast["year"],
        y=forecast["lower"],
        mode="lines",
        name="Dolna granica",
        line=dict(color="rgba(239,68,68,0.25)"),
        showlegend=False,
    )
    fig.add_scatter(
        x=forecast["year"],
        y=forecast["upper"],
        mode="lines",
        name="Górna granica",
        fill="tonexty",
        fillcolor="rgba(239,68,68,0.12)",
        line=dict(color="rgba(239,68,68,0.25)"),
        showlegend=False,
    )
    fig.update_yaxes(title="Inflacja żywności (%)")
    fig.update_xaxes(title="Rok")
    return _apply_layout(fig, f"Prognoza ARIMA - {country_label}")
