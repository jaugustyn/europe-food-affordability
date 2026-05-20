from __future__ import annotations

from html import escape

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
    cumulative_gap_bar,
    driver_bar,
    forecast_plot,
    heatmap_corr,
    heatmap_country_year,
    histogram,
    line_trend,
    pca_biplot,
    pca_scree_plot,
    residuals_plot,
    scatter_income,
    typology_scatter,
)


st.set_page_config(
    page_title="Dostępność Żywności w Europie",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Dostępność Żywności w Europie"

REGION_LABELS_PL = {
    "Northern Europe": "Europa Północna",
    "Western Europe": "Europa Zachodnia",
    "Southern Europe": "Europa Południowa",
    "Central Europe": "Europa Centralna",
    "Eastern Europe": "Europa Wschodnia",
}

METRIC_LABELS_PL = {
    "fpi": "Indeks Presji Cen Żywności",
    "food_inflation_pct": "Inflacja żywności (HICP CP01)",
    "headline_inflation_pct": "Inflacja ogólna (HICP CP00)",
    "food_share_budget_pct": "Udział żywności w wydatkach gospodarstw",
    "median_income_eur": "Mediana dochodu ekwiwalentnego",
    "income_growth_pct": "Wzrost mediany dochodu",
    "minimum_wage_eur_month": "Płaca minimalna",
    "food_price_level_index": "Poziom cen żywności (UE=100)",
    "meal_deprivation_pct": "Brak możliwości pełnowartościowego posiłku",
    "food_affordability_gap_pct": "Luka dostępności żywności",
    "food_inflation_index_2020": "Skumulowany indeks cen żywności (2020=100)",
    "food_price_growth_2020_2024_pct": "Skumulowany wzrost cen żywności 2020-2024",
    "income_growth_2020_2024_pct": "Skumulowany wzrost dochodu 2020-2024",
    "cumulative_affordability_gap_pct": "Skumulowana luka dostępności 2020-2024",
    "pressure_segment": "Typ presji",
}

METRIC_DESCRIPTIONS_PL = {
    "fpi": "Wskaźnik syntetyczny: inflacja żywności (%) / wzrost mediany dochodu (%). Wartości powyżej 1 oznaczają, że ceny żywności rosły szybciej niż dochody. Gdy dochód spada, FPI może być ujemny, więc warto porównać go z luką dostępności.",
    "food_inflation_pct": "Roczna średnia zmiana zharmonizowanego indeksu cen żywności i napojów bezalkoholowych.",
    "headline_inflation_pct": "Roczna średnia zmiana zharmonizowanego indeksu cen dla wszystkich kategorii.",
    "food_share_budget_pct": "Udział żywności i napojów bezalkoholowych w wydatkach konsumpcyjnych gospodarstw domowych.",
    "median_income_eur": "Roczna mediana ekwiwalentnego dochodu netto.",
    "income_growth_pct": "Roczna dynamika mediany ekwiwalentnego dochodu netto.",
    "minimum_wage_eur_month": "Miesięczna płaca minimalna w EUR; dane półroczne uśrednione do lat.",
    "food_price_level_index": "Poziom cen żywności i napojów bezalkoholowych względem UE27_2020=100.",
    "meal_deprivation_pct": "Odsetek osób, których nie stać na posiłek z mięsem, rybą lub odpowiednikiem wegetariańskim co drugi dzień.",
    "food_affordability_gap_pct": "Inflacja żywności minus wzrost mediany dochodu. Wartości dodatnie oznaczają, że ceny żywności rosły szybciej niż dochody i dostępność się pogarszała.",
    "food_inflation_index_2020": "Skumulowany indeks zbudowany z rocznych zmian HICP CP01, z bazą 2020=100.",
    "food_price_growth_2020_2024_pct": "Łączny wzrost indeksu cen żywności od 2020 do 2024 roku.",
    "income_growth_2020_2024_pct": "Łączny wzrost mediany dochodu ekwiwalentnego od 2020 do 2024 roku.",
    "cumulative_affordability_gap_pct": "Skumulowany wzrost cen żywności minus skumulowany wzrost dochodu. Dodatni wynik oznacza pogorszenie dostępności w latach 2020-2024.",
}

MAP_METRICS = [
    "food_affordability_gap_pct",
    "fpi",
    "food_inflation_pct",
    "food_share_budget_pct",
    "food_price_level_index",
    "meal_deprivation_pct",
]

SECTION_HELP_PL = {
    "kpi": "KPI to szybkie podsumowanie sytuacji w roku referencyjnym. Najważniejsza jest luka dostępności: inflacja żywności minus wzrost dochodu. Dodatni wynik mówi, że ceny żywności rosły szybciej niż dochody.",
    "drivers": "Ta sekcja rozkłada wynik kraju na podstawowe składniki: inflację żywności, wzrost dochodu, lukę cen-dochód, udział żywności w wydatkach oraz deprywację posiłku. Dzięki temu ranking nie jest tylko listą krajów, ale ma wyjaśnienie.",
    "map": "Mapa pokazuje, które kraje mają wysokie lub niskie wartości wybranej metryki w konkretnym roku. Szare kraje nie mają kompletnej wartości dla tej metryki i roku. Skala kolorów jest stała dla aktualnych filtrów, więc ten sam kolor oznacza porównywalną wartość przy zmianie roku.",
    "cumulative": "Porównanie 2020-2024 pokazuje efekt skumulowany, a nie tylko jeden rok. Dodatnia skumulowana luka oznacza, że od 2020 roku ceny żywności wzrosły bardziej niż mediana dochodu.",
    "typology": "Typologia dzieli kraje na proste grupy interpretacyjne. Bierze pod uwagę lukę dostępności, udział żywności w budżecie, poziom dochodu i deprywację posiłku, więc pomaga znaleźć kraje o podobnym profilu ryzyka.",
    "trends": "Trend pokazuje zmianę w czasie. Linia inflacji żywności mówi, jak szybko rosły ceny jedzenia, a linia luki dostępności pokazuje, czy tempo wzrostu cen było większe niż tempo wzrostu dochodów.",
    "income": "Każda kropka to kraj w roku referencyjnym. Oś pozioma pokazuje dochód, oś pionowa inflację żywności, a rozmiar kropki udział żywności w wydatkach. Linia OLS to prosta tendencja: pomaga zobaczyć ogólny kierunek relacji, ale nie dowodzi przyczynowości.",
    "distributions": "Box plot pokazuje medianę, typowy zakres wartości i obserwacje skrajne między regionami. Histogram pokazuje, gdzie skupia się większość krajów. Tabela Z-score wskazuje obserwacje najbardziej oddalone od średniej dla aktualnych filtrów.",
    "correlations": "Korelacja mieści się od -1 do 1: wartości blisko 1 rosną razem, blisko -1 poruszają się przeciwnie, a blisko 0 nie mają prostej zależności. p-value pomaga ocenić, czy zależność może być przypadkowa. PCA streszcza kilka podobnych metryk do dwóch osi, żeby zobaczyć podobieństwa krajów.",
    "tests": "Testy statystyczne sprawdzają, czy różnice między regionami są większe niż losowe wahania danych. p-value poniżej 0,05 traktujemy jako sygnał istotności, a przedziały ufności pokazują niepewność średnich.",
    "prediction": "Modele predykcyjne próbują odtworzyć FPI na podstawie innych zmiennych. Reszty pokazują błędy modelu, efekty stałe kontrolują stałe różnice między krajami, a prognoza ARIMA ekstrapoluje sam trend inflacji żywności.",
    "conclusions": "Wnioski są generowane z aktualnie przefiltrowanych danych. Zmiana lat, regionów lub krajów może zmienić ranking, korelacje i średnie. Ograniczenia pomagają nie interpretować dashboardu jako modelu przyczynowego.",
    "export": "Eksport zapisuje dokładnie ten wycinek danych, który widzisz po zastosowaniu filtrów. To ułatwia dalszą analizę lub dołączenie tabeli do raportu.",
}

EFFECT_SIZE_LABELS_PL = {
    "negligible": "pomijalna",
    "small": "mała",
    "medium": "średnia",
    "large": "duża",
    "n/a": "brak danych",
}


COLUMN_LABELS = {
    "country_name": "Kraj",
    "country_code": "Kod kraju",
    "iso3": "ISO-3",
    "region": "Region",
    "year": "Rok",
    "zscore": "Z-score",
    "forecast": "Prognoza",
    "lower": "Dolna granica",
    "upper": "Górna granica",
    **METRIC_LABELS_PL,
}


def display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={k: v for k, v in COLUMN_LABELS.items() if k in frame.columns}).copy()
    if "Region" in out.columns:
        out["Region"] = out["Region"].map(REGION_LABELS_PL).fillna(out["Region"])
    return out


def metric_label(metric: str) -> str:
    return METRIC_LABELS_PL.get(metric, METRICS[metric]["label"] if metric in METRICS else metric)


def metric_description(metric: str) -> str:
    return METRIC_DESCRIPTIONS_PL.get(metric, METRICS[metric]["desc"] if metric in METRICS else "")


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def section_anchor(anchor: str, title: str, subtitle: str | None = None, help_text: str | None = None) -> None:
    tooltip = help_text or subtitle
    help_html = (
        " "
        f"<span class='section-help' tabindex='0' aria-label='{escape(tooltip)}'>"
        f"ⓘ<span class='section-tooltip'>{escape(tooltip)}</span></span>"
        if tooltip
        else ""
    )
    st.markdown(f"<h2 id='{anchor}'>{title}{help_html}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def metric_color_range(frame: pd.DataFrame, metric: str, robust: bool = False) -> tuple[float, float] | None:
    values = frame[metric].dropna()
    if values.empty:
        return None

    if robust and len(values) >= 20:
        cmin, cmax = values.quantile([0.05, 0.95]).astype(float).tolist()
    else:
        cmin = float(values.min())
        cmax = float(values.max())
    if METRICS.get(metric, {}).get("diverging") and cmin < 0 < cmax:
        limit = max(abs(cmin), abs(cmax))
        cmin, cmax = -limit, limit
    if cmin == cmax:
        pad = abs(cmin) * 0.05 or 1.0
        cmin -= pad
        cmax += pad
    return cmin, cmax


def fmt_value(value: float, metric: str) -> str:
    if metric in METRICS:
        return fmt(value, metric)
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1f}%"


def cumulative_pressure_summary(frame: pd.DataFrame, start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
    cols = [
        "country_code",
        "country_name",
        "region",
        "year",
        "food_inflation_index_2020",
        "median_income_eur",
    ]
    base = frame.loc[frame["year"] == start_year, cols].rename(
        columns={
            "food_inflation_index_2020": "food_price_index_start",
            "median_income_eur": "median_income_start",
        }
    )
    end = frame.loc[frame["year"] == end_year, cols].rename(
        columns={
            "food_inflation_index_2020": "food_price_index_end",
            "median_income_eur": "median_income_end",
        }
    )
    merged = base.merge(
        end[["country_code", "food_price_index_end", "median_income_end"]],
        on="country_code",
        how="inner",
    )
    merged = merged.dropna(
        subset=[
            "food_price_index_start",
            "food_price_index_end",
            "median_income_start",
            "median_income_end",
        ]
    ).copy()
    merged = merged[(merged["food_price_index_start"] > 0) & (merged["median_income_start"] > 0)]
    if merged.empty:
        return merged

    merged["food_price_growth_2020_2024_pct"] = (
        merged["food_price_index_end"] / merged["food_price_index_start"] - 1
    ) * 100
    merged["income_growth_2020_2024_pct"] = (
        merged["median_income_end"] / merged["median_income_start"] - 1
    ) * 100
    merged["cumulative_affordability_gap_pct"] = (
        merged["food_price_growth_2020_2024_pct"] - merged["income_growth_2020_2024_pct"]
    )
    return merged.sort_values("cumulative_affordability_gap_pct", ascending=False).reset_index(drop=True)


def classify_pressure_segments(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "food_affordability_gap_pct",
        "food_share_budget_pct",
        "median_income_eur",
        "meal_deprivation_pct",
    ]
    out = frame.copy()
    if out.empty or out[required].dropna(how="all").empty:
        out["pressure_segment"] = "Brak klasyfikacji"
        return out

    medians = out[required].median(numeric_only=True)

    def classify(row: pd.Series) -> str:
        if pd.isna(row.get("food_affordability_gap_pct")):
            return "Brak klasyfikacji"
        high_gap = row["food_affordability_gap_pct"] > medians["food_affordability_gap_pct"]
        high_budget = row.get("food_share_budget_pct", np.nan) > medians["food_share_budget_pct"]
        low_income = row.get("median_income_eur", np.nan) < medians["median_income_eur"]
        high_deprivation = row.get("meal_deprivation_pct", np.nan) > medians["meal_deprivation_pct"]
        if high_gap and (high_budget or low_income or high_deprivation):
            return "Największe ryzyko dostępności"
        if high_gap:
            return "Presja cenowa"
        if high_budget or high_deprivation:
            return "Wrażliwy budżet"
        return "Relatywnie stabilna sytuacja"

    out["pressure_segment"] = out.apply(classify, axis=1)
    return out


def pressure_driver_notes(row: pd.Series, reference_frame: pd.DataFrame) -> list[str]:
    notes = []
    med = reference_frame[
        [
            "food_share_budget_pct",
            "meal_deprivation_pct",
            "median_income_eur",
            "food_affordability_gap_pct",
        ]
    ].median(numeric_only=True)

    gap = row.get("food_affordability_gap_pct", np.nan)
    if pd.notna(gap) and gap > 0:
        notes.append(
            f"Ceny żywności rosły szybciej niż dochody o {gap:.1f} p.p., więc dostępność pogorszyła się w tym roku."
        )
    elif pd.notna(gap):
        notes.append(
            f"Dochody rosły szybciej niż ceny żywności o {abs(gap):.1f} p.p., więc presja cenowa była amortyzowana."
        )

    food_share = row.get("food_share_budget_pct", np.nan)
    if pd.notna(food_share) and pd.notna(med.get("food_share_budget_pct")) and food_share > med["food_share_budget_pct"]:
        notes.append(
            f"Udział żywności w wydatkach ({food_share:.1f}%) jest powyżej mediany wybranych krajów, więc wzrost cen mocniej obciąża budżet."
        )

    meal_deprivation = row.get("meal_deprivation_pct", np.nan)
    if pd.notna(meal_deprivation) and pd.notna(med.get("meal_deprivation_pct")) and meal_deprivation > med["meal_deprivation_pct"]:
        notes.append(
            f"Odsetek osób bez możliwości pełnowartościowego posiłku ({meal_deprivation:.1f}%) jest powyżej mediany, co zwiększa ryzyko społeczne."
        )

    income = row.get("median_income_eur", np.nan)
    if pd.notna(income) and pd.notna(med.get("median_income_eur")) and income < med["median_income_eur"]:
        notes.append(
            "Mediana dochodu jest poniżej mediany wybranych krajów, więc gospodarstwa mają mniejszy bufor na wzrost cen."
        )

    if not notes:
        notes.append("Dla tego kraju nie widać pojedynczego silnego czynnika presji względem aktualnie wybranych krajów.")
    return notes


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner=False)
def get_exclusions() -> pd.DataFrame:
    return load_exclusions()


try:
    df = get_data()
except FileNotFoundError:
    st.error("Nie znaleziono pliku z przetworzonymi danymi. Najpierw uruchom `python etl.py`.")
    st.stop()

exclusions = get_exclusions()
min_year, max_year = int(df["year"].min()), int(df["year"].max())

st.sidebar.title("Filtry globalne")
year_range = st.sidebar.slider("Zakres lat", min_year, max_year, (max(min_year, max_year - 9), max_year))
reference_year = st.sidebar.slider("Rok referencyjny", year_range[0], year_range[1], year_range[1])
regions = st.sidebar.multiselect(
    "Regiony",
    options=sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique()),
    format_func=lambda region: REGION_LABELS_PL.get(region, region),
)

country_options = (
    df[df["region"].isin(regions)]["country_name"].drop_duplicates().sort_values().tolist()
    if regions
    else df["country_name"].drop_duplicates().sort_values().tolist()
)
countries = st.sidebar.multiselect(
    "Kraje",
    options=country_options,
    default=country_options,
)

df_filtered = df[
    df["year"].between(year_range[0], year_range[1])
    & df["region"].isin(regions)
    & df["country_name"].isin(countries)
].copy()
scope_df = df[df["region"].isin(regions) & df["country_name"].isin(countries)].copy()
latest = df_filtered[df_filtered["year"] == reference_year].copy()

st.title(APP_TITLE)
st.caption(
    "Dashboard do porównywania presji cen żywności, inflacji ogólnej, dochodów gospodarstw "
    "domowych i dostępności żywności w krajach Europy na podstawie danych Eurostatu."
)

st.markdown(
    """
<style>
    html {scroll-behavior: smooth;}
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
    h2 {margin-top: 2rem;}
    .section-help {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.15rem;
        height: 1.15rem;
        color: #64748b;
        cursor: help;
        font-size: 0.9rem;
        vertical-align: middle;
        border-radius: 999px;
    }
    .section-help:hover,
    .section-help:focus {
        color: #0f172a;
        outline: none;
        background: #e2e8f0;
    }
    .section-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        left: 1.45rem;
        top: 50%;
        transform: translateY(-50%);
        z-index: 20;
        width: min(440px, 70vw);
        padding: 0.65rem 0.75rem;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.16);
        color: #0f172a;
        font-size: 0.86rem;
        font-weight: 400;
        line-height: 1.35;
        text-align: left;
        white-space: normal;
        pointer-events: none;
        transition: opacity 120ms ease;
    }
    .section-help:hover .section-tooltip,
    .section-help:focus .section-tooltip {
        visibility: visible;
        opacity: 1;
    }
    div[data-testid="stMetricValue"] {font-size: 1.75rem;}
</style>
""",
    unsafe_allow_html=True,
)

if df_filtered.empty or latest.empty:
    st.warning("Brak obserwacji dla wybranych filtrów.")
    st.stop()

latest_typology = classify_pressure_segments(latest)
cumulative_2020_2024 = cumulative_pressure_summary(scope_df)

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
median_fpi = latest["fpi"].median()

with kpi_cols[0]:
    if top_gap.empty:
        st.metric("Największa luka dostępności", "n/a")
    else:
        row = top_gap.iloc[0]
        st.metric(
            "Największa luka dostępności",
            row["country_name"],
            fmt(row["food_affordability_gap_pct"], "food_affordability_gap_pct"),
            help=metric_description("food_affordability_gap_pct"),
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
        delta=f"FPI mediana: {fmt(median_fpi, 'fpi')}",
        help=metric_description("food_affordability_gap_pct"),
    )

with st.expander("Definicje metryk", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [
                {"Metryka": metric_label(key), "Jednostka": meta["unit"], "Opis": metric_description(key)}
                for key, meta in METRICS.items()
                if key in KEY_METRICS
            ]
        ),
        width="stretch",
        hide_index=True,
    )

section_anchor(
    "sec-drivers",
    "2. Diagnoza kraju",
    "Co konkretnie napędza presję cen żywności w wybranym kraju.",
    help_text=SECTION_HELP_PL["drivers"],
)
driver_options = latest_typology["country_name"].drop_duplicates().sort_values().tolist()
default_driver = top_gap.iloc[0]["country_name"] if not top_gap.empty else driver_options[0]
driver_country = st.selectbox(
    "Kraj do diagnozy",
    options=driver_options,
    index=driver_options.index(default_driver) if default_driver in driver_options else 0,
)
driver_row = latest_typology[latest_typology["country_name"] == driver_country].iloc[0]
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
        st.metric(label, fmt_value(driver_row.get(metric), metric), help=metric_description(metric))

driver_chart, driver_notes_col = st.columns([1.15, 0.85])
with driver_chart:
    st.plotly_chart(driver_bar(driver_row, driver_country, reference_year), width="stretch")
with driver_notes_col:
    st.markdown("**Interpretacja**")
    st.markdown("\n".join(f"- {note}" for note in pressure_driver_notes(driver_row, latest)))
    st.caption(f"Typ kraju: {driver_row.get('pressure_segment', 'Brak klasyfikacji')}")

section_anchor(
    "sec-map",
    "3. Mapa presji cenowej",
    "Mapa porównuje kraje w wybranym roku. Braki danych są wyszarzone.",
    help_text=SECTION_HELP_PL["map"],
)

map_controls, map_view = st.columns([0.26, 0.74])
with map_controls:
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
        help="Wybierz, czy kolor ma oznaczać presję cenową, inflację, udział żywności w budżecie, poziom cen albo deprywację posiłku.",
    )
    st.caption(metric_description(map_metric))
    use_full_map_range = st.toggle(
        "Pełny zakres min-max",
        value=False,
        help=(
            "Domyślnie mapa używa stałej skali 5-95 percentyl, żeby pojedyncze skrajne wartości "
            "nie spłaszczały kolorów. Włącz tę opcję, jeśli chcesz objąć wszystkie wartości."
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

section_anchor(
    "sec-cumulative",
    "4. Presja skumulowana 2020-2024",
    "Porównanie łącznego wzrostu cen żywności z łącznym wzrostem dochodów.",
    help_text=SECTION_HELP_PL["cumulative"],
)
if cumulative_2020_2024.empty:
    st.info("Brak kompletnych danych dla porównania 2020-2024 w aktualnie wybranych krajach.")
else:
    top_cumulative = cumulative_2020_2024.iloc[0]
    cum_cols = st.columns(3)
    with cum_cols[0]:
        st.metric(
            "Największa skumulowana luka",
            top_cumulative["country_name"],
            f"{top_cumulative['cumulative_affordability_gap_pct']:+.1f} p.p.",
            help=metric_description("cumulative_affordability_gap_pct"),
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
    "sec-typology",
    "5. Typologia krajów",
    "Grupy krajów o podobnym profilu presji cenowej i społecznej.",
    help_text=SECTION_HELP_PL["typology"],
)
typology_cols = st.columns([1.15, 0.85])
with typology_cols[0]:
    st.plotly_chart(typology_scatter(latest_typology, reference_year), width="stretch")
with typology_cols[1]:
    segment_counts = (
        latest_typology["pressure_segment"]
        .value_counts()
        .rename_axis("Typ presji")
        .reset_index(name="Liczba krajów")
    )
    st.dataframe(segment_counts, width="stretch", hide_index=True)
    typology_view = latest_typology[
        [
            "country_name",
            "region",
            "pressure_segment",
            "food_affordability_gap_pct",
            "food_share_budget_pct",
            "meal_deprivation_pct",
        ]
    ].sort_values("food_affordability_gap_pct", ascending=False)
    st.dataframe(display_columns(typology_view), width="stretch", hide_index=True)

section_anchor(
    "sec-trends",
    "6. Trendy inflacji żywności",
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
    "7. Dochody i ceny żywności",
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
        "fpi",
    ]
    st.dataframe(
        display_columns(latest[selected_cols].sort_values("food_affordability_gap_pct", ascending=False)),
        width="stretch",
        hide_index=True,
    )

section_anchor(
    "sec-distributions",
    "8. Rozkłady i obserwacje odstające",
    "Zróżnicowanie regionalne oraz skrajne obserwacje kraj-rok.",
    help_text=SECTION_HELP_PL["distributions"],
)
dist_metric = st.selectbox(
    "Metryka rozkładu",
    ["food_affordability_gap_pct", "fpi", "food_inflation_pct", "food_share_budget_pct", "meal_deprivation_pct"],
    format_func=metric_label,
)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(boxplot_region(df_filtered, dist_metric, reference_year, metric_label(dist_metric)), width="stretch")
with col_b:
    st.plotly_chart(histogram(df_filtered, dist_metric, reference_year, metric_label(dist_metric)), width="stretch")

outliers = top_outliers(df_filtered, metric=dist_metric, n=10)
st.dataframe(
    display_columns(outliers[["country_name", "region", "year", dist_metric, "zscore"]]),
    width="stretch",
    hide_index=True,
)
st.caption(
    "Z-score mówi, o ile odchyleń standardowych obserwacja odbiega od średniej. "
    "Im większa wartość bezwzględna, tym bardziej nietypowy kraj-rok w aktualnym wyborze."
)

section_anchor(
    "sec-correlations",
    "9. Korelacje i struktura danych",
    "Zależności między metrykami, p-value, heatmapa i PCA.",
    help_text=SECTION_HELP_PL["correlations"],
)
corr_metrics = [
    "food_affordability_gap_pct",
    "fpi",
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
    ["pearson", "spearman"],
    format_func=lambda method: "Pearson - zależność liniowa" if method == "pearson" else "Spearman - zgodność rang",
    help=(
        "Pearson sprawdza prostą zależność liniową między wartościami. "
        "Spearman sprawdza, czy kraje układają się podobnie w rankingach, nawet gdy zależność nie jest liniowa."
    ),
)
corr_method_label = "Pearson" if corr_method == "pearson" else "Spearman"

col_a, col_b = st.columns(2)
with col_a:
    corr = correlation_matrix(df_filtered, corr_metrics, method=corr_method)
    corr.index = [metric_label(idx) for idx in corr.index]
    corr.columns = [metric_label(col) for col in corr.columns]
    st.plotly_chart(heatmap_corr(corr, f"Korelacje metryk ({corr_method_label})"), width="stretch")
with col_b:
    pvals = pvalue_matrix(df_filtered, corr_metrics, method=corr_method)
    pvals.index = [metric_label(idx) for idx in pvals.index]
    pvals.columns = [metric_label(col) for col in pvals.columns]
    st.dataframe(pvals.map(fmt_p), width="stretch")
    st.caption(
        "Tabela pokazuje p-value po korekcie Holma. Wartości poniżej 0,05 oznaczają, "
        "że zależność jest mało prawdopodobna jako czysty przypadek przy wielu porównaniach."
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
        f"PC1 i PC2 wyjaśniają łącznie {sum(pca_result['explained_variance']) * 100:.1f}% wariancji. "
        "Scree plot pokazuje, ile informacji wnosi każdy kolejny komponent."
    )
except ValueError as exc:
    st.info(str(exc))

section_anchor(
    "sec-tests",
    "10. Testy statystyczne",
    "Różnice regionalne i niepewność estymacji.",
    help_text=SECTION_HELP_PL["tests"],
)
test_metric = st.selectbox(
    "Metryka testowana",
    ["food_affordability_gap_pct", "fpi", "food_inflation_pct", "food_share_budget_pct"],
    format_func=metric_label,
)
test_df = df_filtered[df_filtered["year"] == reference_year].copy()

col_a, col_b = st.columns(2)
with col_a:
    try:
        stats_result = region_anova(test_df, test_metric)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Test": "ANOVA jednoczynnikowa", "Statystyka": stats_result.anova_stat, "p-value": fmt_p(stats_result.anova_p)},
                    {"Test": "Kruskal-Wallis", "Statystyka": stats_result.kruskal_stat, "p-value": fmt_p(stats_result.kruskal_p)},
                    {"Test": "Test Levene'a", "Statystyka": stats_result.levene_stat, "p-value": fmt_p(stats_result.levene_p)},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if not stats_result.pairwise.empty:
            pairwise = stats_result.pairwise.copy()
            pairwise["p_adj"] = pairwise["p_adj"].map(fmt_p)
            pairwise = pairwise.rename(
                columns={
                    "group_A": "Grupa A",
                    "group_B": "Grupa B",
                    "n_A": "N A",
                    "n_B": "N B",
                    "mean_A": "Średnia A",
                    "mean_B": "Średnia B",
                    "median_A": "Mediana A",
                    "median_B": "Mediana B",
                    "p_value": "p-value",
                    "p_adj": "p skorygowane",
                    "cliffs_delta": "Delta Cliffa",
                    "effect_size": "Siła efektu",
                    "significant_05": "Istotne (0,05)",
                }
            )
            for col in ["Grupa A", "Grupa B"]:
                if col in pairwise.columns:
                    pairwise[col] = pairwise[col].map(REGION_LABELS_PL).fillna(pairwise[col])
            if "Siła efektu" in pairwise.columns:
                pairwise["Siła efektu"] = pairwise["Siła efektu"].map(EFFECT_SIZE_LABELS_PL).fillna(pairwise["Siła efektu"])
            st.dataframe(pairwise, width="stretch", hide_index=True)
    except ValueError as exc:
        st.info(str(exc))

with col_b:
    ci = bootstrap_region_means(test_df, test_metric)
    st.plotly_chart(bar_with_ci(ci, f"Średnia regionalna i 95% CI · {metric_label(test_metric)}"), width="stretch")

chi2 = chi_square_high_pressure(test_df, value_col=test_metric)
if chi2:
    st.caption(f"Test chi-kwadrat dla klas presji cenowej według regionów: p-value {fmt_p(chi2['p_value'])}.")
    chi_table = chi2["table"].rename(columns={"low": "niska", "medium": "średnia", "high": "wysoka"}).reset_index()
    st.dataframe(display_columns(chi_table), width="stretch", hide_index=True)

section_anchor(
    "sec-prediction",
    "11. Predykcja",
    "Regresja, efekty stałe panelowe i krótkoterminowa prognoza.",
    help_text=SECTION_HELP_PL["prediction"],
)
tab_reg, tab_panel, tab_forecast = st.tabs(["Regresja", "Efekty stałe panelowe", "Prognoza"])

with tab_reg:
    model_type = st.radio("Model", ["ridge", "random_forest"], format_func=lambda v: "Ridge" if v == "ridge" else "Random Forest", horizontal=True)
    features = st.multiselect(
        "Predyktory",
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
        st.warning("Wzrost dochodu nie powinien być używany jako predyktor w tym samym roku, ponieważ jest częścią wzoru FPI.")
    try:
        reg = fit_fpi_regression(df_filtered, features=features, model_type=model_type)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Test R²", f"{reg['r2_test']:.3f}")
            st.metric("Test MAE", f"{reg['mae_test']:.2f}")
            imp = reg["feature_importance"].copy()
            imp["feature"] = imp["feature"].map(metric_label)
            imp = imp.rename(columns={"feature": "Zmienna", "importance": "Ważność"})
            st.dataframe(imp, width="stretch", hide_index=True)
        with col_b:
            predictions = reg["predictions"].merge(df[["country_code", "year", "region"]], on=["country_code", "year"], how="left")
            st.plotly_chart(residuals_plot(predictions), width="stretch")
            st.dataframe(display_columns(predictions.head(15)), width="stretch", hide_index=True)
    except ValueError as exc:
        st.info(str(exc))

with tab_panel:
    try:
        panel = fit_panel_fixed_effects(df_filtered, features=DEFAULT_FEATURES)
        st.metric("Skorygowane R²", f"{panel['adj_r2']:.3f}", help=panel["formula"])
        coef = panel["coefficients"].copy()
        coef["term"] = coef["term"].map(metric_label)
        coef["p_value"] = coef["p_value"].map(fmt_p)
        coef = coef.rename(columns={"term": "Zmienna", "coef": "Współczynnik", "p_value": "p-value", "std_err": "Błąd standardowy"})
        st.dataframe(coef, width="stretch", hide_index=True)
        with st.expander("Pełny wynik statsmodels", expanded=False):
            st.code(panel["summary_text"], language="text")
    except ValueError as exc:
        st.info(str(exc))

with tab_forecast:
    forecast_country = st.selectbox("Kraj", options=country_options, index=country_options.index("Poland") if "Poland" in country_options else 0)
    periods = st.slider("Horyzont prognozy", 1, 5, 3)
    try:
        forecast_result = forecast_food_inflation(df, forecast_country, periods=periods)
        st.plotly_chart(
            forecast_plot(forecast_result["history"], forecast_result["forecast"], forecast_country),
            width="stretch",
        )
        st.dataframe(display_columns(forecast_result["forecast"]), width="stretch", hide_index=True)
    except ValueError as exc:
        st.info(str(exc))

section_anchor(
    "sec-conclusions",
    "12. Wnioski i ograniczenia",
    "Automatycznie generowane obserwacje dla aktualnych filtrów oraz ograniczenia interpretacji.",
    help_text=SECTION_HELP_PL["conclusions"],
)
top_year = latest.dropna(subset=["food_affordability_gap_pct"]).sort_values("food_affordability_gap_pct", ascending=False).head(5)
top_year_str = ", ".join(top_year["country_name"].tolist()) if not top_year.empty else "brak krajów"
avg_food_vs_headline_gap = latest["food_inflation_pct"].mean() - latest["headline_inflation_pct"].mean()
avg_affordability_gap = latest["food_affordability_gap_pct"].mean()
income_corr = df_filtered[["median_income_eur", "food_inflation_pct"]].corr().iloc[0, 1]
share_corr = df_filtered[["food_share_budget_pct", "food_affordability_gap_pct"]].corr().iloc[0, 1]
cumulative_leader = (
    cumulative_2020_2024.iloc[0]["country_name"]
    if not cumulative_2020_2024.empty
    else "brak danych"
)

st.markdown(
    f"""
- W roku **{reference_year}** najwyższa luka dostępności żywności występuje w krajach: **{top_year_str}**.
- Średnia luka dostępności w wybranych krajach wynosi **{avg_affordability_gap:.2f} p.p.**.
- W wybranych krajach średnia inflacja żywności jest o **{avg_food_vs_headline_gap:.2f} p.p.** wyższa od inflacji ogólnej.
- Korelacja między medianą dochodu a inflacją żywności wynosi **{income_corr:.2f}**.
- Korelacja między udziałem wydatków na żywność a luką dostępności wynosi **{share_corr:.2f}**.
- W analizie skumulowanej 2020-2024 najwyżej w rankingu presji znajduje się: **{cumulative_leader}**.
"""
)

with st.expander("Ograniczenia interpretacji", expanded=False):
    st.markdown(
        """
- Dashboard pokazuje zależności opisowe, a nie dowodzi przyczynowości.
- Dane są zagregowane do poziomu kraju, więc nie pokazują różnic między gospodarstwami domowymi wewnątrz kraju.
- Część braków danych jest uzupełniana interpolacją, dlatego pojedyncze wartości należy traktować jako przybliżenia.
- FPI jest wrażliwy na bardzo niski lub ujemny wzrost dochodu; z tego powodu główną metryką interpretacyjną jest luka dostępności.
- Porównanie 2020-2024 zależy od dostępności danych dla obu lat i nie opisuje pełnej ścieżki zmian między nimi.
"""
    )

if exclusions is not None and not exclusions.empty:
    with st.expander("Wykluczenia danych", expanded=False):
        st.dataframe(display_columns(exclusions), width="stretch", hide_index=True)

section_anchor(
    "sec-export",
    "13. Eksport",
    "Przefiltrowany zbiór danych do dalszej analizy.",
    help_text=SECTION_HELP_PL["export"],
)
export_cols = ["country_name", "country_code", "region", "year", *KEY_METRICS]
export_df = display_columns(df_filtered[export_cols].sort_values(["year", "country_name"]))
st.download_button(
    "Pobierz przefiltrowany CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"europe_food_affordability_{year_range[0]}_{year_range[1]}.csv",
    mime="text/csv",
)
st.dataframe(export_df, width="stretch", hide_index=True)
