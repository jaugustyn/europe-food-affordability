from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    load_category_data,
    load_data,
    load_data_quality,
    load_exclusions,
)
from src.metrics import KEY_METRICS, METRICS, fmt
from src.pca_analysis import fit_pca
from src.stats_tests import (
    bootstrap_region_means,
    correlation_matrix,
    iqr_outliers,
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


st.set_page_config(
    page_title="Dostępność Żywności w Europie",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Dostępność Żywności w Europie"

REGION_LABELS_PL = {
    "Northern": "Europa Północna",
    "Western": "Europa Zachodnia",
    "Southern": "Europa Południowa",
    "Eastern": "Europa Wschodnia",
}

METRIC_LABELS_PL = {
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
    "category_food_inflation_pct": "Inflacja kategorii żywności",
    "category_affordability_gap_pct": "Luka dostępności kategorii",
}

METRIC_DESCRIPTIONS_PL = {
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
    "category_food_inflation_pct": "Roczna średnia zmiana HICP dla wybranej szczegółowej kategorii żywności.",
    "category_affordability_gap_pct": "Inflacja wybranej kategorii minus wzrost mediany dochodu w tym samym kraju i roku.",
}

METRIC_DIRECTIONS_PL = {
    "food_inflation_pct": "Wyższa wartość oznacza szybszy wzrost cen żywności i większe obciążenie konsumentów.",
    "headline_inflation_pct": "Wyższa wartość oznacza szybszy wzrost ogólnego poziomu cen w gospodarce.",
    "food_share_budget_pct": "Wyższa wartość oznacza większą wrażliwość budżetu gospodarstw domowych na ceny żywności.",
    "median_income_eur": "Wyższa wartość oznacza większy nominalny bufor dochodowy gospodarstw domowych.",
    "income_growth_pct": "Wyższa wartość oznacza szybszy wzrost dochodów i większą zdolność kompensowania wzrostu cen.",
    "minimum_wage_eur_month": "Wyższa wartość oznacza wyższą nominalną płacę minimalną; pełna ocena wymaga zestawienia z poziomem cen.",
    "food_price_level_index": "Wyższa wartość oznacza wyższy poziom cen żywności względem średniej UE.",
    "meal_deprivation_pct": "Wyższa wartość oznacza większy odsetek osób bez możliwości regularnego pełnowartościowego posiłku.",
    "food_affordability_gap_pct": "Wyższa wartość oznacza gorszą dostępność. Wartość dodatnia wskazuje, że ceny żywności rosną szybciej niż dochody.",
    "food_inflation_index_2020": "Wyższa wartość oznacza wyższy skumulowany poziom cen żywności względem 2020 roku.",
    "food_price_growth_2020_2024_pct": "Wyższa wartość oznacza silniejszy skumulowany wzrost cen żywności od 2020 roku.",
    "income_growth_2020_2024_pct": "Wyższa wartość oznacza silniejszy skumulowany wzrost dochodów od 2020 roku.",
    "cumulative_affordability_gap_pct": "Wyższa wartość oznacza gorszą dostępność. Wartość dodatnia wskazuje, że od 2020 roku ceny żywności wzrosły mocniej niż dochody.",
    "category_food_inflation_pct": "Wyższa wartość oznacza szybszy wzrost cen danej kategorii.",
    "category_affordability_gap_pct": "Dodatnia wartość oznacza, że ceny kategorii rosły szybciej niż dochód.",
}

CUSTOM_METRIC_UNITS = {
    "food_price_growth_2020_2024_pct": "%",
    "income_growth_2020_2024_pct": "%",
    "cumulative_affordability_gap_pct": "p.p.",
    "category_food_inflation_pct": "%",
    "category_affordability_gap_pct": "p.p.",
}

MAP_METRICS = [
    "food_affordability_gap_pct",
    "food_inflation_pct",
    "food_share_budget_pct",
    "food_price_level_index",
    "meal_deprivation_pct",
]

METRIC_REFERENCE = [
    "food_affordability_gap_pct",
    "food_inflation_pct",
    "headline_inflation_pct",
    "median_income_eur",
    "income_growth_pct",
    "food_share_budget_pct",
    "food_price_level_index",
    "meal_deprivation_pct",
    "food_inflation_index_2020",
    "food_price_growth_2020_2024_pct",
    "income_growth_2020_2024_pct",
    "cumulative_affordability_gap_pct",
]

SECTION_HELP_PL = {
    "kpi": "KPI to szybkie podsumowanie sytuacji w roku referencyjnym. Najważniejsza jest luka dostępności: inflacja żywności minus wzrost dochodu. Dodatni wynik mówi, że ceny żywności rosły szybciej niż dochody.",
    "quality": "Sekcja pokazuje ziarno obu widoków, typy kolumn, braki danych przed i po ETL oraz statystyki opisowe po aktualnych filtrach.",
    "drivers": "Ta sekcja rozkłada wynik kraju na podstawowe składniki: inflację żywności, wzrost dochodu, lukę cen-dochód, udział żywności w wydatkach oraz deprywację posiłku. Dzięki temu ranking nie jest tylko listą krajów, ale ma wyjaśnienie.",
    "map": "Mapa pokazuje, które kraje mają wysokie lub niskie wartości wybranej metryki w konkretnym roku. Szare kraje nie mają kompletnej wartości dla tej metryki i roku. Skala kolorów jest stała dla aktualnych filtrów, więc ten sam kolor oznacza porównywalną wartość przy zmianie roku.",
    "cumulative": "Porównanie 2020-2024 pokazuje efekt skumulowany, a nie tylko jeden rok. Dodatnia skumulowana luka oznacza, że od 2020 roku ceny żywności wzrosły bardziej niż mediana dochodu.",
    "categories": "Widok kraj-rok-kategoria zawiera wyłącznie zaobserwowane wartości HICP i służy do porównywania szczegółowych grup żywności.",
    "trends": "Trend pokazuje zmianę w czasie. Linia inflacji żywności mówi, jak szybko rosły ceny jedzenia, a linia luki dostępności pokazuje, czy tempo wzrostu cen było większe niż tempo wzrostu dochodów.",
    "income": "Każda kropka to kraj w roku referencyjnym. Oś pozioma pokazuje dochód, oś pionowa inflację żywności, a rozmiar kropki udział żywności w wydatkach. Linia OLS to prosta tendencja: pomaga zobaczyć ogólny kierunek relacji, ale nie dowodzi przyczynowości.",
    "distributions": "Box plot i histogram opisują rok referencyjny. Tabela IQR wskazuje obserwacje poza granicami Q1−1,5×IQR i Q3+1,5×IQR.",
    "correlations": "Korelacja mieści się od -1 do 1: wartości blisko 1 rosną razem, blisko -1 poruszają się przeciwnie, a blisko 0 nie mają prostej zależności. p-value pomaga ocenić, czy zależność może być przypadkowa. PCA streszcza kilka podobnych metryk do dwóch osi, żeby zobaczyć podobieństwa krajów.",
    "tests": "Testy statystyczne sprawdzają, czy różnice między regionami są większe niż losowe wahania danych. p-value poniżej 0,05 traktujemy jako sygnał istotności, a przedziały ufności pokazują niepewność średnich.",
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
    "food_category_code": "Kod kategorii",
    "food_category_name": "Kategoria żywności",
    "iqr_distance": "Odległość poza granicą IQR",
    "iqr_lower": "Dolna granica IQR",
    "iqr_upper": "Górna granica IQR",
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


def metric_direction(metric: str) -> str:
    return METRIC_DIRECTIONS_PL.get(metric, "Kierunek interpretacji zależy od kontekstu.")


def metric_unit(metric: str) -> str:
    if metric in METRICS:
        return METRICS[metric]["unit"]
    return CUSTOM_METRIC_UNITS.get(metric, "")


def metric_help(metric: str) -> str:
    desc = metric_description(metric)
    direction = metric_direction(metric)
    return f"{desc} {direction}".strip()


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


def country_list(frame: pd.DataFrame, value_col: str, n: int = 3, ascending: bool = False) -> str:
    rows = frame.dropna(subset=[value_col]).sort_values(value_col, ascending=ascending).head(n)
    if rows.empty:
        return "brak danych"
    return ", ".join(f"{row['country_name']} ({row[value_col]:+.1f})" for _, row in rows.iterrows())


def interpret_current_situation(current_frame: pd.DataFrame, current_year: int) -> list[str]:
    """Generate data-driven conclusions for the latest year available in the dashboard dataset."""
    current = current_frame.dropna(subset=["food_affordability_gap_pct"]).copy()
    if current.empty:
        return [f"Dla roku {current_year} brakuje kompletnych danych do oceny bieżącej sytuacji."]

    n_countries = len(current)
    positive_gap = current[current["food_affordability_gap_pct"] > 0]
    median_gap = current["food_affordability_gap_pct"].median()
    median_food = current["food_inflation_pct"].median()
    median_income_growth = current["income_growth_pct"].median()

    if len(positive_gap) == 0:
        pressure_note = (
            f"W ostatnim roku danych (**{current_year}**) krótkoterminowa presja cenowa jest ograniczona w tym przekroju: "
            f"w żadnym z {n_countries} wybranych krajów ceny żywności nie zwiększyły luki względem dochodów. "
            f"Mediana luki wynosi {median_gap:+.1f} p.p."
        )
    else:
        pressure_note = (
            f"W ostatnim roku danych (**{current_year}**) bieżąca presja jest skoncentrowana, a nie powszechna: "
            f"dodatnią lukę dostępności ma {len(positive_gap)} z {n_countries} wybranych krajów. "
            f"Najbardziej widoczne wyjątki to {country_list(positive_gap, 'food_affordability_gap_pct', n=3)} p.p."
        )

    typical_note = (
        f"Typowa sytuacja w {current_year} r. wskazuje na fazę wygaszania szoku inflacyjnego: "
        f"mediana inflacji żywności wynosi {median_food:+.1f}%, a mediana wzrostu dochodu {median_income_growth:+.1f}%. "
        "Nie oznacza to jednak pełnego powrotu dostępności, bo poziom cen po latach 2021-2023 pozostaje podwyższony."
    )

    deprivation_names = current_frame.nlargest(3, "meal_deprivation_pct")["country_name"].tolist()
    budget_names = current_frame.nlargest(3, "food_share_budget_pct")["country_name"].tolist()
    vulnerability_note = (
        f"Najwyższą deprywację posiłku mają: **{', '.join(deprivation_names)}**; "
        f"najwyższy udział żywności w wydatkach mają: **{', '.join(budget_names)}**. "
        "Są to dwa jawne rankingi, a nie arbitralny indeks syntetyczny."
    )

    return [pressure_note, typical_note, vulnerability_note]


def interpret_cumulative_pressure(cumulative_frame: pd.DataFrame) -> list[str]:
    if cumulative_frame.empty:
        return ["Brak kompletnych danych dla porównania 2020-2024 w aktualnie wybranym zakresie krajów."]

    top = cumulative_frame.sort_values("cumulative_affordability_gap_pct", ascending=False).head(3)
    bottom = cumulative_frame.sort_values("cumulative_affordability_gap_pct").head(3)
    median_cum_gap = cumulative_frame["cumulative_affordability_gap_pct"].median()
    top_names = ", ".join(
        f"{row.country_name} ({row.cumulative_affordability_gap_pct:+.1f} p.p.)"
        for row in top.itertuples()
    )
    bottom_names = ", ".join(
        f"{row.country_name} ({row.cumulative_affordability_gap_pct:+.1f} p.p.)"
        for row in bottom.itertuples()
    )

    notes = [
        (
            "Perspektywa 2020-2024 pokazuje, czy gospodarstwa zdążyły skompensować wcześniejszy szok cenowy. "
            f"Mediana skumulowanej luki w wybranych krajach wynosi {median_cum_gap:+.1f} p.p."
        ),
        (
            f"Największy nierozwiązany efekt skumulowany widać w krajach: **{top_names}**. "
            "W tych przypadkach poprawa w ostatnim roku nie kompensuje całej skumulowanej różnicy między wzrostem cen żywności a wzrostem dochodów."
        ),
        (
            f"Najsilniejsze nominalne doganianie dochodów względem cen żywności widać w krajach: **{bottom_names}**. "
            "Ta poprawa nie musi oznaczać wysokiej dostępności, jeżeli punkt startowy dochodów był niski."
        ),
    ]

    regional = cumulative_frame.groupby("region")["cumulative_affordability_gap_pct"].median().dropna()
    if not regional.empty:
        highest_region = regional.sort_values(ascending=False).index[0]
        lowest_region = regional.sort_values().index[0]
        notes.append(
            f"Regionalnie najwyższa mediana skumulowanej presji występuje w grupie "
            f"**{REGION_LABELS_PL.get(highest_region, highest_region)}**, a najniższa w grupie "
            f"**{REGION_LABELS_PL.get(lowest_region, lowest_region)}**. "
            "To sugeruje, że bieżący ranking z jednego roku warto zawsze czytać razem z trendem wieloletnim."
        )

    return notes


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Full descriptive statistics for numeric analytical variables."""
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["year"], errors="ignore")
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.agg(["count", "mean", "median", "std", "min", "max"]).T
    out["q1"] = numeric.quantile(0.25)
    out["q3"] = numeric.quantile(0.75)
    return out[["count", "mean", "median", "std", "q1", "q3", "min", "max"]].reset_index(
        names="column"
    )


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


try:
    df = get_data()
    category_df = get_category_data()
    data_quality = get_data_quality()
except FileNotFoundError:
    st.error("Brakuje aktualnych plików danych. Uruchom `python etl.py`, aby odtworzyć oba widoki i raport jakości.")
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
category_filtered = category_df[
    category_df["year"].between(year_range[0], year_range[1])
    & category_df["region"].isin(regions)
    & category_df["country_name"].isin(countries)
].copy()
scope_df = df[df["region"].isin(regions) & df["country_name"].isin(countries)].copy()
latest = df_filtered[df_filtered["year"] == reference_year].copy()

st.title(APP_TITLE)
st.caption(
    "Dashboard do porównywania presji cen żywności, inflacji ogólnej, dochodów gospodarstw "
    "domowych i dostępności żywności w krajach Europy na podstawie danych Eurostatu."
)
st.info(
    "Podstawowa interpretacja: **dodatnia luka dostępności żywności** oznacza, że ceny żywności "
    "rosną szybciej niż dochody, co wskazuje na pogorszenie dostępności. W przypadku dochodu "
    "i wzrostu dochodu wyższe wartości oznaczają większy bufor finansowy gospodarstw domowych."
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

has_full_cumulative_period = year_range[0] <= 2020 and year_range[1] >= 2024
cumulative_2020_2024 = (
    cumulative_pressure_summary(scope_df) if has_full_cumulative_period else pd.DataFrame()
)

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
driver_country = st.selectbox(
    "Kraj do diagnozy",
    options=driver_options,
    index=driver_options.index(default_driver) if default_driver in driver_options else 0,
)
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
    selected_category = st.selectbox("Kategoria żywności", category_options)
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

col_a, col_b = st.columns(2)
with col_a:
    corr = correlation_matrix(latest, corr_metrics, method=corr_method)
    corr.index = [metric_label(idx) for idx in corr.index]
    corr.columns = [metric_label(col) for col in corr.columns]
    st.plotly_chart(heatmap_corr(corr, f"Korelacje metryk ({corr_method_label})"), width="stretch")
with col_b:
    pvals = pvalue_matrix(latest, corr_metrics, method=corr_method)
    pvals.index = [metric_label(idx) for idx in pvals.index]
    pvals.columns = [metric_label(col) for col in pvals.columns]
    st.dataframe(pvals.map(fmt_p), width="stretch")
    pair_n = sample_size_matrix(latest, corr_metrics)
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
    "Korelacja nie oznacza przyczynowości."
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
test_df = df_filtered[df_filtered["year"] == reference_year].copy()

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
    st.plotly_chart(bar_with_ci(ci, f"Średnia regionalna i 95% CI · {metric_label(test_metric)}"), width="stretch")


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
    "income_growth_pct",
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
