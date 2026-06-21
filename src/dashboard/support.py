"""Dashboard copy, formatting and pure analytical helpers."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from src.metrics import METRICS, fmt


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


def region_label(region: object) -> str:
    """Return a Polish region label for UI controls and tables."""
    key = str(region)
    return REGION_LABELS_PL.get(key, key)


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
        out["Region"] = out["Region"].map(region_label)
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


def fmt_value(value: object, metric: str) -> str:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return "—"
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return "—"
    if metric in METRICS:
        return fmt(numeric_value, metric)
    return f"{numeric_value:+.1f}%"


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


