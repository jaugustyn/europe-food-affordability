from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from src.config import REQUIRED_ANALYTIC_COLUMNS
from src.etl_pipeline import _quality_profile
from src.eurostat_sources import combine_ppp_sources
from src.pca_analysis import fit_pca
from src.stats_tests import (
    iqr_outliers,
    pvalue_matrix,
    sample_size_matrix,
    select_region_test,
)
from src.transforms import apply_missing_policy
from src.viz import freedman_diaconis_bins


ROOT = Path(__file__).resolve().parents[1]


def _ppp_part(value: float | None, source: str, priority: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_code": ["PL"],
            "year": [2020],
            "food_price_level_index": [value],
            "food_price_level_source": [source],
            "source_priority": [priority],
        }
    )


def test_ppp_prefers_valid_new_value() -> None:
    result = combine_ppp_sources(
        [_ppp_part(110.0, "new", 0), _ppp_part(95.0, "old", 1)]
    ).iloc[0]
    assert result["food_price_level_index"] == 110.0
    assert result["food_price_level_source"] == "new"


def test_ppp_falls_back_when_new_value_is_missing() -> None:
    result = combine_ppp_sources(
        [_ppp_part(None, "new", 0), _ppp_part(95.0, "old", 1)]
    ).iloc[0]
    assert result["food_price_level_index"] == 95.0
    assert result["food_price_level_source"] == "old"


def test_ppp_keeps_missing_when_both_sources_are_missing() -> None:
    result = combine_ppp_sources(
        [_ppp_part(None, "new", 0), _ppp_part(None, "old", 1)]
    ).iloc[0]
    assert pd.isna(result["food_price_level_index"])
    assert pd.isna(result["food_price_level_source"])


def test_generated_data_contracts() -> None:
    main_path = ROOT / "data" / "merged.parquet"
    category_path = ROOT / "data" / "food_categories.parquet"
    quality_path = ROOT / "data" / "data_quality.csv"
    missing_artifacts = [
        str(path.relative_to(ROOT))
        for path in [main_path, category_path, quality_path]
        if not path.exists()
    ]
    assert not missing_artifacts, (
        f"Missing ETL artifacts: {', '.join(missing_artifacts)}. "
        "Run `python etl.py` before the test suite."
    )

    main = pd.read_parquet(main_path)
    categories = pd.read_parquet(category_path)
    quality = pd.read_csv(quality_path)

    assert not main.duplicated(["country_code", "year"]).any()
    assert not categories.duplicated(
        ["country_code", "year", "food_category_code"]
    ).any()
    assert len(categories) >= 1000
    assert categories["food_category_code"].nunique() == 10
    assert len(categories.select_dtypes(include=[np.number]).columns) >= 5
    assert "fpi" not in main.columns
    assert "fpi" not in REQUIRED_ANALYTIC_COLUMNS
    assert {"before_missing_policy", "after_etl"}.issubset(set(quality["stage"]))
    raw_category_metric = quality[
        (quality["view"] == "country_year_category")
        & (quality["stage"] == "raw_grid_before_etl")
        & (quality["column"] == "category_food_inflation_pct")
    ]
    assert len(raw_category_metric) == 1
    raw_row = raw_category_metric.iloc[0]
    assert raw_row["row_count"] == 4950
    assert raw_row["missing_count"] == 50
    assert raw_row["row_count"] - raw_row["missing_count"] == 4900

    imputed_flags = [col for col in main.columns if col.endswith("_imputed")]
    assert imputed_flags
    for flag_col in imputed_flags:
        value_col = flag_col.removesuffix("_imputed")
        report_row = quality[
            (quality["view"] == "country_year")
            & (quality["stage"] == "after_etl")
            & (quality["column"] == value_col)
        ]
        assert len(report_row) == 1
        assert int(report_row.iloc[0]["imputed_count"]) == int(main[flag_col].sum())
    assert not np.isinf(main.select_dtypes(include=[np.number]).to_numpy()).any()


def test_missing_policy_marks_only_values_it_fills() -> None:
    frame = pd.DataFrame(
        {
            "country_code": ["AA"] * 4,
            "year": [2020, 2021, 2022, 2023],
            "strict": [1.0, np.nan, 3.0, np.nan],
            "soft": [np.nan, 2.0, np.nan, 4.0],
        }
    )
    clean, _ = apply_missing_policy(frame, ["strict"], ["soft"])
    assert clean["strict_imputed"].tolist() == [False, True, False]
    assert clean["soft_imputed"].tolist() == [True, False, True]
    profile = _quality_profile(clean, "test", "after_etl")
    counts = profile.set_index("column")["imputed_count"]
    assert counts["strict"] == 1
    assert counts["soft"] == 2


def test_iqr_detects_only_extreme_observation() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 100]})
    result = iqr_outliers(frame, "value")
    assert result["value"].tolist() == [100]
    assert result["iqr_distance"].iloc[0] > 0


def test_histogram_bins_use_fd_and_sturges_fallback() -> None:
    assert freedman_diaconis_bins(np.arange(100)) > 1
    assert freedman_diaconis_bins(np.ones(16)) == 5


def test_pairwise_correlation_matrices_use_complete_pairs() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, np.nan, 3.0, 4.0], "c": [4, 3, 2, 1]}
    )
    sizes = sample_size_matrix(frame, ["a", "b", "c"])
    pvalues = pvalue_matrix(frame, ["a", "b", "c"], method="spearman")
    assert sizes.loc["a", "b"] == 3
    assert sizes.loc["a", "c"] == 4
    assert pvalues.loc["a", "c"] == pvalues.loc["c", "a"]
    assert pvalues.loc["a", "a"] == 0


def _pca_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = rng.normal(size=n)
    return pd.DataFrame(
        {
            "food_inflation_pct": base + rng.normal(scale=0.05, size=n),
            "headline_inflation_pct": base + rng.normal(scale=0.05, size=n),
            "food_share_budget_pct": -base + rng.normal(scale=0.1, size=n),
            "median_income_eur": 20_000 + base * 2_000 + rng.normal(scale=100, size=n),
            "food_price_level_index": 100 + base * 5 + rng.normal(scale=0.5, size=n),
            "meal_deprivation_pct": -base + rng.normal(scale=0.1, size=n),
            "country_name": [f"C{i % 20}" for i in range(n)],
            "year": 2010 + np.arange(n) % 15,
            "region": ["Northern", "Western", "Southern", "Eastern"] * (n // 4),
        }
    )


def test_pca_selects_smallest_k_reaching_threshold() -> None:
    result = fit_pca(_pca_frame(), variance_threshold=0.80)
    cumulative = np.asarray(result["cumulative_variance_full"])
    k = result["selected_k"]
    assert cumulative[k - 1] >= 0.80
    if k > 1:
        assert cumulative[k - 2] < 0.80
    assert list(result["loadings"].columns) == ["feature", "PC1", "PC2"]


def _regional_frame(groups: list[np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame({"region": f"R{i}", "value": values}) for i, values in enumerate(groups)],
        ignore_index=True,
    )


def test_region_workflow_selects_anova_and_tukey() -> None:
    normal = stats.norm.ppf((np.arange(1, 11) - 0.5) / 10)
    frame = _regional_frame([normal + shift for shift in [0.0, 1.0, 2.0, 3.0]])
    result = select_region_test(frame, "value")
    assert result["method"] == "anova"
    assert result["reject_h0"] is True
    assert result["effect_name"] == "omega_squared"
    assert "średnie" in result["hypothesis_null"]
    assert result["group_statistic_name"] == "średnia"
    assert "średnimi" in result["interpretation"]
    assert not result["posthoc"].empty
    assert list(result["posthoc"].columns) == [
        "group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"
    ]


def test_region_workflow_selects_kruskal_for_non_normal_groups() -> None:
    skewed = np.array([0, 0, 0, 0, 0.1, 0.2, 0.3, 5.0])
    frame = _regional_frame([skewed + shift for shift in [0.0, 1.0, 2.0, 3.0]])
    result = select_region_test(frame, "value")
    assert result["method"] == "kruskal"
    assert result["effect_name"] == "epsilon_squared"
    assert "rozkłady" in result["hypothesis_null"]
    assert result["group_statistic_name"] == "mediana"
    assert "nie jest automatycznie" in result["interpretation"]


def test_posthoc_is_blocked_after_non_significant_global_test() -> None:
    normal = stats.norm.ppf((np.arange(1, 11) - 0.5) / 10)
    frame = _regional_frame([normal.copy() for _ in range(4)])
    result = select_region_test(frame, "value")
    assert result["reject_h0"] is False
    assert result["posthoc"].empty


def test_streamlit_app_smoke() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    rendered = "\n".join(
        str(element.value)
        for kind in ["title", "header", "subheader", "markdown"]
        for element in getattr(app, kind)
    )
    for section in [
        "1. KPI",
        "2. Struktura i jakość danych",
        "6. Szczegółowe kategorie żywności",
        "10. Korelacje i PCA",
        "11. Testy statystyczne",
        "13. Eksport",
    ]:
        assert section in rendered


def _element_by_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def _main_test_name(app: AppTest) -> str:
    table = next(
        element.value
        for element in app.dataframe
        if "Test główny" in element.value.columns
    )
    return str(table.iloc[0]["Test główny"])


def test_streamlit_interactions_cover_methods_filters_and_posthoc_gate() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    correlation = _element_by_label(app.selectbox, "Metoda korelacji")
    assert correlation.value == "spearman"
    correlation.set_value("pearson").run()
    assert not app.exception
    assert _element_by_label(app.selectbox, "Metoda korelacji").value == "pearson"
    assert any("Pearson" in caption.value for caption in app.caption)

    tested_metric = _element_by_label(app.selectbox, "Metryka testowana")
    tested_metric.set_value("food_inflation_pct").run()
    assert _main_test_name(app) == "ANOVA jednoczynnikowa"
    assert any("średnie regionalne" in item.value for item in app.markdown)
    assert any("Post-hoc pominięto" in item.value for item in app.caption)
    assert not any(
        "Post-hoc wykonany" in item.value for item in app.markdown
    )

    _element_by_label(app.selectbox, "Metryka testowana").set_value(
        "food_affordability_gap_pct"
    ).run()
    assert _main_test_name(app) == "Kruskal–Wallis"
    assert any("rozkłady i rangi" in item.value for item in app.markdown)
    assert any("Post-hoc wykonany" in item.value for item in app.markdown)

    region_filter = _element_by_label(app.multiselect, "Regiony")
    region_filter.set_value(["Eastern", "Western"]).run()
    assert _element_by_label(app.multiselect, "Regiony").value == ["Eastern", "Western"]
    country_filter = _element_by_label(app.multiselect, "Kraje")
    selected_countries = country_filter.options[:4]
    country_filter.set_value(selected_countries).run()
    assert set(_element_by_label(app.multiselect, "Kraje").value) == set(selected_countries)
    assert not app.exception

    year_filter = _element_by_label(app.slider, "Zakres lat")
    year_filter.set_range(2015, 2023).run()
    assert not app.exception
    assert any(
        "wymaga, aby globalny filtr lat obejmował jednocześnie 2020 i 2024"
        in item.value
        for item in app.info
    )
