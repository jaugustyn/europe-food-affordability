# Europe Food Affordability

Interactive Streamlit project analysing food-price pressure and household affordability across Europe.

**Analytical question:** which countries and food categories face the strongest price pressure, and how does it relate to headline inflation, household income and social vulnerability?

The main interpretation metric is the **Food Affordability Gap**:

```text
food inflation − median income growth
```

A positive value means that food prices grew faster than nominal median income. It is a transparent pressure proxy, not a causal estimate or a complete measure of the cost of a healthy diet.

## How to run

```powershell
cd europe-food-affordability
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python etl.py
streamlit run app.py
```

The dashboard is available at `http://localhost:8501` by default.

## Required acceptance sequence

Run acceptance checks in this order so that tests always use freshly generated
artifacts:

```powershell
python etl.py
python -m py_compile app.py etl.py src/config.py src/data_loader.py src/etl_pipeline.py src/eurostat_sources.py src/metrics.py src/pca_analysis.py src/stats_tests.py src/transforms.py src/viz.py
python -m pytest -q
python -m pytest -q tests/test_methodology.py -k streamlit
```

Missing ETL artifacts are a test failure, not a skipped test. The final command
reruns the Streamlit smoke and interaction checks explicitly.

## Analytical grains

The ETL deliberately maintains two different analytical views.

| View | File | Unique key | Current size | Purpose |
| --- | --- | --- | ---: | --- |
| country–year | `data/merged.parquet` | `country_code, year` | 480 | KPI, maps, cumulative metrics, correlations, PCA and regional tests |
| country–year–category | `data/food_categories.parquet` | `country_code, year, food_category_code` | 4,800 | category-level EDA, rankings, distributions and trends |

The category view satisfies the course requirement of at least 1,000 genuine observations. Country-level income and social indicators are contextual variables in this view. They are not treated as 4,800 independent observations in correlations or statistical tests.

The category pipeline distinguishes three counts: the raw Eurostat grid has 4,950
country–year–category rows, including 50 missing HICP values; 4,900 HICP values are
observed; and 4,800 rows remain in the final analytical view after joining to valid
country–year context. Missing category HICP values are never interpolated.

## Food categories

The category view uses observed annual HICP rates from `prc_hicp_aind` with unit `RCH_A_AVG`.

| Code | Category |
| --- | --- |
| `CP0111` | Bread and cereals |
| `CP0112` | Meat |
| `CP0113` | Fish and seafood |
| `CP0114` | Milk, cheese and eggs |
| `CP0115` | Oils and fats |
| `CP0116` | Fruit |
| `CP0117` | Vegetables |
| `CP0118` | Sugar, jam, honey, chocolate and confectionery |
| `CP0119` | Food products n.e.c. |
| `CP012` | Non-alcoholic beverages |

Category HICP values are not interpolated. The category view contains only observed Eurostat values joined to valid country–year context.

## Data sources

| Eurostat dataset | Project variables | Contents |
| --- | --- | --- |
| `prc_hicp_aind` | `food_inflation_pct`, `headline_inflation_pct`, `category_food_inflation_pct` | annual HICP rates for CP01, CP00 and detailed food categories |
| `ilc_di03` | `median_income_eur`, `income_growth_pct` | median equivalised net income and annual growth |
| `earn_mw_cur` | `minimum_wage_eur_month` | monthly minimum wage; semi-annual values averaged to years |
| `prc_ppp_ind_1`, fallback `prc_ppp_ind` | `food_price_level_index` | food price level relative to EU27_2020=100 |
| `nama_10_co3_p3` | `food_share_budget_pct` | food share of household final consumption |
| `ilc_mdes03` | `meal_deprivation_pct` | inability to afford a proper meal every second day |

All source tables are reshaped to long form and normalised to ISO-2 country codes. The primary merge key is `country_code, year`. The category dimension is added only to the dedicated category view.

## ETL and data quality

Raw downloads are cached in `data/raw/*.parquet`. ETL writes:

- `data/merged.parquet`;
- `data/food_categories.parquet`;
- `data/data_quality.csv` with column types, missingness and imputation counts before and after ETL;
- `data/exclusions.csv` with removed country–year observations;
- `etl.log` with source and validation diagnostics.

### Missing-data policy

| Situation | Rule |
| --- | --- |
| gaps up to 2 years in HICP CP01, HICP CP00 or median income | linear interpolation inside each country series |
| remaining gaps in strict variables | row excluded and logged |
| soft variables | interpolation plus limited `ffill`/`bfill` up to 3 years |
| category HICP | no imputation; observed values only |

Every interpolated source variable has a matching Boolean `*_imputed` column. These
flags identify the exact country–year cells filled by the missing-data policy and are
also carried into the category view for its repeated contextual variables.
Flags are propagated to derived income-growth and affordability-gap metrics. Imputed
values remain available for descriptive maps and trends, but are treated as missing in
correlations, regional hypothesis tests and PCA.

The PPP adapter prefers a non-null value from `prc_ppp_ind_1` and falls back at the value level to `prc_ppp_ind`. `food_price_level_source` records the source; values filled by the missing-data policy are labelled `interpolated`.

ETL rejects outputs with duplicate keys, fewer than five numeric variables, years outside 2010–2024, infinite values or fewer than 1,000 category observations.

## Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| `food_affordability_gap_pct` | food inflation − income growth | positive means aggregate food prices outpaced nominal income |
| `category_affordability_gap_pct` | category inflation − income growth | positive means the selected category outpaced nominal income |
| `food_inflation_index_2020` | compound HICP CP01 index, 2020=100 | accumulated food-price level relative to 2020 |
| `food_price_growth_2020_2024_pct` | change in the compound food index | accumulated food-price growth |
| `income_growth_2020_2024_pct` | change in median income | accumulated nominal income growth |
| `cumulative_affordability_gap_pct` | food-price growth − income growth | positive means food prices outpaced income over 2020–2024 |

The unstable Food Pressure Index ratio was removed from ETL and all analytical outputs.

## Statistical methodology

### EDA and anomalies

The dashboard reports both data grains, column types, missing values before and after ETL, and descriptive statistics: count, mean, median, standard deviation, quartiles, minimum and maximum.

Histograms use the Freedman–Diaconis rule with a Sturges fallback. Anomalies are detected in the reference-year cross-section using Tukey's `1.5 × IQR` fences. An anomaly is an unusual observation, not automatically a data error.

### Correlations

Correlations use one country observation in the selected reference year. Spearman is the default and Pearson is optional. Imputed cells are masked before calculation. Pairwise-complete samples, pair counts and Holm-adjusted p-values are reported. The affordability gap is excluded from this matrix because it is constructed from food inflation and income growth.

### PCA

PCA is fitted on standardised country–year indicators after removing rows containing an imputed PCA feature. The selected dimensionality is the smallest number of components explaining at least 80% of cumulative variance. The dashboard reports individual and cumulative variance, the selected `k`, complete-case count, number of excluded imputed rows, correlation loadings and a PC1–PC2 biplot.

### Regional tests

Regional inference uses one country observation in the reference year, observed values only and `α = 0.05`. Imputed observations are excluded separately for the selected metric and their count is reported.

1. Shapiro–Wilk is checked within regions.
2. Brown–Forsythe/Levene uses median centring.
3. If all normality checks and equal variances pass, the main test is one-way ANOVA with omega squared.
4. Otherwise the main test is Kruskal–Wallis with epsilon squared.
5. The dashboard states an explicit decision about H0.
6. Post-hoc runs only after a significant global result: Tukey HSD after ANOVA or pairwise Mann–Whitney with Holm correction and Cliff's delta after Kruskal–Wallis.

Kruskal–Wallis and Mann–Whitney are interpreted as distribution/rank comparisons, not universal tests of medians.

## Dashboard structure

1. KPI
2. Data structure and quality
3. Country diagnosis
4. Europe map
5. Cumulative pressure 2020–2024
6. Detailed food categories
7. Time trends
8. Income and food prices
9. Distributions and IQR anomalies
10. Reference-year correlations and PCA
11. Statistical tests
12. Conclusions and limitations
13. Export of both analytical grains

Global year, region and country filters affect both data views. The category selector is local to the category section. The global ETL audit intentionally remains independent of dashboard filters. The cumulative 2020–2024 section is calculated only when both endpoints are included in the selected year range.

## Project structure

```text
europe-food-affordability/
├── README.md
├── requirements.txt
├── etl.py
├── app.py
├── src/
│   ├── config.py
│   ├── eurostat_sources.py
│   ├── transforms.py
│   ├── etl_pipeline.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── stats_tests.py
│   ├── pca_analysis.py
│   ├── viz.py
│   └── dashboard/
│       ├── layout.py
│       ├── state.py
│       ├── support.py
│       └── sections/
│           ├── overview.py
│           ├── exploration.py
│           ├── methodology.py
│           └── output.py
├── tests/
│   └── test_methodology.py
└── data/
    ├── regions.csv
    ├── raw/
    ├── merged.parquet
    ├── food_categories.parquet
    ├── data_quality.csv
    └── exclusions.csv
```

## Scope and limitations

- Results are descriptive and comparative, not causal.
- Country aggregates do not represent within-country household inequality.
- Interpolation can smooth abrupt changes; category HICP is therefore kept observed-only, and all imputations are excluded from inferential analyses.
- Median income in EUR can be affected by exchange-rate changes outside the euro area.
- Regional groups are analytical groupings with small sample sizes.
- Pairwise-complete samples can differ between correlations.
- The 2020–2024 endpoint comparison does not replace inspection of the full time series.

Ridge, Random Forest, panel fixed-effects regression, ARIMA, rule-based country typology and the synthetic vulnerability score were deliberately excluded because they were optional and could not be validated strongly enough within the project's scope.
