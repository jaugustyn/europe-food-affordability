# Europe Food Affordability

Interactive data-processing and visualisation project focused on food price affordability across Europe.

**Analytical question:** which European countries face the highest food price pressure, and how does it relate to headline inflation and household income?

The dashboard combines annual Eurostat data for European countries and builds a **Food Pressure Index (FPI)** comparing food inflation with median household income growth.

## How To Run

```powershell
# 1. Environment (Python 3.11+)
cd europe-food-affordability
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. ETL: downloads Eurostat data and writes data/merged.parquet
python etl.py

# 3. Dashboard
streamlit run app.py
```

The app runs at `http://localhost:8501` by default.

ETL caches raw downloads in `data/raw/*.parquet`, writes the analytical table to `data/merged.parquet`, and logs excluded observations to `data/exclusions.csv`.

## Data Sources

| # | Eurostat dataset | Project columns | Contents |
| --- | --- | --- | --- |
| 1 | `prc_hicp_aind` | `food_inflation_pct`, `headline_inflation_pct` | annual food inflation (CP01) and all-items inflation (CP00) |
| 2 | `ilc_di03` | `median_income_eur`, `income_growth_pct` | median equivalised net income and year-over-year income growth |
| 3 | `earn_mw_cur` | `minimum_wage_eur_month` | monthly minimum wage in EUR; semi-annual data averaged to years |
| 4 | `prc_ppp_ind_1` with fallback to `prc_ppp_ind` | `food_price_level_index` | food price level index relative to EU27_2020=100 |
| 5 | `nama_10_co3_p3` | `food_share_budget_pct` | food share in household final consumption expenditure |
| 6 | `ilc_mdes03` | `meal_deprivation_pct` | share of people unable to afford a proper meal every second day |

The project satisfies the requirement of at least three data sources while keeping all inputs thematically aligned around prices, income, and food affordability.

## Merge Logic

Common key:

```python
["country_code", "year"]
```

Pipeline:

```text
HICP food/headline
  -> median income + minimum wage
  -> PPP food price level + food expenditure share
  -> meal deprivation
  -> outer merge on country_code/year
  -> missing-data policy
  -> derived metrics
```

Country codes are normalised to ISO-2. `regions.csv` stores `country_code`, `iso3`, `country_name`, and `region`.

## Missing-Data Policy

| Situation | Rule |
| --- | --- |
| gaps up to 2 years in hard columns: HICP and median income | linear interpolation by country |
| gaps longer than 2 years in hard columns | row excluded and logged to `data/exclusions.csv` |
| soft columns: PPP, expenditure share, meal deprivation, minimum wage | interpolation plus short `ffill`/`bfill` up to 3 years |

## Metrics

| Metric | Type | Description |
| --- | --- | --- |
| `fpi` | synthetic ratio | `food_inflation_pct / income_growth_pct`; values above 1 mean food prices grew faster than income |
| `food_inflation_pct` | % YoY | annual HICP CP01 food inflation |
| `headline_inflation_pct` | % YoY | annual HICP CP00 all-items inflation |
| `median_income_eur` | EUR/year | median equivalised net income |
| `income_growth_pct` | % YoY | country-level median income growth |
| `food_share_budget_pct` | % | food share in household final consumption |
| `food_price_level_index` | EU27_2020=100 | relative food price level |
| `meal_deprivation_pct` | % of population | inability to afford a proper meal every second day |
| `food_affordability_gap_pct` | percentage points | `food_inflation_pct - income_growth_pct` |
| `food_inflation_index_2020` | index | cumulative food price index with 2020=100 |

When `income_growth_pct` is close to zero, ETL sets FPI to missing to avoid unstable division.

## Dashboard Structure

| # | Section | Visualisation | Filters |
| --- | --- | --- | --- |
| 1 | KPI Bar | scorecards for highest FPI, average food inflation, food budget share, median FPI | year |
| 2 | Europe Map | Plotly choropleth | year, metric |
| 3 | Time Trends | multi-country line charts | countries, years, metric |
| 4 | Income vs Food Inflation | scatter with OLS trend | regions, years |
| 5 | Distributions and Anomalies | box plot, histogram, IQR outlier table | metric |
| 6 | Correlations and PCA | Pearson/Spearman heatmap, Holm-adjusted p-values, country-year heatmap, PCA biplot | method |
| 7 | Statistical Tests | ANOVA, Mann-Whitney, clustered bootstrap CI, Chi-square | metric |
| 8 | Prediction and Forecast | predictive regression, panel fixed effects, ARIMA forecast | features, country |
| 9 | Analytical Conclusions | static and filter-aware notes | global filters |
| 10 | Data Export | CSV download for current view and full dataset | global filters |

All sections respond to global sidebar filters.

## Project Structure

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
│   ├── regression.py
│   ├── pca_analysis.py
│   ├── forecast.py
│   └── viz.py
├── data/
│   ├── regions.csv
│   ├── raw/
│   ├── merged.parquet
│   └── exclusions.csv
├── .streamlit/
│   └── config.toml
└── etl.log
```

## Method Notes

The project uses clustered bootstrap confidence intervals for regional means by resampling whole countries instead of individual country-year rows.

ANOVA is reported with Kruskal-Wallis and Levene checks because regional distributions can be skewed and heteroskedastic.

Pairwise regional tests use Mann-Whitney U with Holm-Bonferroni correction. Correlation tables also report Holm-adjusted p-values.

The predictive regression excludes `income_growth_pct` by default because it is part of the FPI target definition. Including it is available in the UI only to demonstrate target leakage.

`food_price_level_index` and `minimum_wage_eur_month` are optional regression features because their historical coverage is weaker.

Eurostat is migrating PPP data to COICOP 2018 in `prc_ppp_ind_1`; the ETL first tries the newer source and falls back to `prc_ppp_ind` for historical food price-level data.

The dashboard is exploratory. It describes observed relationships and does not establish causality.
