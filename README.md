# Europe Food Affordability

Interactive data-processing and visualisation project focused on food price affordability across Europe.

**Analytical question:** which European countries face the highest food price pressure, and how does it relate to headline inflation and household income?

The dashboard combines annual Eurostat data for European countries and uses the **Food Affordability Gap** as the main interpretation metric: food inflation minus median income growth. It also reports the **Food Pressure Index (FPI)** as a secondary synthetic ratio.

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

| Metric | Type | Description | Interpretation |
| --- | --- | --- | --- |
| `food_affordability_gap_pct` | percentage points | `food_inflation_pct - income_growth_pct` | Higher values indicate weaker food affordability; positive values mean food prices grew faster than income. |
| `fpi` | synthetic ratio | `food_inflation_pct / income_growth_pct`; values above 1 mean food prices grew faster than income; negative values can occur when income growth is negative | Higher values indicate stronger price pressure, but the ratio is unstable when income growth is close to zero. |
| `food_inflation_pct` | % YoY | annual HICP CP01 food inflation | Higher values indicate faster food price growth and greater consumer pressure. |
| `headline_inflation_pct` | % YoY | annual HICP CP00 all-items inflation | Higher values indicate faster growth of the general price level. |
| `median_income_eur` | EUR/year | median equivalised net income | Higher values indicate a larger nominal income buffer. |
| `income_growth_pct` | % YoY | country-level median income growth | Higher values indicate stronger income growth and greater capacity to absorb price increases. |
| `food_share_budget_pct` | % | food share in household final consumption | Higher values indicate greater household budget sensitivity to food prices. |
| `food_price_level_index` | EU27_2020=100 | relative food price level | Higher values indicate a higher food price level relative to the EU average. |
| `meal_deprivation_pct` | % of population | inability to afford a proper meal every second day | Higher values indicate more severe social deprivation. |
| `food_inflation_index_2020` | index | cumulative food price index with 2020=100 | Higher values indicate a higher accumulated food price level relative to 2020. |

When `income_growth_pct` is close to zero, ETL sets FPI to missing to avoid unstable division.

The dashboard also computes view-level helper metrics for the 2020-2024 comparison:

| Metric | Type | Description | Interpretation |
| --- | --- | --- | --- |
| `food_price_growth_2020_2024_pct` | % change | cumulative food price growth between 2020 and 2024 | Higher values indicate stronger accumulated food price growth. |
| `income_growth_2020_2024_pct` | % change | cumulative median income growth between 2020 and 2024 | Higher values indicate stronger accumulated income growth. |
| `cumulative_affordability_gap_pct` | percentage points | cumulative food price growth minus cumulative income growth | Higher values indicate weaker food affordability; positive values mean food prices outpaced income. |

## Dashboard Structure

| # | Section | Visualisation | Filters |
| --- | --- | --- | --- |
| 1 | KPI Bar | scorecards for largest affordability gap, average food inflation, food budget share, median gap | year |
| 2 | Country Diagnosis | driver metrics and driver bar for a selected country | country, year |
| 3 | Europe Map | Plotly choropleth with missing countries greyed out | year, metric |
| 4 | 2020-2024 Cumulative Pressure | bar ranking and table comparing food price growth with income growth | countries, regions |
| 5 | Country Typology | segment scatter and segment counts | year |
| 6 | Time Trends | multi-country line charts for food inflation and affordability gap | countries, years |
| 7 | Income vs Food Inflation | scatter with OLS trend | regions, years |
| 8 | Distributions and Anomalies | box plot, histogram, Z-score outlier table | metric |
| 9 | Correlations and PCA | Pearson/Spearman heatmap, Holm-adjusted p-values, country-year heatmap, PCA biplot, scree plot | method |
| 10 | Statistical Tests | ANOVA, Mann-Whitney, bootstrap CI, Chi-square | metric |
| 11 | Prediction and Forecast | affordability-gap regression, panel fixed effects, ARIMA forecast | features, country |
| 12 | Analytical Conclusions and Limitations | filter-aware notes and interpretation caveats | global filters |
| 13 | Data Export | CSV download for current view and full dataset | global filters |

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

The project uses bootstrap confidence intervals for regional means in the selected reference year. Each country appears once in that section, so the resampling unit is the country observation.

ANOVA is reported with Kruskal-Wallis, Levene checks, and Shapiro-Wilk diagnostics because regional distributions can be skewed, non-normal, and heteroskedastic.

Pairwise regional tests use Mann-Whitney U with Holm-Bonferroni correction. Correlation matrices use pairwise complete observations and correlation p-values are Holm-adjusted.

The Chi-square section reports Cramer's V and the minimum expected cell count. Results with expected counts below 5 are treated as descriptive diagnostics rather than strong confirmatory evidence.

Outlier tables rank country-year observations by the absolute Z-score of the selected metric. This highlights values farthest from the current filtered mean.

Country typology is rule-based and uses current-filter medians of the affordability gap, food spending share, income, and meal deprivation. It is intended as an interpretation aid, not a formal clustering model.

PCA is fitted on standardised variables, so features measured in EUR, percentages, and indices are comparable. The biplot shows country-year similarity in the first two components, while the scree plot shows how much variance each component explains.

The predictive regression and panel fixed-effects model use `food_affordability_gap_pct` as the default target because it is the main interpretation metric and is more stable than the FPI ratio. `income_growth_pct` is excluded from the default predictors because it is part of the affordability-gap formula and would create target leakage.

`food_price_level_index` and `minimum_wage_eur_month` are optional regression features because their historical coverage is weaker.

The ARIMA forecast is a short-run exploratory forecast for annual food inflation. The dashboard reports ADF p-value, AIC, BIC, observation count, and simple baselines. Seasonal models such as SARIMA are not used because the project uses annual series with about 15 observations per country.

Eurostat is migrating PPP data to COICOP 2018 in `prc_ppp_ind_1`; the ETL first tries the newer source and falls back to `prc_ppp_ind` for historical food price-level data.

The dashboard is exploratory. It describes observed relationships and does not establish causality.

Main interpretation limits:

- country-level aggregates do not show inequality within countries;
- some missing values are interpolated in ETL;
- FPI is sensitive when income growth is close to zero or negative, so the affordability gap is the safer headline metric;
- 2020-2024 cumulative comparisons require data in both endpoint years and do not replace full time-series interpretation.

## Methodological Fit and Exclusions

The current analytical table contains 480 country-year observations, covering 33 countries from 2010 to 2024. This is below the literal 1000-observation EDA threshold mentioned in the course notes, but it is a coherent macroeconomic panel: each row is a country-year observation built from official Eurostat series. The project therefore prioritises consistency, country coverage, and interpretable joins over artificially expanding the dataset.

PCA is retained because the dashboard uses several numeric indicators measured on different scales and the method is appropriate after standardisation. In the current cached dataset, PCA has 197 complete rows because `food_price_level_index` has weaker historical coverage.

The following methods are intentionally not used:

- t-SNE and UMAP: they could suggest unstable visual clusters with this sample size and small number of core numeric features.
- LDA: it requires supervised class labels and a classification objective, while this project is exploratory and comparative.
- Autoencoders: they are not appropriate for a small tabular macroeconomic dataset.
- SARIMA and seasonal decomposition: the data are annual, so there is no within-year seasonality to model.
- ACF/PACF as a dashboard section: with about 15 annual points per country, these diagnostics would be unstable and could be overinterpreted.

Main methodological caveats:

- interpolation smooths short missing-data gaps and can dampen abrupt changes;
- PPP food price-level data have weaker coverage than HICP, income, and expenditure-share data;
- country-level aggregates do not capture inequality within countries;
- regional statistical tests have small group sizes, so non-parametric results and effect sizes should be read alongside p-values;
- the dashboard is exploratory and comparative, not causal.
