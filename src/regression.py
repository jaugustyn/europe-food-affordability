from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "median_income_eur",
    "food_share_budget_pct",
    "headline_inflation_pct",
    "meal_deprivation_pct",
]


@dataclass
class ModelSummary:
    model_name: str
    r2_test: float
    mae_test: float
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame


def _training_frame(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    needed = ["fpi", "country_name", "country_code", "year", *features]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modelling: {', '.join(missing)}")

    frame = df[needed].dropna(subset=["fpi"]).copy()
    if len(frame) < 20:
        raise ValueError("Not enough non-null FPI observations for regression modelling.")
    return frame


def fit_fpi_regression(
    df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    model_type: str = "ridge",
    random_state: int = 42,
) -> dict[str, object]:
    frame = _training_frame(df, features)

    x = frame[list(features)]
    y = frame["fpi"]
    idx_train, idx_test = train_test_split(frame.index, test_size=0.25, random_state=random_state)

    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        model_name = "Random Forest"
    else:
        estimator = Ridge(alpha=1.0)
        model_name = "Ridge Regression"

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x.loc[idx_train], y.loc[idx_train])

    y_pred = pipeline.predict(x.loc[idx_test])
    r2 = r2_score(y.loc[idx_test], y_pred)
    mae = mean_absolute_error(y.loc[idx_test], y_pred)

    if model_type == "random_forest":
        importances = pipeline.named_steps["model"].feature_importances_
    else:
        importances = np.abs(pipeline.named_steps["model"].coef_)

    importance_df = (
        pd.DataFrame({"feature": list(features), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    predictions = frame.loc[idx_test, ["country_name", "country_code", "year", "fpi"]].copy()
    predictions["predicted_fpi"] = y_pred
    predictions["residual"] = predictions["fpi"] - predictions["predicted_fpi"]

    return {
        "model_name": model_name,
        "r2_test": float(r2),
        "mae_test": float(mae),
        "feature_importance": importance_df,
        "predictions": predictions.sort_values("residual", key=lambda s: s.abs(), ascending=False),
        "features": list(features),
    }


def fit_panel_fixed_effects(df: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES) -> dict[str, object]:
    frame = _training_frame(df, features).dropna(subset=list(features)).copy()
    if frame["country_code"].nunique() < 3 or frame["year"].nunique() < 3:
        raise ValueError("Panel fixed-effects model requires at least 3 countries and 3 years.")

    formula = "fpi ~ " + " + ".join(features) + " + C(country_code) + C(year)"
    result = smf.ols(formula, data=frame).fit(cov_type="HC3")

    coef = (
        pd.DataFrame(
            {
                "term": result.params.index,
                "coef": result.params.values,
                "p_value": result.pvalues.values,
                "std_err": result.bse.values,
            }
        )
        .query("term in @features")
        .sort_values("p_value")
        .reset_index(drop=True)
    )

    return {
        "formula": formula,
        "r2": float(result.rsquared),
        "adj_r2": float(result.rsquared_adj),
        "n": int(result.nobs),
        "coefficients": coef,
        "summary_text": result.summary().as_text(),
    }
