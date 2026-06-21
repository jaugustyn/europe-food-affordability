"""Principal Component Analysis on food-affordability features."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_PCA_FEATURES = [
    "food_inflation_pct",
    "headline_inflation_pct",
    "food_share_budget_pct",
    "median_income_eur",
    "food_price_level_index",
    "meal_deprivation_pct",
]


def fit_pca(
    df: pd.DataFrame,
    features: list[str] | None = None,
    n_components: int = 2,
    variance_threshold: float = 0.80,
) -> dict:
    """Fit PCA on standardised features and return embedding and diagnostics."""
    feats = features or DEFAULT_PCA_FEATURES
    imputation_flags = [f"{feature}_imputed" for feature in feats]
    available_flags = [flag for flag in imputation_flags if flag in df.columns]
    if available_flags:
        imputed_rows = df[available_flags].fillna(False).astype(bool).any(axis=1)
    else:
        imputed_rows = pd.Series(False, index=df.index, dtype=bool)
    eligible = df.loc[~imputed_rows]
    sub = eligible[feats + ["country_name", "year", "region"]].dropna().copy()
    if len(sub) < n_components + 5:
        raise ValueError(f"Za mało kompletnych obserwacji ({len(sub)}) do PCA.")

    X = StandardScaler().fit_transform(sub[feats].values)
    max_components = min(len(feats), len(sub))
    pca = PCA(n_components=max_components, random_state=42)
    coords = pca.fit_transform(X)
    if pca.components_ is None or pca.explained_variance_ is None:
        raise RuntimeError("PCA did not expose fitted components and variances.")

    embedding = sub[["country_name", "year", "region"]].copy()
    for i in range(n_components):
        embedding[f"PC{i + 1}"] = coords[:, i]

    # Correlation loadings, not only raw eigenvectors.
    loading_values = pca.components_[:n_components].T * np.sqrt(
        pca.explained_variance_[:n_components]
    )
    loadings = pd.DataFrame(
        loading_values,
        index=feats,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    ).reset_index(names="feature")

    explained_full = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(explained_full)
    selected_k = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    selected_k = min(selected_k, len(explained_full))
    explained = explained_full[:n_components]

    return {
        "embedding": embedding,
        "scores": embedding,
        "loadings": loadings,
        "explained_variance_ratio": explained,
        "explained_variance": explained,
        "explained_variance_full": explained_full,
        "cumulative_variance_full": cumulative.tolist(),
        "selected_k": selected_k,
        "variance_threshold": float(variance_threshold),
        "n": len(sub),
        "imputed_rows_excluded": int(imputed_rows.sum()),
        "features": feats,
    }
