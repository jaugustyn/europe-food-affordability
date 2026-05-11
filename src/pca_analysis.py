"""Principal Component Analysis on food-affordability features."""
from __future__ import annotations

import pandas as pd
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
) -> dict:
    """Fit PCA on standardised features and return embedding and diagnostics."""
    feats = features or DEFAULT_PCA_FEATURES
    sub = df[feats + ["country_name", "year", "region"]].dropna().copy()
    if len(sub) < n_components + 5:
        return {"error": f"Too few observations ({len(sub)}) for PCA."}

    X = StandardScaler().fit_transform(sub[feats].values)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)

    embedding = sub[["country_name", "year", "region"]].copy()
    for i in range(n_components):
        embedding[f"PC{i + 1}"] = coords[:, i]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feats,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    return {
        "embedding": embedding,
        "loadings": loadings,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n": len(sub),
        "features": feats,
    }
