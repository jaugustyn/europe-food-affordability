"""Statistical tests used by the Streamlit dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class RegionTestResult:
    anova_stat: float
    anova_p: float
    kruskal_stat: float
    kruskal_p: float
    levene_stat: float
    levene_p: float
    pairwise: pd.DataFrame


def _eta_squared(groups: list[np.ndarray]) -> float:
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))
    if ss_total == 0:
        return float("nan")
    ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
    return ss_between / ss_total


def _cliffs_delta_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return float("nan")
    return float((2 * u_stat) / (n_a * n_b) - 1)


def _cliffs_delta_label(delta: float) -> str:
    if np.isnan(delta):
        return "n/a"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def anova_regions(df: pd.DataFrame, metric: str, region_col: str = "region") -> dict:
    """One-way ANOVA, Kruskal-Wallis and Levene tests by region."""
    groups = [g[metric].dropna().values for _, g in df.groupby(region_col)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {
            "f": np.nan,
            "p": np.nan,
            "h": np.nan,
            "p_kruskal": np.nan,
            "levene_w": np.nan,
            "levene_p": np.nan,
            "eta_squared": np.nan,
            "n_groups": len(groups),
        }

    f, p = stats.f_oneway(*groups)
    h, p_kruskal = stats.kruskal(*groups)
    levene_w, levene_p = stats.levene(*groups, center="median")
    return {
        "f": float(f),
        "p": float(p),
        "h": float(h),
        "p_kruskal": float(p_kruskal),
        "levene_w": float(levene_w),
        "levene_p": float(levene_p),
        "eta_squared": _eta_squared(groups),
        "n_groups": len(groups),
    }


def pairwise_mann_whitney(
    df: pd.DataFrame,
    metric: str,
    region_col: str = "region",
    correction: str = "holm",
) -> pd.DataFrame:
    """Two-sided Mann-Whitney U for every pair of regions."""
    rows = []
    by_region = {r: g[metric].dropna().values for r, g in df.groupby(region_col)}
    for a, b in combinations(sorted(by_region), 2):
        x, y = by_region[a], by_region[b]
        if len(x) < 3 or len(y) < 3:
            continue
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "group_A": a,
            "group_B": b,
            "n_A": len(x),
            "n_B": len(y),
            "mean_A": float(np.mean(x)),
            "mean_B": float(np.mean(y)),
            "median_A": float(np.median(x)),
            "median_B": float(np.median(y)),
            "U": float(u),
            "p_value": float(p),
            "cliffs_delta": _cliffs_delta_from_u(float(u), len(x), len(y)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    _, p_adj, _, _ = multipletests(out["p_value"].values, alpha=0.05, method=correction)
    out["p_adj"] = p_adj
    out["effect_size"] = out["cliffs_delta"].map(_cliffs_delta_label)
    out["significant_05"] = out["p_adj"] < 0.05
    return out


def correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
) -> pd.DataFrame:
    sub = df[cols].dropna()
    if method == "spearman":
        return sub.corr(method="spearman")
    return sub.corr(method="pearson")


def correlation_significance(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    correction: str = "holm",
) -> pd.DataFrame:
    """Pairwise correlations with p-values and multiple-testing correction."""
    rows = []
    for a, b in combinations(cols, 2):
        sub = df[[a, b]].dropna()
        if len(sub) < 4 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        if method == "spearman":
            r, p = stats.spearmanr(sub[a], sub[b])
        else:
            r, p = stats.pearsonr(sub[a], sub[b])
        rows.append({
            "var_a": a,
            "var_b": b,
            "n": int(len(sub)),
            "r": float(r),
            "p_value": float(p),
            "r_squared": float(r ** 2),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    _, p_adj, _, _ = multipletests(out["p_value"].values, alpha=0.05, method=correction)
    out["p_adj"] = p_adj
    out["significant_05"] = out["p_adj"] < 0.05
    return out.sort_values("r", key=lambda s: s.abs(), ascending=False)


def chi_square_region_pressure(
    df: pd.DataFrame,
    metric: str = "fpi",
    region_col: str = "region",
    n_bins: int = 3,
) -> dict:
    """Test independence of region and a quantile-binned pressure metric."""
    bin_labels = ["low", "medium", "high"][:n_bins]
    sub = df[[region_col, metric]].dropna().copy()
    if len(sub) < n_bins * 2:
        return {"error": f"Too few observations ({len(sub)}) for Chi-square."}

    sub["pressure_class"] = pd.qcut(
        sub[metric],
        q=n_bins,
        labels=bin_labels,
        duplicates="drop",
    )
    contingency = pd.crosstab(sub[region_col], sub["pressure_class"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"error": "Contingency table is too small."}

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else float("nan")

    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "n": int(n),
        "cramers_v": cramers_v,
        "min_expected": float(expected.min()),
        "contingency": contingency,
        "expected": pd.DataFrame(
            expected,
            index=contingency.index,
            columns=contingency.columns,
        ),
    }


def region_anova(df: pd.DataFrame, metric: str, region_col: str = "region") -> RegionTestResult:
    """Return regional ANOVA/Kruskal/Levene results in the shape expected by the dashboard."""
    result = anova_regions(df, metric, region_col=region_col)
    if result["n_groups"] < 2:
        raise ValueError("At least two regions with two or more observations are required.")

    return RegionTestResult(
        anova_stat=result["f"],
        anova_p=result["p"],
        kruskal_stat=result["h"],
        kruskal_p=result["p_kruskal"],
        levene_stat=result["levene_w"],
        levene_p=result["levene_p"],
        pairwise=pairwise_mann_whitney(df, metric, region_col=region_col),
    )


def pvalue_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    correction: str = "holm",
) -> pd.DataFrame:
    """Symmetric matrix of Holm-adjusted pairwise correlation p-values."""
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for col in cols:
        out.loc[col, col] = 0.0

    pairs = []
    p_values = []
    for a, b in combinations(cols, 2):
        sub = df[[a, b]].dropna()
        if len(sub) < 4 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        if method == "spearman":
            _, p_value = stats.spearmanr(sub[a], sub[b])
        else:
            _, p_value = stats.pearsonr(sub[a], sub[b])
        pairs.append((a, b))
        p_values.append(float(p_value))

    if p_values:
        _, adjusted, _, _ = multipletests(p_values, alpha=0.05, method=correction)
        for (a, b), p_value in zip(pairs, adjusted):
            out.loc[a, b] = float(p_value)
            out.loc[b, a] = float(p_value)
    return out


def bootstrap_region_means(
    df: pd.DataFrame,
    metric: str,
    region_col: str = "region",
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap regional means and 95% confidence intervals for a metric."""
    rng = np.random.default_rng(seed)
    rows = []

    for region, group in df.groupby(region_col):
        values = group[metric].dropna().to_numpy()
        if len(values) == 0:
            continue
        means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
        ci_low, ci_high = np.percentile(means, [2.5, 97.5])
        rows.append(
            {
                region_col: region,
                "mean": float(values.mean()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n": int(len(values)),
            }
        )

    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def top_outliers(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    """Return observations with the largest absolute z-scores for the selected metric."""
    sub = df.dropna(subset=[metric]).copy()
    if sub.empty:
        sub["zscore"] = pd.Series(dtype=float)
        return sub

    std = sub[metric].std(ddof=0)
    if std == 0 or np.isnan(std):
        sub["zscore"] = 0.0
    else:
        sub["zscore"] = (sub[metric] - sub[metric].mean()) / std
    return sub.sort_values("zscore", key=lambda s: s.abs(), ascending=False).head(n)


def chi_square_high_pressure(df: pd.DataFrame, value_col: str = "fpi") -> dict | None:
    """Adapt the regional pressure chi-square result for the dashboard display."""
    result = chi_square_region_pressure(df, metric=value_col)
    if "error" in result:
        return None
    return {
        "chi2": result["chi2"],
        "p_value": result["p"],
        "dof": result["dof"],
        "cramers_v": result["cramers_v"],
        "table": result["contingency"],
    }
