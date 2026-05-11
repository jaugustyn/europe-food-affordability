"""Statistical tests used by the Streamlit dashboard."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


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
