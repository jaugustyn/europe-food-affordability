"""Statistical tests used by the Streamlit dashboard."""
from __future__ import annotations

from itertools import combinations
from typing import Literal, TypedDict, cast
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class RegionTestResult(TypedDict):
    """Typed contract returned by the regional testing workflow."""

    method: Literal["anova", "kruskal"]
    method_label: str
    hypothesis_null: str
    hypothesis_alternative: str
    group_statistic_name: str
    group_summary: pd.Series
    interpretation: str
    statistic: float
    p_value: float
    alpha: float
    reject_h0: bool
    effect_name: str
    effect_size: float
    effect_label: str
    levene_stat: float
    levene_p: float
    all_normal: bool
    equal_variances: bool
    diagnostics: pd.DataFrame
    posthoc: pd.DataFrame


def mask_imputed_values(
    df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask imputed metric cells so inferential methods use observed values only."""
    out = df.copy()
    excluded: dict[str, int] = {}
    for metric in metrics:
        flag = f"{metric}_imputed"
        if flag not in out.columns:
            excluded[metric] = 0
            continue
        mask = out[flag].fillna(False).astype(bool)
        excluded[metric] = int(mask.sum())
        out.loc[mask, metric] = np.nan
    return out, excluded


def _finite_series(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    return values[np.isfinite(values)].dropna()


def _finite_array(s: pd.Series) -> np.ndarray:
    """Return finite values as a numeric NumPy array with a stable static type."""
    return _finite_series(s).to_numpy(dtype=float)


def _omega_squared(groups: list[np.ndarray]) -> float:
    all_vals = np.concatenate(groups)
    k = len(groups)
    n = len(all_vals)
    if k < 2 or n <= k:
        return float("nan")
    grand_mean = all_vals.mean()
    ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    ss_total = ss_between + ss_within
    ms_within = ss_within / (n - k)
    denominator = ss_total + ms_within
    if denominator == 0:
        return float("nan")
    return float(max(0.0, (ss_between - (k - 1) * ms_within) / denominator))


def _epsilon_squared(h_stat: float, n: int, k: int) -> float:
    if n <= k:
        return float("nan")
    return float(max(0.0, (h_stat - k + 1) / (n - k)))


def _effect_label(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 0.01:
        return "negligible"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "medium"
    return "large"


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


def pairwise_mann_whitney(
    df: pd.DataFrame,
    metric: str,
    region_col: str = "region",
    correction: str = "holm",
) -> pd.DataFrame:
    """Two-sided Mann-Whitney U for every pair of regions."""
    rows = []
    by_region = {
        str(region): _finite_array(group[metric])
        for region, group in df.groupby(region_col)
    }
    for a, b in combinations(sorted(by_region), 2):
        x, y = by_region[a], by_region[b]
        if len(x) < 3 or len(y) < 3:
            continue
        test_result = stats.mannwhitneyu(x, y, alternative="two-sided")
        u, p = float(test_result.statistic), float(test_result.pvalue)
        rows.append({
            "group_A": a,
            "group_B": b,
            "n_A": len(x),
            "n_B": len(y),
            "mean_A": float(np.mean(x)),
            "mean_B": float(np.mean(y)),
            "median_A": float(np.median(x)),
            "median_B": float(np.median(y)),
            "U": u,
            "p_value": p,
            "cliffs_delta": _cliffs_delta_from_u(u, len(x), len(y)),
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
    """Pairwise correlation matrix using the same pairwise sample logic as p-values."""
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for col in cols:
        values = _finite_series(df[col]) if col in df.columns else pd.Series(dtype=float)
        if len(values) > 0:
            out.loc[col, col] = 1.0

    for a, b in combinations(cols, 2):
        sub = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 2 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        if method == "spearman":
            test_result = stats.spearmanr(sub[a], sub[b])
        else:
            test_result = stats.pearsonr(sub[a], sub[b])
        r_raw, _ = cast(tuple[float, float], test_result)
        r = float(r_raw)
        out.loc[a, b] = float(r)
        out.loc[b, a] = float(r)
    return out


def correlation_significance(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    correction: str = "holm",
) -> pd.DataFrame:
    """Pairwise correlations with p-values and multiple-testing correction."""
    rows = []
    for a, b in combinations(cols, 2):
        sub = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 4 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        if method == "spearman":
            test_result = stats.spearmanr(sub[a], sub[b])
        else:
            test_result = stats.pearsonr(sub[a], sub[b])
        r_raw, p_raw = cast(tuple[float, float], test_result)
        r, p = float(r_raw), float(p_raw)
        rows.append({
            "var_a": a,
            "var_b": b,
            "n": int(len(sub)),
            "r": r,
            "p_value": p,
            "r_squared": r**2,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    _, p_adj, _, _ = multipletests(out["p_value"].values, alpha=0.05, method=correction)
    out["p_adj"] = p_adj
    out["significant_05"] = out["p_adj"] < 0.05
    return out.sort_values("r", key=lambda s: s.abs(), ascending=False)





def shapiro_by_region(df: pd.DataFrame, metric: str, region_col: str = "region") -> pd.DataFrame:
    """Shapiro-Wilk normality diagnostics by region."""
    rows = []
    for region, group in df.groupby(region_col):
        values = _finite_array(group[metric])
        if len(values) < 3:
            rows.append({
                region_col: region,
                "n": int(len(values)),
                "w": np.nan,
                "p_value": np.nan,
                "normal_05": np.nan,
                "note": "Too few observations",
            })
            continue
        sample = values
        note = ""
        if len(values) > 5000:
            sample = values[:5000]
            note = "First 5000 observations used"
        test_result = stats.shapiro(sample)
        w, p = float(test_result.statistic), float(test_result.pvalue)
        rows.append({
            region_col: region,
            "n": int(len(values)),
            "w": w,
            "p_value": p,
            "normal_05": bool(p >= 0.05),
            "note": note,
        })
    return pd.DataFrame(rows).sort_values(region_col).reset_index(drop=True)


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

    pairs: list[tuple[str, str]] = []
    p_values: list[float] = []
    for a, b in combinations(cols, 2):
        sub = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 4 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        if method == "spearman":
            test_result = stats.spearmanr(sub[a], sub[b])
        else:
            test_result = stats.pearsonr(sub[a], sub[b])
        _, p_raw = cast(tuple[float, float], test_result)
        p_value = float(p_raw)
        pairs.append((a, b))
        p_values.append(float(p_value))

    if p_values:
        _, adjusted, _, _ = multipletests(p_values, alpha=0.05, method=correction)
        if adjusted is None:
            raise RuntimeError("Holm correction did not return adjusted p-values.")
        for (a, b), p_value in zip(pairs, adjusted):
            out.loc[a, b] = float(p_value)
            out.loc[b, a] = float(p_value)
    return out


def sample_size_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pairwise-complete sample sizes corresponding to a correlation matrix."""
    out = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for a in cols:
        out.loc[a, a] = int(_finite_series(df[a]).shape[0])
    for a, b in combinations(cols, 2):
        n = int(df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
        out.loc[a, b] = n
        out.loc[b, a] = n
    return out


def bootstrap_region_means(
    df: pd.DataFrame,
    metric: str,
    region_col: str = "region",
    n_boot: int = 2000,
    seed: int = 42,
    min_group_n: int = 3,
) -> pd.DataFrame:
    """Bootstrap regional means when each reported group has enough observations."""
    rng = np.random.default_rng(seed)
    rows = []

    for region, group in df.groupby(region_col):
        values = group[metric].dropna().to_numpy()
        if len(values) < min_group_n:
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

    out = pd.DataFrame(rows, columns=[region_col, "mean", "ci_low", "ci_high", "n"])
    if out.empty:
        return out
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def iqr_outliers(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    """Return observations outside Tukey's 1.5×IQR fences."""
    sub = df.dropna(subset=[metric]).copy()
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return sub.assign(iqr_distance=pd.Series(dtype=float))
    q1, q3 = sub[metric].quantile([0.25, 0.75])
    spread = q3 - q1
    if not np.isfinite(spread) or spread == 0:
        return sub.iloc[0:0].assign(iqr_distance=pd.Series(dtype=float))
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    flagged = sub[(sub[metric] < lower) | (sub[metric] > upper)].copy()
    flagged["iqr_distance"] = np.where(
        flagged[metric] < lower,
        (lower - flagged[metric]) / spread,
        (flagged[metric] - upper) / spread,
    )
    flagged["iqr_lower"] = float(lower)
    flagged["iqr_upper"] = float(upper)
    return flagged.nlargest(n, "iqr_distance")


def _tukey_posthoc(df: pd.DataFrame, metric: str, region_col: str) -> pd.DataFrame:
    sub = df[[region_col, metric]].replace([np.inf, -np.inf], np.nan).dropna()
    result = pairwise_tukeyhsd(sub[metric].astype(float), sub[region_col].astype(str))
    if result.confint is None:
        raise RuntimeError("Tukey HSD did not return confidence intervals.")
    pairs = list(combinations(result.groupsunique, 2))
    return pd.DataFrame(
        {
            "group1": [pair[0] for pair in pairs],
            "group2": [pair[1] for pair in pairs],
            "meandiff": result.meandiffs,
            "p-adj": result.pvalues,
            "lower": result.confint[:, 0],
            "upper": result.confint[:, 1],
            "reject": result.reject,
        }
    )


def select_region_test(
    df: pd.DataFrame,
    metric: str,
    region_col: str = "region",
    alpha: float = 0.05,
) -> RegionTestResult:
    """Select one omnibus regional test, effect size and conditional post-hoc."""
    grouped = {
        str(region): _finite_array(group[metric])
        for region, group in df.groupby(region_col)
    }
    grouped = {region: values for region, values in grouped.items() if len(values) >= 2}
    if len(grouped) < 2:
        raise ValueError(
            "Brak wystarczających danych obserwowanych: wymagane są co najmniej "
            "dwa regiony z dwiema obserwacjami."
        )

    diagnostics = shapiro_by_region(df, metric, region_col=region_col)
    shapiro_values = diagnostics["p_value"].dropna()
    all_normal = bool(
        len(shapiro_values) == len(grouped) and (shapiro_values >= alpha).all()
    )
    groups = list(grouped.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        levene_stat, levene_p = stats.levene(*groups, center="median")
    equal_variances = bool(np.isfinite(levene_p) and levene_p >= alpha)

    if all_normal and equal_variances:
        statistic, p_value = stats.f_oneway(*groups)
        effect = _omega_squared(groups)
        method = "anova"
        effect_name = "omega_squared"
        method_label = "ANOVA jednoczynnikowa"
        hypothesis_null = "średnie regionalne są równe"
        hypothesis_alternative = "co najmniej jedna średnia regionalna jest inna"
        group_statistic_name = "średnia"
        group_summary = (
            df.groupby(region_col)[metric].mean().dropna().sort_values(ascending=False)
        )
        interpretation = "ANOVA ocenia różnice między średnimi regionalnymi."
    else:
        statistic, p_value = stats.kruskal(*groups)
        effect = _epsilon_squared(float(statistic), sum(map(len, groups)), len(groups))
        method = "kruskal"
        effect_name = "epsilon_squared"
        method_label = "Kruskal–Wallis"
        hypothesis_null = "rozkłady i rangi są jednakowe we wszystkich regionach"
        hypothesis_alternative = "co najmniej jeden region ma inny rozkład rang"
        group_statistic_name = "mediana"
        group_summary = (
            df.groupby(region_col)[metric].median().dropna().sort_values(ascending=False)
        )
        interpretation = (
            "Kruskal–Wallis ocenia różnice rozkładów rang; nie jest automatycznie "
            "testem równości median."
        )

    reject = bool(p_value < alpha)
    if not reject:
        posthoc = pd.DataFrame()
    elif method == "anova":
        posthoc = _tukey_posthoc(df, metric, region_col)
    else:
        posthoc = pairwise_mann_whitney(
            df, metric, region_col=region_col, correction="holm"
        )

    return {
        "method": method,
        "method_label": method_label,
        "hypothesis_null": hypothesis_null,
        "hypothesis_alternative": hypothesis_alternative,
        "group_statistic_name": group_statistic_name,
        "group_summary": group_summary,
        "interpretation": interpretation,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "reject_h0": reject,
        "effect_name": effect_name,
        "effect_size": float(effect),
        "effect_label": _effect_label(float(effect)),
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "all_normal": all_normal,
        "equal_variances": equal_variances,
        "diagnostics": diagnostics,
        "posthoc": posthoc,
    }
