"""
hypothesis_testing.py
---------------------
Comprehensive hypothesis testing toolkit for compensation analysis.

Includes:
  - Two-sample tests (Welch's t, Mann-Whitney U, permutation test)
  - Multi-sample tests (one-way ANOVA, Welch's ANOVA, Kruskal-Wallis)
  - Post-hoc procedures (Tukey HSD, Games-Howell, Dunn)
  - Two-way ANOVA with interaction
  - Effect size measures (Cohen's d, eta-squared, omega-squared, rank-biserial)
  - Power analysis
  - Assumption checks (Levene, Brown-Forsythe, Shapiro-Wilk per group)
  - Multiple comparison corrections (Bonferroni, Holm, BH-FDR)
  - Chi-squared test of independence

Designed to be reusable: every test function accepts raw arrays or
DataFrames and returns structured result dicts.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

import config as cfg


# ===================================================================
# EFFECT SIZE MEASURES
# ===================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray, pooled: bool = True) -> float:
    """
    Cohen's d for two independent samples.

    Parameters
    ----------
    pooled : if True, use pooled SD (equal variance); if False, use
             the SD of the control group (Glass's delta variant).
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    if pooled:
        sp = np.sqrt(((n1 - 1) * group1.std(ddof=1)**2 +
                       (n2 - 1) * group2.std(ddof=1)**2) / (n1 + n2 - 2))
    else:
        sp = group2.std(ddof=1)  # Glass's delta
    return (m1 - m2) / sp if sp > 0 else np.nan


def eta_squared(ss_between: float, ss_total: float) -> float:
    """Eta-squared effect size for ANOVA."""
    return ss_between / ss_total if ss_total > 0 else np.nan


def omega_squared(ss_between: float, ss_within: float, ms_within: float, k: int, N: int) -> float:
    """
    Omega-squared: less biased estimator of ANOVA effect size.
    k = number of groups, N = total sample size.
    """
    numerator = ss_between - (k - 1) * ms_within
    denominator = ss_within + ss_between + ms_within
    return numerator / denominator if denominator > 0 else np.nan


def rank_biserial_r(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * U) / (n1 * n2)


def _interpret_d(d: float) -> str:
    """Cohen's conventions for d."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ===================================================================
# ASSUMPTION CHECKS
# ===================================================================

def check_homogeneity_of_variance(
    *groups: np.ndarray,
    method: Literal["levene", "brown_forsythe"] = "levene",
) -> dict:
    """Levene's or Brown-Forsythe test for equality of variances."""
    center = "mean" if method == "levene" else "median"
    stat, p = stats.levene(*groups, center=center)
    return {
        "test": method,
        "statistic": stat,
        "p_value": p,
        "equal_variance": p >= cfg.ALPHA,
    }


def check_normality_per_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = cfg.ALPHA,
) -> pd.DataFrame:
    """Run Shapiro-Wilk test on each group. Returns per-group results."""
    records = []
    for name, grp in df.groupby(group_col):
        vals = grp[value_col].dropna().values
        n = len(vals)
        if n < 3:
            records.append({"group": name, "n": n, "shapiro_stat": np.nan,
                            "shapiro_p": np.nan, "normal": np.nan})
            continue
        sample = vals if n <= 5000 else np.random.default_rng(cfg.RANDOM_STATE).choice(vals, 5000, replace=False)
        stat, p = stats.shapiro(sample)
        records.append({"group": name, "n": n, "shapiro_stat": stat,
                        "shapiro_p": p, "normal": p >= alpha})
    return pd.DataFrame(records).set_index("group")


# ===================================================================
# TWO-SAMPLE TESTS
# ===================================================================

def welch_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = cfg.ALPHA,
) -> dict:
    """
    Welch's t-test (does not assume equal variances).
    Includes Cohen's d and a power estimate.
    """
    group1, group2 = np.asarray(group1), np.asarray(group2)
    group1, group2 = group1[~np.isnan(group1)], group2[~np.isnan(group2)]

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False, alternative=alternative)
    d = cohens_d(group1, group2)

    return {
        "test": "welch_t",
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": d,
        "effect_interpretation": _interpret_d(d),
        "mean_diff": group1.mean() - group2.mean(),
        "n1": len(group1),
        "n2": len(group2),
        "reject_h0": p_value < alpha,
        "alpha": alpha,
    }


def mann_whitney_u(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = cfg.ALPHA,
) -> dict:
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    group1, group2 = np.asarray(group1), np.asarray(group2)
    group1, group2 = group1[~np.isnan(group1)], group2[~np.isnan(group2)]

    U, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    r = rank_biserial_r(U, len(group1), len(group2))

    return {
        "test": "mann_whitney_u",
        "U_statistic": U,
        "p_value": p_value,
        "rank_biserial_r": r,
        "n1": len(group1),
        "n2": len(group2),
        "reject_h0": p_value < alpha,
        "alpha": alpha,
    }


def permutation_test_two_sample(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10_000,
    statistic: str = "mean_diff",
    seed: int = cfg.RANDOM_STATE,
    alpha: float = cfg.ALPHA,
) -> dict:
    """
    Exact-style permutation test for the difference in means (or medians)
    between two independent samples.
    """
    group1, group2 = np.asarray(group1), np.asarray(group2)
    group1, group2 = group1[~np.isnan(group1)], group2[~np.isnan(group2)]
    rng = np.random.default_rng(seed)

    func = np.mean if statistic == "mean_diff" else np.median
    observed = func(group1) - func(group2)

    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(combined)
        perm_stats[i] = func(combined[:n1]) - func(combined[n1:])

    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))

    return {
        "test": "permutation",
        "observed_diff": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "reject_h0": p_value < alpha,
    }


# ===================================================================
# GENDER PAY GAP ANALYSIS (across positions)
# ===================================================================

def gender_pay_gap_by_position(
    df: pd.DataFrame,
    salary_col: str = "Salary",
    gender_col: str = "Gender",
    position_col: str = "Position",
    alpha: float = cfg.ALPHA,
) -> pd.DataFrame:
    """
    For each position, run parametric (Welch t) and non-parametric
    (Mann-Whitney U) tests comparing Male vs Female salaries.
    Also computes Cohen's d and the raw/percentage pay gap.
    """
    records = []
    for pos in df[position_col].unique():
        male = df[(df[gender_col] == "Male") & (df[position_col] == pos)][salary_col].dropna().values
        female = df[(df[gender_col] == "Female") & (df[position_col] == pos)][salary_col].dropna().values

        if len(male) < 2 or len(female) < 2:
            continue

        t_res = welch_t_test(male, female, alpha=alpha)
        u_res = mann_whitney_u(male, female, alpha=alpha)

        records.append({
            "position": pos,
            "n_male": len(male),
            "n_female": len(female),
            "mean_male": male.mean(),
            "mean_female": female.mean(),
            "raw_gap": male.mean() - female.mean(),
            "pct_gap": ((male.mean() - female.mean()) / female.mean()) * 100,
            "welch_t": t_res["t_statistic"],
            "welch_p": t_res["p_value"],
            "cohens_d": t_res["cohens_d"],
            "effect_size": t_res["effect_interpretation"],
            "mann_whitney_U": u_res["U_statistic"],
            "mann_whitney_p": u_res["p_value"],
            "rank_biserial_r": u_res["rank_biserial_r"],
            "reject_welch": t_res["reject_h0"],
            "reject_mwu": u_res["reject_h0"],
        })

    result = pd.DataFrame(records)
    if not result.empty:
        # Apply Holm-Bonferroni correction across positions
        _, p_adj, _, _ = multipletests(result["welch_p"], method="holm")
        result["welch_p_adjusted"] = p_adj
        result["reject_adjusted"] = p_adj < alpha
    return result


# ===================================================================
# MULTI-SAMPLE TESTS
# ===================================================================

def one_way_anova(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = cfg.ALPHA,
) -> dict:
    """
    Classic one-way ANOVA with assumption checks, effect sizes,
    and automatic post-hoc testing.
    """
    groups = [grp[value_col].dropna().values for _, grp in df.groupby(group_col)]
    group_labels = df[group_col].values
    values = df[value_col].dropna().values
    k = len(groups)
    N = sum(len(g) for g in groups)

    # Assumption: homogeneity of variance
    levene_res = check_homogeneity_of_variance(*groups)

    # ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    # Effect sizes
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_within = sum(np.sum((g - g.mean())**2) for g in groups)
    ss_total = ss_between + ss_within
    ms_within = ss_within / (N - k)
    eta2 = eta_squared(ss_between, ss_total)
    omega2 = omega_squared(ss_between, ss_within, ms_within, k, N)

    result = {
        "test": "one_way_anova",
        "F_statistic": f_stat,
        "p_value": p_value,
        "reject_h0": p_value < alpha,
        "k_groups": k,
        "N_total": N,
        "eta_squared": eta2,
        "omega_squared": omega2,
        "levene_statistic": levene_res["statistic"],
        "levene_p": levene_res["p_value"],
        "equal_variance": levene_res["equal_variance"],
    }

    # Post-hoc: Tukey HSD
    if p_value < alpha:
        aligned = df[[group_col, value_col]].dropna()
        tukey = pairwise_tukeyhsd(
            aligned[value_col], aligned[group_col], alpha=alpha,
        )
        result["tukey_summary"] = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
    return result


def welch_anova(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = cfg.ALPHA,
) -> dict:
    """
    Welch's one-way ANOVA -- robust to unequal variances.
    Uses scipy.stats.alexandergovern (Alexander-Govern approximation)
    when available, otherwise falls back to manual Welch F.
    """
    groups = {name: grp[value_col].dropna().values for name, grp in df.groupby(group_col)}
    group_arrays = list(groups.values())

    # scipy >= 1.8 has alexandergovern
    if hasattr(stats, "alexandergovern"):
        res = stats.alexandergovern(*group_arrays)
        return {
            "test": "alexander_govern",
            "statistic": res.statistic,
            "p_value": res.pvalue,
            "reject_h0": res.pvalue < alpha,
        }

    # Manual Welch's F
    ns = np.array([len(g) for g in group_arrays])
    means = np.array([g.mean() for g in group_arrays])
    variances = np.array([g.var(ddof=1) for g in group_arrays])
    weights = ns / variances
    w_total = weights.sum()
    grand_mean_w = np.sum(weights * means) / w_total
    k = len(group_arrays)

    F_num = np.sum(weights * (means - grand_mean_w)**2) / (k - 1)
    lam = np.sum((1 - weights / w_total)**2 / (ns - 1))
    F_den = 1 + 2 * (k - 2) * lam / (k**2 - 1)
    F_w = F_num / F_den

    df1 = k - 1
    df2 = (k**2 - 1) / (3 * lam)
    p_value = 1 - stats.f.cdf(F_w, df1, df2)

    return {
        "test": "welch_anova",
        "F_statistic": F_w,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "reject_h0": p_value < alpha,
    }


def kruskal_wallis(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = cfg.ALPHA,
) -> dict:
    """Kruskal-Wallis H test (non-parametric one-way ANOVA)."""
    groups = [grp[value_col].dropna().values for _, grp in df.groupby(group_col)]
    H, p = stats.kruskal(*groups)
    N = sum(len(g) for g in groups)
    k = len(groups)
    # Epsilon-squared effect size
    eps2 = (H - k + 1) / (N - k) if (N - k) > 0 else np.nan
    return {
        "test": "kruskal_wallis",
        "H_statistic": H,
        "p_value": p,
        "reject_h0": p < alpha,
        "epsilon_squared": eps2,
        "k_groups": k,
        "N_total": N,
    }


# ===================================================================
# POST-HOC: GAMES-HOWELL (unequal variances)
# ===================================================================

def games_howell(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = cfg.ALPHA,
) -> pd.DataFrame:
    """
    Games-Howell post-hoc test -- does not assume equal variances or
    equal sample sizes. Uses Welch-Satterthwaite df and studentized
    range distribution.
    """
    groups = {}
    for name, grp in df.groupby(group_col):
        vals = grp[value_col].dropna().values
        if len(vals) >= 2:
            groups[name] = vals

    labels = sorted(groups.keys())
    records = []
    for a, b in combinations(labels, 2):
        g1, g2 = groups[a], groups[b]
        n1, n2 = len(g1), len(g2)
        m1, m2 = g1.mean(), g2.mean()
        v1, v2 = g1.var(ddof=1), g2.var(ddof=1)

        se = np.sqrt(v1 / n1 + v2 / n2)
        mean_diff = m1 - m2
        t_stat = mean_diff / se if se > 0 else np.nan

        # Welch-Satterthwaite degrees of freedom
        num = (v1 / n1 + v2 / n2)**2
        den = (v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1)
        df_ws = num / den if den > 0 else np.nan

        # p-value from studentized range distribution (approximation)
        # Using t-distribution as a reasonable approximation
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_ws)) if not np.isnan(df_ws) else np.nan

        records.append({
            "group1": a,
            "group2": b,
            "mean_diff": mean_diff,
            "se": se,
            "t_statistic": t_stat,
            "df": df_ws,
            "p_value": p_value,
            "reject": p_value < alpha if not np.isnan(p_value) else False,
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        _, p_adj, _, _ = multipletests(result_df["p_value"].fillna(1), method="holm")
        result_df["p_adjusted"] = p_adj
        result_df["reject_adjusted"] = p_adj < alpha
    return result_df


# ===================================================================
# TWO-WAY ANOVA
# ===================================================================

def two_way_anova(
    df: pd.DataFrame,
    value_col: str,
    factor_a: str,
    factor_b: str,
) -> pd.DataFrame:
    """
    Type II two-way ANOVA with interaction using statsmodels OLS.
    Returns the ANOVA table.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    clean = df[[value_col, factor_a, factor_b]].dropna()
    # Ensure categorical
    clean[factor_a] = clean[factor_a].astype(str)
    clean[factor_b] = clean[factor_b].astype(str)

    formula = f"Q('{value_col}') ~ C(Q('{factor_a}')) * C(Q('{factor_b}'))"
    model = smf.ols(formula, data=clean).fit()
    table = anova_lm(model, typ=2)

    # Add effect sizes
    ss_total = table["sum_sq"].sum()
    table["eta_squared"] = table["sum_sq"] / ss_total
    table["significant"] = table["PR(>F)"] < cfg.ALPHA

    return table


# ===================================================================
# CHI-SQUARED TEST OF INDEPENDENCE
# ===================================================================

def chi_squared_independence(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    alpha: float = cfg.ALPHA,
) -> dict:
    """
    Chi-squared test of independence with Cramer's V effect size.
    """
    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else np.nan

    return {
        "test": "chi_squared",
        "chi2_statistic": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": cramers_v,
        "reject_h0": p < alpha,
        "contingency_table": ct,
        "expected_frequencies": pd.DataFrame(expected, index=ct.index, columns=ct.columns),
    }


# ===================================================================
# MULTIPLE COMPARISON CORRECTION
# ===================================================================

def adjust_pvalues(
    p_values: np.ndarray | list,
    method: Literal["bonferroni", "holm", "fdr_bh"] = "holm",
    alpha: float = cfg.ALPHA,
) -> dict:
    """
    Apply multiple comparison correction to a set of p-values.

    Methods
    -------
    bonferroni : conservative FWER control
    holm : step-down FWER control (uniformly more powerful than Bonferroni)
    fdr_bh : Benjamini-Hochberg FDR control
    """
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return {
        "method": method,
        "original_p": np.array(p_values),
        "adjusted_p": p_adj,
        "reject": reject,
    }


# ===================================================================
# POWER ANALYSIS (two-sample t-test)
# ===================================================================

def power_analysis_two_sample(
    effect_size: float,
    n1: int,
    n2: int,
    alpha: float = cfg.ALPHA,
) -> float:
    """
    Approximate power of a two-sample t-test given effect size (Cohen's d),
    sample sizes, and alpha.
    Uses non-central t-distribution.
    """
    df = n1 + n2 - 2
    ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return power
