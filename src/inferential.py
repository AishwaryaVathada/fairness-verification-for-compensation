"""
inferential.py
--------------
Inferential statistics: point estimation, confidence intervals, and
sampling distribution analysis.

Methods:
  - Parametric CIs (t-based and z-based)
  - Bootstrap CIs (percentile, BCa)
  - Grouped CI computation for cross-group comparisons
  - Sampling distribution simulation for CLT demonstration
  - Tolerance intervals
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg


# ---------------------------------------------------------------------------
# Parametric confidence intervals
# ---------------------------------------------------------------------------

def t_confidence_interval(
    data: np.ndarray | pd.Series,
    confidence: float = cfg.CONFIDENCE_LEVEL,
) -> dict:
    """
    Compute a t-based confidence interval for the population mean.

    Returns dict with point_estimate, se, margin_of_error, ci_low, ci_high, n.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return {"point_estimate": np.nan, "se": np.nan, "margin_of_error": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "n": n}
    mean = data.mean()
    se = stats.sem(data)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    moe = t_crit * se
    return {
        "point_estimate": mean,
        "se": se,
        "margin_of_error": moe,
        "ci_low": mean - moe,
        "ci_high": mean + moe,
        "n": n,
    }


def z_confidence_interval_proportion(
    successes: int,
    n: int,
    confidence: float = cfg.CONFIDENCE_LEVEL,
) -> dict:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return {"proportion": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    p_hat = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return {
        "proportion": p_hat,
        "ci_low": centre - margin,
        "ci_high": centre + margin,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray | pd.Series,
    statistic: str = "mean",
    n_resamples: int = cfg.BOOTSTRAP_N_RESAMPLES,
    confidence: float = cfg.CONFIDENCE_LEVEL,
    method: Literal["percentile", "bca"] = "percentile",
    seed: int = cfg.RANDOM_STATE,
) -> dict:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data : array-like
    statistic : one of 'mean', 'median', 'std', 'trimmed_mean'
    n_resamples : number of bootstrap samples
    confidence : confidence level
    method : 'percentile' or 'bca' (bias-corrected and accelerated)
    seed : RNG seed

    Returns
    -------
    Dict with observed, ci_low, ci_high, bootstrap_se, and the full
    distribution of bootstrap estimates.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)
    rng = np.random.default_rng(seed)

    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "trimmed_mean": lambda x: stats.trim_mean(x, 0.1),
    }
    func = stat_funcs.get(statistic, np.mean)
    observed = func(data)

    # resample
    boot_estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        boot_estimates[i] = func(sample)

    alpha = 1 - confidence

    if method == "percentile":
        ci_low = np.percentile(boot_estimates, 100 * alpha / 2)
        ci_high = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    elif method == "bca":
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_estimates < observed))
        # Acceleration (jackknife)
        jackknife = np.empty(n)
        for i in range(n):
            jk_sample = np.delete(data, i)
            jackknife[i] = func(jk_sample)
        jk_mean = jackknife.mean()
        num = np.sum((jk_mean - jackknife) ** 3)
        den = 6 * (np.sum((jk_mean - jackknife) ** 2) ** 1.5)
        a = num / den if den != 0 else 0

        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        def _adjust(z_a):
            return stats.norm.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a)))

        ci_low = np.percentile(boot_estimates, 100 * _adjust(z_alpha_low))
        ci_high = np.percentile(boot_estimates, 100 * _adjust(z_alpha_high))

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "observed": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_se": boot_estimates.std(),
        "n_resamples": n_resamples,
        "method": method,
        "distribution": boot_estimates,
    }


# ---------------------------------------------------------------------------
# Grouped confidence intervals
# ---------------------------------------------------------------------------

def grouped_ci(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    sub_group_col: str | None = None,
    confidence: float = cfg.CONFIDENCE_LEVEL,
    use_bootstrap: bool = False,
) -> pd.DataFrame:
    """
    Compute confidence intervals for ``value_col`` within each level of
    ``group_col`` (and optionally ``sub_group_col``).

    Parameters
    ----------
    df : DataFrame
    value_col : numeric column to estimate
    group_col : primary grouping variable (e.g. Position)
    sub_group_col : optional second grouping (e.g. Gender)
    confidence : confidence level
    use_bootstrap : if True, use bootstrap percentile CI; else t-based

    Returns
    -------
    DataFrame with one row per group (x sub-group) combination.
    """
    group_keys = [group_col]
    if sub_group_col:
        group_keys.append(sub_group_col)

    records = []
    for keys, grp in df.groupby(group_keys):
        vals = grp[value_col].dropna().values
        if use_bootstrap:
            ci = bootstrap_ci(vals, confidence=confidence)
            rec = {
                "point_estimate": ci["observed"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "se": ci["bootstrap_se"],
                "n": len(vals),
            }
        else:
            rec = t_confidence_interval(vals, confidence=confidence)

        if isinstance(keys, tuple):
            for k, v in zip(group_keys, keys):
                rec[k] = v
        else:
            rec[group_keys[0]] = keys
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sampling distribution simulation
# ---------------------------------------------------------------------------

def simulate_sampling_distribution(
    population: np.ndarray | pd.Series,
    sample_size: int = 100,
    n_samples: int = 5000,
    statistic: str = "mean",
    seed: int = cfg.RANDOM_STATE,
) -> dict:
    """
    Draw ``n_samples`` random samples of size ``sample_size`` from
    ``population`` and compute the chosen statistic for each.
    Useful for demonstrating CLT and assessing estimator properties.

    Returns
    -------
    Dict with population parameter, sampling distribution array,
    mean of sampling dist, SE (theoretical vs. empirical), and
    normality test on the sampling distribution.
    """
    population = np.asarray(population)
    population = population[~np.isnan(population)]
    rng = np.random.default_rng(seed)

    stat_funcs = {"mean": np.mean, "median": np.median, "std": np.std}
    func = stat_funcs.get(statistic, np.mean)

    sampling_dist = np.array([
        func(rng.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ])

    pop_mean = population.mean()
    pop_std = population.std()
    theoretical_se = pop_std / np.sqrt(sample_size)

    # Normality of sampling distribution (Shapiro on subsample)
    sub = sampling_dist[:min(5000, len(sampling_dist))]
    _, norm_p = stats.shapiro(sub)

    return {
        "population_mean": pop_mean,
        "population_std": pop_std,
        "sampling_dist": sampling_dist,
        "sampling_mean": sampling_dist.mean(),
        "empirical_se": sampling_dist.std(),
        "theoretical_se": theoretical_se,
        "se_ratio": sampling_dist.std() / theoretical_se if theoretical_se > 0 else np.nan,
        "normality_p": norm_p,
        "sample_size": sample_size,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Tolerance interval
# ---------------------------------------------------------------------------

def normal_tolerance_interval(
    data: np.ndarray | pd.Series,
    coverage: float = 0.95,
    confidence: float = cfg.CONFIDENCE_LEVEL,
) -> dict:
    """
    Two-sided normal tolerance interval: an interval that contains at
    least ``coverage`` proportion of the population with ``confidence``
    probability.  Uses the k-factor approach.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)
    mean = data.mean()
    s = data.std(ddof=1)

    # k-factor (Howe's approximation)
    z_p = stats.norm.ppf((1 + coverage) / 2)
    chi2_val = stats.chi2.ppf(1 - confidence, df=n - 1)
    k = z_p * np.sqrt((n - 1) * (1 + 1 / n) / chi2_val)

    return {
        "mean": mean,
        "std": s,
        "k_factor": k,
        "tol_low": mean - k * s,
        "tol_high": mean + k * s,
        "coverage": coverage,
        "confidence": confidence,
        "n": n,
    }
