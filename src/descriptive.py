"""
descriptive.py
--------------
Comprehensive descriptive statistics beyond ``DataFrame.describe()``.

Includes:
  - Extended summary statistics (skewness, kurtosis, IQR, CV)
  - Normality testing (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling)
  - Categorical frequency tables with entropy
  - Group-level summaries for compensation decomposition
  - Correlation analysis with multiple methods

All functions return DataFrames/dicts suitable for serialization to CSV
or rendering in reports.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg


# ---------------------------------------------------------------------------
# Extended numeric summaries
# ---------------------------------------------------------------------------

def extended_describe(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute an extended summary for numeric columns:
    count, mean, std, min, Q1, median, Q3, max, IQR, skewness, kurtosis, CV.

    Parameters
    ----------
    df : DataFrame
    cols : columns to summarise (defaults to all numeric).

    Returns
    -------
    DataFrame indexed by column name.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in df.columns]

    records = []
    for c in cols:
        s = df[c].dropna()
        q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
        records.append({
            "variable": c,
            "count": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "Q1": q1,
            "median": median,
            "Q3": q3,
            "max": s.max(),
            "IQR": q3 - q1,
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),  # excess kurtosis
            "CV": s.std() / s.mean() if s.mean() != 0 else np.nan,
        })
    return pd.DataFrame(records).set_index("variable")


# ---------------------------------------------------------------------------
# Normality testing
# ---------------------------------------------------------------------------

def normality_tests(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    alpha: float = cfg.ALPHA,
) -> pd.DataFrame:
    """
    Run Shapiro-Wilk, D'Agostino-Pearson K^2, and Anderson-Darling tests
    on each numeric column.

    Notes
    -----
    - Shapiro-Wilk is limited to n <= 5000 by scipy; for larger samples a
      random subsample of size 5000 is drawn (with seed for reproducibility).
    - Anderson-Darling returns critical values rather than a single p-value;
      we report rejection at the 5 % significance level.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in df.columns]

    results = []
    for c in cols:
        s = df[c].dropna()
        n = len(s)
        row: dict = {"variable": c, "n": n}

        # Shapiro-Wilk (subsample if n > 5000)
        sw_sample = s if n <= 5000 else s.sample(5000, random_state=cfg.RANDOM_STATE)
        stat_sw, p_sw = stats.shapiro(sw_sample)
        row["shapiro_stat"] = stat_sw
        row["shapiro_p"] = p_sw
        row["shapiro_reject"] = p_sw < alpha

        # D'Agostino-Pearson (requires n >= 20)
        if n >= 20:
            stat_da, p_da = stats.normaltest(s)
            row["dagostino_stat"] = stat_da
            row["dagostino_p"] = p_da
            row["dagostino_reject"] = p_da < alpha
        else:
            row["dagostino_stat"] = np.nan
            row["dagostino_p"] = np.nan
            row["dagostino_reject"] = np.nan

        # Anderson-Darling
        ad_result = stats.anderson(s, dist="norm")
        row["anderson_stat"] = ad_result.statistic
        # 5% significance level is index 2 in anderson's critical_values
        idx_5pct = list(ad_result.significance_level).index(5.0) if 5.0 in ad_result.significance_level else 2
        row["anderson_cv_5pct"] = ad_result.critical_values[idx_5pct]
        row["anderson_reject"] = ad_result.statistic > ad_result.critical_values[idx_5pct]

        results.append(row)

    return pd.DataFrame(results).set_index("variable")


# ---------------------------------------------------------------------------
# Categorical summaries
# ---------------------------------------------------------------------------

def categorical_summary(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    For each categorical column, return a frequency table with count,
    proportion, and cumulative proportion. Also compute Shannon entropy
    as a measure of distributional uniformity.
    """
    if cols is None:
        cols = df.select_dtypes(include="object").columns.tolist()
    cols = [c for c in cols if c in df.columns]

    summaries: dict[str, pd.DataFrame] = {}
    for c in cols:
        vc = df[c].value_counts()
        tbl = pd.DataFrame({
            "count": vc,
            "proportion": vc / vc.sum(),
            "cumulative_proportion": (vc / vc.sum()).cumsum(),
        })
        # Shannon entropy (log base 2)
        probs = tbl["proportion"].values
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        tbl.attrs["entropy"] = entropy
        tbl.attrs["max_entropy"] = np.log2(len(probs))  # uniform baseline
        summaries[c] = tbl

    return summaries


# ---------------------------------------------------------------------------
# Group-level compensation decomposition
# ---------------------------------------------------------------------------

def compensation_decomposition(
    df: pd.DataFrame,
    group_col: str = "Position",
    components: Sequence[str] = ("Salary", "HA", "COLA", "CPF"),
    total_col: str = "Gross Salary",
) -> pd.DataFrame:
    """
    For each level of ``group_col``, compute mean values and percentage
    shares of each compensation component relative to gross salary.
    """
    available = [c for c in components if c in df.columns]
    if total_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Required columns missing: {group_col}, {total_col}")

    agg = df.groupby(group_col)[available + [total_col]].mean()
    for comp in available:
        agg[f"{comp}_pct"] = (agg[comp] / agg[total_col]) * 100
    if group_col == "Position":
        agg = agg.reindex(cfg.POSITION_ORDER)
    return agg


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix for selected columns."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr(method=method)


def pairwise_correlation_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    method: str = "pearson",
) -> dict:
    """
    Pearson or Spearman correlation with p-value and confidence interval
    via Fisher z-transformation.
    """
    clean = df[[col_a, col_b]].dropna()
    x, y = clean[col_a].values, clean[col_b].values
    n = len(x)

    if method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.pearsonr(x, y)

    # Fisher z CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - cfg.ALPHA / 2)
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)

    return {
        "method": method,
        "r": r,
        "p_value": p,
        f"ci_{cfg.CONFIDENCE_LEVEL:.0%}_low": ci_low,
        f"ci_{cfg.CONFIDENCE_LEVEL:.0%}_high": ci_high,
        "n": n,
        "significant": p < cfg.ALPHA,
    }


def grouped_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    group_col: str,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise correlation within each group level."""
    records = []
    for name, grp in df.groupby(group_col):
        res = pairwise_correlation_test(grp, col_a, col_b, method=method)
        res["group"] = name
        records.append(res)
    return pd.DataFrame(records).set_index("group")
