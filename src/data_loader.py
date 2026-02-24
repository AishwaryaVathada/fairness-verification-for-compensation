"""
data_loader.py
--------------
Data ingestion, validation, and preprocessing for the compensation analysis
pipeline. Handles CSV loading, schema validation, missing-value diagnostics,
position title normalization, and column subsetting.

Designed for reuse: any CSV with a compatible schema can be piped through
``load_and_prepare()`` to produce a clean analysis-ready DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path | str = cfg.DATA_FILE) -> pd.DataFrame:
    """Read the raw CSV and coerce basic types."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d cols from %s", *df.shape, path.name)
    return df


# ---------------------------------------------------------------------------
# Validation & diagnostics
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column (count and %)."""
    total = df.isnull().sum()
    pct = (total / len(df)) * 100
    report = pd.DataFrame({"missing_count": total, "missing_pct": pct})
    report = report[report.missing_count > 0].sort_values("missing_pct", ascending=False)
    return report


def schema_check(df: pd.DataFrame) -> dict:
    """Basic schema diagnostics: shape, dtypes, duplicate rows."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_summary": missing_value_report(df),
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_positions(
    series: pd.Series,
    keywords: Sequence[str] = cfg.POSITION_KEYWORDS,
    fallback: str = "Other",
) -> pd.Series:
    """
    Collapse granular job titles into broad position levels by matching
    the first keyword found in each title.

    Parameters
    ----------
    series : pd.Series
        Raw position/title strings.
    keywords : sequence of str
        Ordered keywords to match (first match wins).
    fallback : str
        Label assigned when no keyword matches.

    Returns
    -------
    pd.Series with normalized position labels.
    """
    def _match(title: str) -> str:
        for kw in keywords:
            if kw.lower() in title.lower():
                return kw
        return fallback

    return series.astype(str).apply(_match)


def drop_irrelevant_columns(
    df: pd.DataFrame,
    cols: Sequence[str] = cfg.DROP_COLS,
) -> pd.DataFrame:
    """Drop columns not needed for compensation analysis."""
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features useful for downstream analysis:
      - comp_ratio       : Salary / group median salary (within position)
      - ha_pct_gross     : HA as percentage of Gross Salary
      - cola_pct_gross   : COLA as percentage of Gross Salary
      - cpf_pct_gross    : CPF as percentage of Gross Salary
      - tenure_exp_ratio : In Company Years / Year of Experience
    """
    df = df.copy()

    if "Position" in df.columns and "Salary" in df.columns:
        median_salary = df.groupby("Position")["Salary"].transform("median")
        df["comp_ratio"] = df["Salary"] / median_salary

    for comp, col in [("HA", "ha_pct_gross"), ("COLA", "cola_pct_gross"), ("CPF", "cpf_pct_gross")]:
        if comp in df.columns and "Gross Salary" in df.columns:
            df[col] = (df[comp] / df["Gross Salary"]) * 100

    if "In Company Years" in df.columns and "Year of Experience" in df.columns:
        df["tenure_exp_ratio"] = np.where(
            df["Year of Experience"] > 0,
            df["In Company Years"] / df["Year of Experience"],
            np.nan,
        )

    return df


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def load_and_prepare(path: Path | str = cfg.DATA_FILE) -> pd.DataFrame:
    """
    Full pipeline: load -> validate -> normalize positions -> drop cols
    -> add derived features.

    Returns a clean, analysis-ready DataFrame.
    """
    df = load_csv(path)

    diag = schema_check(df)
    logger.info("Shape: %s | Duplicates: %d", diag["shape"], diag["duplicated_rows"])
    if not diag["missing_summary"].empty:
        logger.warning("Missing values detected:\n%s", diag["missing_summary"])

    if "Position" in df.columns:
        df["Position"] = normalize_positions(df["Position"])

    df = drop_irrelevant_columns(df)
    df = add_derived_features(df)

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df
