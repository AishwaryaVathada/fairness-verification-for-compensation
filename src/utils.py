"""
utils.py
--------
Utility functions for logging setup, results serialization,
and report generation.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config as cfg


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging."""
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


def ensure_dirs():
    """Create output directories if they do not exist."""
    for d in [cfg.DATA_DIR, cfg.RESULTS_DIR, cfg.FIGURES_DIR, cfg.TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, name: str, directory: Path = cfg.TABLES_DIR):
    """Save a DataFrame as CSV in the tables directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    df.to_csv(path)
    logging.getLogger(__name__).info("Saved table: %s", path)


def save_result_dict(result: dict, name: str, directory: Path = cfg.TABLES_DIR):
    """
    Save a results dict as JSON. DataFrames and numpy arrays within
    the dict are converted to serializable forms.
    """
    directory.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    clean = {k: _convert(v) for k, v in result.items()
             if not isinstance(v, type) and k != "distribution"}

    path = directory / f"{name}.json"
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logging.getLogger(__name__).info("Saved result: %s", path)


def print_section(title: str, width: int = 70):
    """Print a formatted section header to stdout."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def timestamp() -> str:
    """Return ISO-format timestamp string."""
    return datetime.now().isoformat(timespec="seconds")
