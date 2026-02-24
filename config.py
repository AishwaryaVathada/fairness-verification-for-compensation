"""
Configuration for Employee Compensation Analysis.

Centralizes all parameters, column definitions, and analysis settings
so that adapting this pipeline to a different dataset requires only
modifying this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

DATA_FILE = DATA_DIR / "Employee_Data.csv"

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "Salary", "Age", "Dependents", "HA", "COLA", "CPF",
    "Gross Salary", "In Company Years", "Year of Experience",
]

CATEGORICAL_COLS = [
    "Gender", "Insurance", "Marital Status", "Department", "Position",
]

COMPENSATION_COMPONENTS = ["Salary", "HA", "COLA", "CPF", "Gross Salary"]

# Columns dropped before analysis (not relevant to compensation fairness)
DROP_COLS = ["Name", "DOJ", "Dependents", "Insurance", "Marital Status"]

# ---------------------------------------------------------------------------
# Position hierarchy mapping
# ---------------------------------------------------------------------------
# Keywords used to collapse granular job titles into broad position levels.
POSITION_KEYWORDS = [
    "Director", "Manager", "Lead", "Senior", "Head",
    "Executive", "Engineer", "Associate", "Representative", "Intern",
]

# Canonical ordering from junior to senior (used for plot axes).
POSITION_ORDER = [
    "Intern", "Representative", "Associate", "Engineer", "Executive",
    "Senior", "Head", "Lead", "Manager", "Director",
]

# ---------------------------------------------------------------------------
# Statistical parameters
# ---------------------------------------------------------------------------
ALPHA = 0.05                   # global significance level
CONFIDENCE_LEVEL = 0.95        # for confidence intervals
BOOTSTRAP_N_RESAMPLES = 10_000 # bootstrap resampling iterations
RANDOM_STATE = 42              # reproducibility seed
