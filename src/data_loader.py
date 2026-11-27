"""
data_loader.py
--------------
Loads and performs initial exploration of the German Credit Dataset.

The German Credit Dataset (UCI) contains 1000 applicants described by
20 attributes. The task is to classify credit risk as GOOD (1) or BAD (0).

Sensitive attributes available:
  - Age  : originally a numeric attribute, we binarise into Young (<=25) / Older (>25)
  - Sex  : encoded from attribute 9 of the original raw data

IMPORTANT LIMITATIONS documented here (for viva):
  1. Historical bias  — data was collected in 1994 Germany; societal biases of
     that era may be encoded in the labels themselves.
  2. Proxy attributes — features such as 'employment' or 'housing' can act as
     proxies for protected characteristics.
  3. Small subgroup sizes — some intersectional groups are very small,
     making fairness estimates statistically noisy.
  4. Single jurisdiction  — results may not generalise beyond Germany or the
     credit domain.
"""

import os
import io
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Column names for the German Credit Dataset (Statlog version, 20 attributes)
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "checking_account",      # A1  – categorical
    "duration",              # A2  – numeric (months)
    "credit_history",        # A3  – categorical
    "purpose",               # A4  – categorical
    "credit_amount",         # A5  – numeric (DM)
    "savings_account",       # A6  – categorical
    "employment",            # A7  – categorical
    "installment_rate",      # A8  – numeric (% of income)
    "personal_status_sex",   # A9  – categorical (encodes sex + marital status)
    "other_debtors",         # A10 – categorical
    "residence_since",       # A11 – numeric
    "property",              # A12 – categorical
    "age",                   # A13 – numeric
    "other_installment",     # A14 – categorical
    "housing",               # A15 – categorical
    "existing_credits",      # A16 – numeric
    "job",                   # A17 – categorical
    "num_dependents",        # A18 – numeric
    "telephone",             # A19 – categorical
    "foreign_worker",        # A20 – categorical
    "credit_risk",           # Target: 1 = Good, 2 = Bad (we remap to 1/0)
]

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment", "personal_status_sex", "other_debtors", "property",
    "other_installment", "housing", "job", "telephone", "foreign_worker",
]

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the German Credit CSV from *filepath*.

    The file is expected to be the space-separated Statlog version.
    Returns a cleaned DataFrame with:
      - Binary target  : credit_risk  (1 = Good credit, 0 = Bad credit)
      - Derived feature: age_binary   (1 = Older >25,   0 = Young <=25)
      - Derived feature: sex          (1 = Male,        0 = Female)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download it from:\n"
            "  https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data\n"
            "and save it as  data/german_credit.csv"
        )

    df = pd.read_csv(filepath, sep=" ", header=None, names=COLUMN_NAMES)

    # -- Remap target: 1 (Good) → 1, 2 (Bad) → 0 -------------------------
    df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})

    # -- Derive binary age feature ----------------------------------------
    # Threshold 25 is commonly used in fairness literature for this dataset.
    df["age_binary"] = (df["age"] > 25).astype(int)   # 1 = Older, 0 = Young

    # -- Derive sex from personal_status_sex (A9) -------------------------
    # A91 = male divorced/separated
    # A92 = female divorced/separated/married
    # A93 = male single
    # A94 = male married/widowed
    # A95 = female single
    male_codes = {"A91", "A93", "A94"}
    df["sex"] = df["personal_status_sex"].apply(
        lambda x: 1 if x in male_codes else 0
    )   # 1 = Male, 0 = Female

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of descriptive statistics about the dataset.
    Used by the Dataset Explorer page to populate summary cards.
    """
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_missing": int(df.isnull().sum().sum()),
        "n_categorical": len(CATEGORICAL_COLS),
        "n_numeric": len(NUMERIC_COLS),
        "target_distribution": df["credit_risk"].value_counts().to_dict(),
        "age_binary_distribution": df["age_binary"].value_counts().to_dict(),
        "sex_distribution": df["sex"].value_counts().to_dict(),
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
    }


def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing missing values per column."""
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
    return missing[missing["Missing Count"] > 0]


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a quick type/unique-count summary for every column."""
    rows = []
    for col in df.columns:
        rows.append({
            "Feature": col,
            "Type": str(df[col].dtype),
            "Unique Values": df[col].nunique(),
            "Sample Values": str(list(df[col].unique()[:4])),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: download the raw dataset programmatically
# (used only if the file is not already present)
# ---------------------------------------------------------------------------
def download_dataset(save_path: str) -> None:
    """
    Download the German Credit dataset from the UCI repository and save it.
    Requires an internet connection. Called from the Settings page.
    """
    import urllib.request
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "statlog/german/german.data"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
