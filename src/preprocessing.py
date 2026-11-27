"""
preprocessing.py
----------------
Cleans and prepares the German Credit Dataset for model training.

Steps performed (in order):
  1. Drop columns that would cause data leakage.
  2. One-hot encode categorical features.
  3. Scale numeric features with StandardScaler.
  4. Separate sensitive attributes BEFORE encoding so they are
     never accidentally transformed.
  5. Perform train / validation / test split (60 / 20 / 20).

Data-leakage prevention:
  - Sensitive attributes (age_binary, sex) are separated and stored
    separately; they are NEVER used as model input features.
  - The scaler is fit ONLY on training data and then applied to val/test.
  - The splits are done with a fixed random_state for reproducibility.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import CATEGORICAL_COLS, NUMERIC_COLS


# Columns to drop from features (target + raw sensitive sources)
DROP_COLS = ["credit_risk", "personal_status_sex", "age"]


def preprocess(df: pd.DataFrame, test_size: float = 0.20,
               val_size: float = 0.20, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df           : Raw DataFrame from data_loader.load_dataset()
    test_size    : Fraction for the final held-out test set (default 20 %)
    val_size     : Fraction of the TRAINING set used for validation (default 20 %)
    random_state : Seed for reproducibility

    Returns
    -------
    A dictionary with keys:
        X_train, X_val, X_test      – feature matrices (numpy arrays)
        y_train, y_val, y_test      – target vectors
        A_train, A_val, A_test      – sensitive attribute Series (age_binary)
        feature_names               – list of feature column names
        scaler                      – fitted StandardScaler (save for inference)
        sensitive_col               – name of the default sensitive attribute
    """

    # ------------------------------------------------------------------ #
    # 1. Separate target and sensitive attributes FIRST                   #
    # ------------------------------------------------------------------ #
    y = df["credit_risk"].copy()
    sensitive = df[["age_binary", "sex"]].copy()

    # Build feature matrix: drop target, raw age, raw personal_status_sex
    # Keep the derived age_binary and sex OUT of X to prevent leakage
    feature_df = df.drop(columns=DROP_COLS + ["age_binary", "sex"])

    # ------------------------------------------------------------------ #
    # 2. One-hot encode categoricals                                      #
    # ------------------------------------------------------------------ #
    # Only encode columns that actually exist in feature_df
    cats_present = [c for c in CATEGORICAL_COLS
                    if c in feature_df.columns and c != "personal_status_sex"]

    feature_encoded = pd.get_dummies(feature_df, columns=cats_present,
                                     drop_first=False)
    feature_names = list(feature_encoded.columns)

    X = feature_encoded.values.astype(np.float64)
    y_arr = y.values

    # ------------------------------------------------------------------ #
    # 3. Train / test split                                               #
    # ------------------------------------------------------------------ #
    (X_train_full, X_test,
     y_train_full, y_test,
     A_train_full, A_test) = train_test_split(
        X, y_arr, sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y_arr,
    )

    # ------------------------------------------------------------------ #
    # 4. Train / validation split (from training portion)                 #
    # ------------------------------------------------------------------ #
    # val_size is the fraction of the original training set
    actual_val_frac = val_size / (1.0 - test_size)
    (X_train, X_val,
     y_train, y_val,
     A_train, A_val) = train_test_split(
        X_train_full, y_train_full, A_train_full,
        test_size=actual_val_frac,
        random_state=random_state,
        stratify=y_train_full,
    )

    # ------------------------------------------------------------------ #
    # 5. Scale numeric features (fit on train only)                       #
    # ------------------------------------------------------------------ #
    # Identify positions of numeric columns in the encoded feature matrix
    num_indices = [feature_names.index(c)
                   for c in NUMERIC_COLS if c in feature_names]

    scaler = StandardScaler()
    X_train[:, num_indices] = scaler.fit_transform(X_train[:, num_indices])
    X_val[:, num_indices]   = scaler.transform(X_val[:, num_indices])
    X_test[:, num_indices]  = scaler.transform(X_test[:, num_indices])

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "A_train": A_train["age_binary"],
        "A_val":   A_val["age_binary"],
        "A_test":  A_test["age_binary"],
        "A_sex_train": A_train["sex"],
        "A_sex_val":   A_val["sex"],
        "A_sex_test":  A_test["sex"],
        "feature_names": feature_names,
        "scaler": scaler,
        "sensitive_col": "age_binary",
    }
