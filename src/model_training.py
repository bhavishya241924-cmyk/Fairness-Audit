"""
model_training.py
-----------------
Trains three baseline ML models and saves them to disk.

Models:
  1. Logistic Regression  – linear, interpretable, good baseline
  2. Decision Tree        – tree-based, easy to visualise
  3. Random Forest        – ensemble, typically best raw accuracy

All models are trained on X_train / y_train and evaluated on X_test / y_test.
The fitted scaler must be applied before inference (handled in preprocessing.py).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

MODELS_DIR = "models"


def _get_model(model_name: str, random_state: int = 42):
    """Return an un-fitted sklearn estimator for *model_name*."""
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",   # helps with class imbalance
        )
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=5,               # limit depth to prevent overfitting
            random_state=random_state,
            class_weight="balanced",
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                random_state: int = 42):
    """
    Fit a model and return it.

    Parameters
    ----------
    model_name   : one of "Logistic Regression", "Decision Tree", "Random Forest"
    X_train      : training feature matrix
    y_train      : training labels
    random_state : seed for reproducibility

    Returns
    -------
    Fitted sklearn estimator.
    """
    model = _get_model(model_name, random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = "") -> dict:
    """
    Evaluate a fitted model and return a metrics dictionary.

    Returns
    -------
    dict with keys:
        model_name, accuracy, precision, recall, f1,
        confusion_matrix (2-D list), classification_report (str),
        y_pred (array)
    """
    y_pred = model.predict(X_test)

    return {
        "model_name":            model_name,
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "precision":             round(precision_score(y_test, y_pred,
                                                       zero_division=0), 4),
        "recall":                round(recall_score(y_test, y_pred,
                                                    zero_division=0), 4),
        "f1":                    round(f1_score(y_test, y_pred,
                                               zero_division=0), 4),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred,
                                                       zero_division=0),
        "y_pred":                y_pred,
    }


def save_model(model, model_name: str) -> str:
    """Save a fitted model to the models/ directory. Returns the file path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = model_name.replace(" ", "_") + ".pkl"
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(model_name: str):
    """Load a previously saved model. Returns None if not found."""
    filename = model_name.replace(" ", "_") + ".pkl"
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     random_state: int = 42) -> dict:
    """
    Train all three baseline models, evaluate them, save them, and return
    a dictionary keyed by model name.

    Returns
    -------
    {
        "Logistic Regression": {model, metrics},
        "Decision Tree":       {model, metrics},
        "Random Forest":       {model, metrics},
    }
    """
    results = {}
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        model = train_model(name, X_train, y_train, random_state)
        metrics = evaluate_model(model, X_test, y_test, name)
        save_model(model, name)
        results[name] = {"model": model, "metrics": metrics}
    return results
