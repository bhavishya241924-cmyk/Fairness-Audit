"""
fairness_metrics.py
-------------------
Implements four fairness metrics for binary classification:

  1. Demographic Parity    — equal selection rates across groups
  2. Equalized Odds        — equal TPR and FPR across groups
  3. Equal Opportunity     — equal TPR (True Positive Rate) across groups
  4. Predictive Parity     — equal precision across groups

Each function:
  - Accepts y_true, y_pred, and sensitive attribute A (binary 0/1)
  - Returns a detailed dict with per-group values AND the difference
  - Is compatible with Fairlearn's MetricFrame for consistency checks

Privileged group  : A == 1  (Older, age > 25)
Unprivileged group: A == 0  (Younger, age <= 25)
Positive outcome  : credit_risk == 1  (Good credit — loan approved)

Important notes for viva:
  - Different fairness metrics can CONFLICT with each other.
  - Satisfying one metric does NOT mean the model is unbiased.
  - Small subgroup sizes increase statistical uncertainty.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Helper: per-group confusion matrix components
# ---------------------------------------------------------------------------
def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   A: np.ndarray, group_val: int) -> dict:
    """
    Compute TP, FP, TN, FN, TPR, FPR, Precision, Selection Rate
    for the sub-group where A == group_val.
    """
    mask = (A == group_val)
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(mask.sum())

    if n == 0:
        return {"n": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0,
                "TPR": np.nan, "FPR": np.nan,
                "precision": np.nan, "selection_rate": np.nan}

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    sel = yp.mean()

    return {
        "n": n, "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
        "TPR": round(float(tpr), 4),
        "FPR": round(float(fpr), 4),
        "precision": round(float(prec), 4),
        "selection_rate": round(float(sel), 4),
    }


# ---------------------------------------------------------------------------
# 1. Demographic Parity
# ---------------------------------------------------------------------------
def demographic_parity(y_true: np.ndarray, y_pred: np.ndarray,
                       A: np.ndarray) -> dict:
    """
    Demographic Parity checks whether the model approves applicants at
    the SAME RATE regardless of group membership.

    Formula:
        DPD = |P(Ŷ=1 | A=0) - P(Ŷ=1 | A=1)|
        DPR = P(Ŷ=1 | A=0) / P(Ŷ=1 | A=1)

    A model satisfies demographic parity when DPD ≈ 0 (or DPR ≈ 1).
    Common fairness threshold: DPD < 0.1
    """
    g0 = _group_metrics(y_true, y_pred, A, group_val=0)
    g1 = _group_metrics(y_true, y_pred, A, group_val=1)

    sr0 = g0["selection_rate"]
    sr1 = g1["selection_rate"]

    dpd = abs(sr0 - sr1) if not (np.isnan(sr0) or np.isnan(sr1)) else np.nan
    dpr = (sr0 / sr1) if (sr1 and not np.isnan(sr0) and not np.isnan(sr1)
                          and sr1 != 0) else np.nan

    return {
        "metric": "Demographic Parity",
        "group_0": g0,
        "group_1": g1,
        "selection_rate_group_0": sr0,
        "selection_rate_group_1": sr1,
        "demographic_parity_difference": round(float(dpd), 4) if not np.isnan(dpd) else None,
        "demographic_parity_ratio":      round(float(dpr), 4) if not np.isnan(dpr) else None,
        "threshold":  0.10,
        "satisfied": bool(dpd < 0.10) if not np.isnan(dpd) else None,
    }


# ---------------------------------------------------------------------------
# 2. Equalized Odds
# ---------------------------------------------------------------------------
def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray,
                   A: np.ndarray) -> dict:
    """
    Equalized Odds requires BOTH the True Positive Rate AND the False
    Positive Rate to be equal across groups.

    Formula:
        EOdds_diff = max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)

    A smaller difference indicates fairer odds.
    """
    g0 = _group_metrics(y_true, y_pred, A, group_val=0)
    g1 = _group_metrics(y_true, y_pred, A, group_val=1)

    tpr_diff = abs(g0["TPR"] - g1["TPR"]) if not (
        np.isnan(g0["TPR"]) or np.isnan(g1["TPR"])) else np.nan
    fpr_diff = abs(g0["FPR"] - g1["FPR"]) if not (
        np.isnan(g0["FPR"]) or np.isnan(g1["FPR"])) else np.nan

    eodds_diff = max(tpr_diff, fpr_diff) if not (
        np.isnan(tpr_diff) or np.isnan(fpr_diff)) else np.nan

    return {
        "metric": "Equalized Odds",
        "group_0": g0,
        "group_1": g1,
        "TPR_group_0": g0["TPR"],
        "TPR_group_1": g1["TPR"],
        "FPR_group_0": g0["FPR"],
        "FPR_group_1": g1["FPR"],
        "TPR_difference": round(float(tpr_diff), 4) if not np.isnan(tpr_diff) else None,
        "FPR_difference": round(float(fpr_diff), 4) if not np.isnan(fpr_diff) else None,
        "equalized_odds_difference": round(float(eodds_diff), 4) if not np.isnan(eodds_diff) else None,
        "threshold": 0.10,
        "satisfied": bool(eodds_diff < 0.10) if not np.isnan(eodds_diff) else None,
    }


# ---------------------------------------------------------------------------
# 3. Equal Opportunity
# ---------------------------------------------------------------------------
def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray,
                      A: np.ndarray) -> dict:
    """
    Equal Opportunity focuses ONLY on the True Positive Rate (recall)
    for the positive class (y_true == 1).

    Formula:
        EO_diff = |TPR_0 - TPR_1|

    Among all truly creditworthy applicants, both groups should have
    an equal chance of being approved.
    """
    g0 = _group_metrics(y_true, y_pred, A, group_val=0)
    g1 = _group_metrics(y_true, y_pred, A, group_val=1)

    tpr_diff = abs(g0["TPR"] - g1["TPR"]) if not (
        np.isnan(g0["TPR"]) or np.isnan(g1["TPR"])) else np.nan

    return {
        "metric": "Equal Opportunity",
        "group_0": g0,
        "group_1": g1,
        "TPR_group_0": g0["TPR"],
        "TPR_group_1": g1["TPR"],
        "equal_opportunity_difference": round(float(tpr_diff), 4) if not np.isnan(tpr_diff) else None,
        "threshold": 0.10,
        "satisfied": bool(tpr_diff < 0.10) if not np.isnan(tpr_diff) else None,
    }


# ---------------------------------------------------------------------------
# 4. Predictive Parity
# ---------------------------------------------------------------------------
def predictive_parity(y_true: np.ndarray, y_pred: np.ndarray,
                      A: np.ndarray) -> dict:
    """
    Predictive Parity checks that when the model predicts a POSITIVE
    outcome, it is equally accurate for both groups (equal precision).

    Formula:
        PP_diff = |Precision_0 - Precision_1|
              where Precision = P(Y=1 | Ŷ=1)

    This protects against models that approve one group at a higher
    error rate than the other.
    """
    g0 = _group_metrics(y_true, y_pred, A, group_val=0)
    g1 = _group_metrics(y_true, y_pred, A, group_val=1)

    prec_diff = abs(g0["precision"] - g1["precision"]) if not (
        np.isnan(g0["precision"]) or np.isnan(g1["precision"])) else np.nan

    return {
        "metric": "Predictive Parity",
        "group_0": g0,
        "group_1": g1,
        "precision_group_0": g0["precision"],
        "precision_group_1": g1["precision"],
        "predictive_parity_difference": round(float(prec_diff), 4) if not np.isnan(prec_diff) else None,
        "threshold": 0.10,
        "satisfied": bool(prec_diff < 0.10) if not np.isnan(prec_diff) else None,
    }


# ---------------------------------------------------------------------------
# Convenience: run all four metrics at once
# ---------------------------------------------------------------------------
def run_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    A: np.ndarray, model_name: str = "") -> dict:
    """
    Run all four fairness metrics and return a consolidated dictionary.

    Parameters
    ----------
    y_true  : ground truth labels
    y_pred  : model predictions
    A       : sensitive attribute (binary 0/1)
    model_name : label for display purposes

    Returns
    -------
    {
        "model_name": ...,
        "demographic_parity": {...},
        "equalized_odds": {...},
        "equal_opportunity": {...},
        "predictive_parity": {...},
    }
    """
    return {
        "model_name":        model_name,
        "demographic_parity":  demographic_parity(y_true, y_pred, A),
        "equalized_odds":      equalized_odds(y_true, y_pred, A),
        "equal_opportunity":   equal_opportunity(y_true, y_pred, A),
        "predictive_parity":   predictive_parity(y_true, y_pred, A),
    }


def metrics_summary_df(all_results: dict) -> pd.DataFrame:
    """
    Convert run_all_metrics() output for MULTIPLE models into a
    comparison DataFrame suitable for display in the dashboard.

    Parameters
    ----------
    all_results : {model_name: run_all_metrics(...)}

    Returns
    -------
    DataFrame with one row per model and columns for each fairness score.
    """
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model": name,
            "Dem. Parity Diff": res["demographic_parity"]["demographic_parity_difference"],
            "Eq. Odds Diff":    res["equalized_odds"]["equalized_odds_difference"],
            "Eq. Opp. Diff":    res["equal_opportunity"]["equal_opportunity_difference"],
            "Pred. Parity Diff": res["predictive_parity"]["predictive_parity_difference"],
        })
    return pd.DataFrame(rows)
