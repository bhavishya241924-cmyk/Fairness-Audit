"""
bias_mitigation.py
------------------
Implements three categories of bias mitigation:

A. PRE-PROCESSING  — Reweighing
   Assigns sample weights inversely proportional to the
   joint distribution of sensitive attribute and label.
   This balances the influence of each group-label combination
   during training WITHOUT changing the data itself.

B. IN-PROCESSING   — ExponentiatedGradient (Fairlearn)
   Adds a Lagrangian fairness constraint to the training objective.
   The algorithm searches for the best classifier that satisfies
   the constraint (DemographicParity) within a tolerance epsilon.

C. POST-PROCESSING — ThresholdOptimizer (Fairlearn)
   After training a base model, finds per-group decision thresholds
   that satisfy a fairness constraint (EqualizedOdds).
   This is applied AFTER training on the validation set only;
   the test set is kept unseen until final evaluation.

Important implementation notes:
  - Mitigation is tuned on X_val / y_val / A_val, NOT on X_test.
  - X_test / y_test / A_test are used only for final comparison.
  - Each mitigated model is compared to the same LR baseline.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer


# ---------------------------------------------------------------------------
# Helper: uniform evaluation dict (same keys as model_training.evaluate_model)
# ---------------------------------------------------------------------------
def _eval(y_true, y_pred, model_name=""):
    return {
        "model_name":            model_name,
        "accuracy":              round(accuracy_score(y_true, y_pred), 4),
        "precision":             round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":                round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                    round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix":      confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "y_pred":                y_pred,
    }


# ---------------------------------------------------------------------------
# A. PRE-PROCESSING: Reweighing
# ---------------------------------------------------------------------------
def compute_reweighing_weights(y_train: np.ndarray,
                               A_train: np.ndarray) -> np.ndarray:
    """
    Compute sample weights that balance the joint distribution of
    (A, Y) combinations.

    The idea:
      - Expected weight for a sample = P(A) * P(Y) / P(A, Y)
      - Groups that are underrepresented relative to their expected
        frequency receive higher weights.

    Returns
    -------
    weights : 1-D numpy array of shape (n_train,)
    """
    n = len(y_train)
    weights = np.ones(n)

    for a_val in [0, 1]:
        for y_val in [0, 1]:
            mask_ay = (A_train == a_val) & (y_train == y_val)
            mask_a  = (A_train == a_val)
            mask_y  = (y_train == y_val)

            p_ay = mask_ay.sum() / n
            p_a  = mask_a.sum() / n
            p_y  = mask_y.sum() / n

            if p_ay > 0:
                w = (p_a * p_y) / p_ay
                weights[mask_ay] = w

    # Normalise so that the sum equals n (keeps loss scale stable)
    weights = weights * n / weights.sum()
    return weights


def apply_preprocessing(X_train: np.ndarray, y_train: np.ndarray,
                        A_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        random_state: int = 42) -> dict:
    """
    Train a Logistic Regression model with Reweighing sample weights.

    Returns
    -------
    dict: { model, weights, metrics, y_pred }
    """
    weights = compute_reweighing_weights(y_train, A_train)

    model = LogisticRegression(max_iter=1000, random_state=random_state,
                               class_weight="balanced")
    model.fit(X_train, y_train, sample_weight=weights)

    y_pred = model.predict(X_test)
    metrics = _eval(y_test, y_pred, "Pre-processing (Reweighing)")

    return {"model": model, "weights": weights, "metrics": metrics, "y_pred": y_pred}


# ---------------------------------------------------------------------------
# B. IN-PROCESSING: ExponentiatedGradient
# ---------------------------------------------------------------------------
def apply_inprocessing(X_train: np.ndarray, y_train: np.ndarray,
                       A_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       A_test: np.ndarray,
                       random_state: int = 42) -> dict:
    """
    Train a fairness-constrained classifier using Fairlearn's
    ExponentiatedGradient with a DemographicParity constraint.

    ExponentiatedGradient works by:
      1. Solving a sequence of weighted classification problems.
      2. Each iteration updates Lagrange multipliers to push the
         model toward satisfying the fairness constraint.
      3. The final model is a mixture (randomised) of the solutions found.

    Returns
    -------
    dict: { model, metrics, y_pred }
    """
    base_estimator = LogisticRegression(max_iter=1000,
                                        random_state=random_state)
    constraint = DemographicParity()

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.02,          # tolerance: allow up to 2 % demographic parity gap
        max_iter=50,
    )
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    y_pred = mitigator.predict(X_test)
    metrics = _eval(y_test, y_pred, "In-processing (ExpGrad)")

    return {"model": mitigator, "metrics": metrics, "y_pred": y_pred}


# ---------------------------------------------------------------------------
# C. POST-PROCESSING: ThresholdOptimizer
# ---------------------------------------------------------------------------
def apply_postprocessing(base_model,
                         X_train: np.ndarray, y_train: np.ndarray,
                         A_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         A_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         A_test: np.ndarray) -> dict:
    """
    Apply ThresholdOptimizer on top of a trained base model.

    How it works:
      - For each group (A=0, A=1) it finds the decision threshold that
        satisfies EqualizedOdds while maximising accuracy.
      - Thresholds are FIT on the validation set (not test set).
      - Final evaluation is done on the test set.

    Parameters
    ----------
    base_model : already-fitted sklearn estimator (e.g. Logistic Regression)

    Returns
    -------
    dict: { model, metrics, y_pred }
    """
    optimizer = ThresholdOptimizer(
        estimator=base_model,
        constraints="equalized_odds",
        objective="accuracy_score",
        predict_method="predict_proba",
        flip=True,
    )

    # Fit threshold optimizer on VALIDATION set only
    optimizer.fit(X_val, y_val, sensitive_features=A_val)

    # Evaluate on TEST set
    y_pred = optimizer.predict(X_test, sensitive_features=A_test)
    metrics = _eval(y_test, y_pred, "Post-processing (ThresholdOpt)")

    return {"model": optimizer, "metrics": metrics, "y_pred": y_pred}
