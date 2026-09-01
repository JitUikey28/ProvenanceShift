"""
Candidate persona direction — construction, projection, classifier baselines, and statistics.

Methodological rules:
    - Direction / classifier is computed strictly on TRAINING data only.
    - Test data is strictly held out for final validation.
    - Sign convention: higher projection values correspond to the ASSISTANT side.
    - Effect size is reported via Cohen's d with bootstrap confidence intervals.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from src.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Mean-Difference Direction
# ---------------------------------------------------------------------------

def compute_mean_difference_direction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    assistant_label: int = 1,
) -> np.ndarray:
    """Compute the contrastive mean-difference direction vector on training data.

    v = mu_assistant - mu_alternative
    v_hat = v / ||v||_2

    Sign convention: higher projection (x . v_hat) corresponds to the assistant class.

    Parameters
    ----------
    X_train:
        2D array of shape ``(n_train, hidden_dim)``.
    y_train:
        1D array of binary labels (1 = assistant, 0 = alternative).
    assistant_label:
        The integer label for the assistant condition (default 1).

    Returns
    -------
    np.ndarray
        1D unit vector of shape ``(hidden_dim,)``.

    Raises
    ------
    ValueError
        If either class is missing in the training split or if the difference norm is zero.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    mask_asst = (y_train == assistant_label)
    mask_alt = (y_train != assistant_label)

    if not np.any(mask_asst):
        raise ValueError("No assistant training examples found to compute direction.")
    if not np.any(mask_alt):
        raise ValueError("No alternative training examples found to compute direction.")

    mu_asst = np.mean(X_train[mask_asst], axis=0)
    mu_alt = np.mean(X_train[mask_alt], axis=0)

    diff = mu_asst - mu_alt
    norm = np.linalg.norm(diff)

    if norm < 1e-12:
        raise ValueError("Mean difference vector has zero norm.")

    direction = diff / norm
    return direction


def project_representations(
    X: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Project representations onto a 1D candidate direction.

    score(x) = x . v_hat

    Parameters
    ----------
    X:
        2D array of shape ``(N, hidden_dim)``.
    direction:
        1D array of shape ``(hidden_dim,)``.

    Returns
    -------
    np.ndarray
        1D array of scalar projection scores of shape ``(N,)``.
    """
    X = np.asarray(X, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    # Ensure unit norm
    norm = np.linalg.norm(direction)
    if norm > 1e-12 and not np.isclose(norm, 1.0):
        direction = direction / norm

    return np.dot(X, direction)


# ---------------------------------------------------------------------------
# Effect Size & Statistical Metrics
# ---------------------------------------------------------------------------

def compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two 1D distributions.

    Formula:
        d = (mean_1 - mean_2) / s_pooled
        s_pooled = sqrt( ((n1 - 1)*s1^2 + (n2 - 1)*s2^2) / (n1 + n2 - 2) )

    Parameters
    ----------
    x1:
        1D array of scores for group 1 (e.g. assistant).
    x2:
        1D array of scores for group 2 (e.g. alternative).

    Returns
    -------
    float
        Standardized mean difference (Cohen's d).
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var <= 1e-12:
        return 0.0

    return float((m1 - m2) / np.sqrt(pooled_var))


def compute_projection_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute separation and classification metrics for 1D projection scores.

    Parameters
    ----------
    y_true:
        1D binary array (1 = assistant, 0 = alternative).
    scores:
        1D array of scalar projection scores.
    threshold:
        Decision boundary threshold. If None, uses the midpoint of class means.

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, balanced_accuracy, roc_auc, f1, cohens_d,
        mean_assistant, mean_alternative, mean_diff, and threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    asst_scores = scores[y_true == 1]
    alt_scores = scores[y_true == 0]

    mean_asst = float(np.mean(asst_scores)) if len(asst_scores) > 0 else 0.0
    mean_alt = float(np.mean(alt_scores)) if len(alt_scores) > 0 else 0.0
    mean_diff = mean_asst - mean_alt

    d = compute_cohens_d(asst_scores, alt_scores)

    # Determine threshold if not provided
    if threshold is None:
        threshold = (mean_asst + mean_alt) / 2.0

    y_pred = (scores >= threshold).astype(int)

    # Classification metrics
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0

    # ROC-AUC (only valid if both classes are present)
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "f1": f1,
        "cohens_d": d,
        "mean_assistant_projection": mean_asst,
        "mean_alternative_projection": mean_alt,
        "mean_projection_diff": mean_diff,
        "threshold": float(threshold),
    }


def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval for a metric.

    Parameters
    ----------
    metric_fn:
        Function taking ``(y_true_boot, scores_boot) -> float``.
    y_true:
        1D binary array of labels.
    scores:
        1D array of prediction scores or classes.
    n_bootstrap:
        Number of resamples (default 1000).
    confidence_level:
        Confidence level, e.g. 0.95 for 95% CI.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 2:
        return (0.0, 0.0)

    estimates = []
    alpha = (1.0 - confidence_level) / 2.0

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        s_b = scores[idx]
        # Skip resamples that have only 1 class if metric requires both
        if len(np.unique(y_b)) < 2:
            continue
        try:
            val = metric_fn(y_b, s_b)
            if not np.isnan(val) and not np.isinf(val):
                estimates.append(val)
        except Exception:
            continue

    if len(estimates) < 10:
        return (0.0, 0.0)

    lower = float(np.percentile(estimates, 100.0 * alpha))
    upper = float(np.percentile(estimates, 100.0 * (1.0 - alpha)))
    return (lower, upper)


# ---------------------------------------------------------------------------
# Linear Classifier Baseline
# ---------------------------------------------------------------------------

def train_linear_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    seed: int = 42,
) -> LogisticRegression:
    """Train a baseline Logistic Regression classifier on training representations.

    Parameters
    ----------
    X_train:
        2D array of shape ``(n_train, hidden_dim)``.
    y_train:
        1D binary labels.
    C:
        Inverse regularization strength.
    seed:
        Random seed.

    Returns
    -------
    LogisticRegression
        Fitted scikit-learn classifier.
    """
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a trained linear classifier.

    Parameters
    ----------
    clf:
        Fitted LogisticRegression model.
    X:
        Feature matrix.
    y:
        True labels.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, balanced_accuracy, roc_auc, and f1.
    """
    y_pred = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    if len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, probs))
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "clf_accuracy": acc,
        "clf_balanced_accuracy": bal_acc,
        "clf_roc_auc": auc,
        "clf_f1": f1,
    }


# ---------------------------------------------------------------------------
# Controls & Baselines
# ---------------------------------------------------------------------------

def generate_random_direction(dim: int, seed: int = 42) -> np.ndarray:
    """Generate a uniformly distributed random unit vector in d-dimensional space.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    seed:
        Random seed.

    Returns
    -------
    np.ndarray
        1D unit vector of shape ``(dim,)``.
    """
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def shuffle_labels(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Permute labels for the label-shuffling control experiment.

    Parameters
    ----------
    y:
        1D array of labels.
    seed:
        Random seed.

    Returns
    -------
    np.ndarray
        Permuted 1D array.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(y)
