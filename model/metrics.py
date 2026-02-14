from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_auc_score(
    y_true: np.ndarray,
    y_probabilities: np.ndarray | None,
    n_classes: int,
) -> float:
    """Compute AUC for binary and multiclass settings."""
    if y_probabilities is None:
        return float("nan")

    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_probabilities[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                y_probabilities,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        return float("nan")


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probabilities: np.ndarray | None,
    n_classes: int,
) -> dict[str, float]:
    """Calculate the assignment metrics for one trained model."""
    auc_score = compute_auc_score(
        y_true=y_true,
        y_probabilities=y_probabilities,
        n_classes=n_classes,
    )

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc_score,
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    return {
        metric_name: (round(metric_value, 4) if not math.isnan(metric_value) else float("nan"))
        for metric_name, metric_value in metrics.items()
    }
