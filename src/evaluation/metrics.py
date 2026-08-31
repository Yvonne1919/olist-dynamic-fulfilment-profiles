"""Probability metrics appropriate for imbalanced classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def probability_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    prevalence = float(y.mean())
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier_score": brier_score_loss(y, p),
        "prevalence": prevalence,
        "pr_lift": average_precision_score(y, p) / prevalence,
    }
