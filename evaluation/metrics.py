"""Evaluation helpers for the Week 1 Branch A baseline."""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0
    probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[~positive_mask])
    probabilities[~positive_mask] = exp_logits / (1.0 + exp_logits)
    return probabilities


def compute_binary_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    average_loss: float,
) -> Dict[str, float]:
    probabilities = _stable_sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=cast(Any, 0))),
        "loss": float(average_loss),
    }
