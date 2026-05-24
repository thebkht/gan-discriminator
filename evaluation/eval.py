"""Evaluation helpers shared across training phases."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Sequence, cast

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

try:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:
    matplotlib = None
    plt = None


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0
    probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[~positive_mask])
    probabilities[~positive_mask] = exp_logits / (1.0 + exp_logits)
    return probabilities


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=cast(Any, 0)))


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    class_names: Sequence[str] = ("real", "fake"),
    *,
    normalize: bool = False,
) -> None:
    if plt is None:
        return
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        matrix = matrix.astype(np.float64)
        column_totals = matrix.sum(axis=0, keepdims=True)
        np.divide(matrix, np.maximum(column_totals, 1.0), out=matrix, where=column_totals > 0)
        title = "Confusion Matrix Normalized"
        value_format = ".2f"
    else:
        title = "Confusion Matrix"
        value_format = "d"

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title, fontsize=18, pad=16)
    axis.set_xlabel("True", fontsize=14)
    axis.set_ylabel("Predicted", fontsize=14)
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=14)
    axis.set_yticklabels(class_names, fontsize=14)

    threshold = float(np.max(matrix)) / 2.0 if matrix.size > 0 else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = "white" if value > threshold else "black"
            axis.text(
                col_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color=color,
                fontsize=14,
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(Path(save_path), dpi=150)
    plt.close(figure)


def compute_binary_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    average_loss: float,
) -> Dict[str, float]:
    probabilities = _stable_sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "balanced_accuracy": compute_balanced_accuracy(labels, predictions),
        "f1": compute_f1(labels, predictions),
        "auc_roc": compute_auc_roc(labels, probabilities),
        "loss": float(average_loss),
    }


__all__ = [
    "compute_auc_roc",
    "compute_balanced_accuracy",
    "compute_binary_classification_metrics",
    "compute_f1",
    "plot_confusion_matrix",
]
