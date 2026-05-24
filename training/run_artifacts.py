"""Run-level plot artifacts for training outputs."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from evaluation import plot_confusion_matrix

try:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:
    matplotlib = None
    plt = None


class _HistoryRow(Protocol):
    epoch: int
    split: str
    loss: float
    balanced_accuracy: float
    f1: float


def _smooth(values: Sequence[float], window: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    if array.size < window:
        window = max(1, int(array.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(array, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def write_confusion_matrix_artifacts(
    run_dir: str | Path,
    *,
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
) -> None:
    if plt is None:
        return
    run_path = Path(run_dir)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    normalized = matrix.astype(np.float64)
    column_totals = normalized.sum(axis=0, keepdims=True)
    np.divide(normalized, np.maximum(column_totals, 1.0), out=normalized, where=column_totals > 0)

    plot_confusion_matrix(labels, predictions, run_path / "confusion_matrix.png", class_names)
    _plot_confusion_matrix(
        run_path / "confusion_matrix_normalized.png",
        matrix=normalized,
        class_names=class_names,
        title="Confusion Matrix Normalized",
        value_format=".2f",
    )


def write_results_plot(run_dir: str | Path, history: Iterable[_HistoryRow]) -> None:
    if plt is None:
        return
    run_path = Path(run_dir)
    train_rows = [row for row in history if getattr(row, "split", None) == "train"]
    val_rows = [row for row in history if getattr(row, "split", None) == "val"]
    if not train_rows or not val_rows:
        return

    epochs = [int(row.epoch) for row in train_rows]
    train_loss = [float(row.loss) for row in train_rows]
    val_loss = [float(row.loss) for row in val_rows]
    accuracy_top1 = [float(row.balanced_accuracy) for row in val_rows]
    accuracy_top5 = [float(row.f1) for row in val_rows]

    figure, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    plot_specs = [
        ("train/loss", train_loss),
        ("metrics/accuracy_top1", accuracy_top1),
        ("val/loss", val_loss),
        ("metrics/accuracy_top5", accuracy_top5),
    ]
    for axis, (title, values) in zip(axes, plot_specs):
        smoothed = _smooth(values)
        axis.plot(epochs, values, marker="o", linewidth=2.5, label="results")
        axis.plot(epochs, smoothed, linestyle=":", linewidth=3.0, label="smooth")
        axis.set_title(title, fontsize=18)
        axis.grid(alpha=0.2)
        if title.startswith("metrics/"):
            minimum = min(values)
            maximum = max(values)
            if minimum == maximum:
                pad = 0.05 if minimum == 0 else max(0.02, abs(minimum) * 0.05)
                axis.set_ylim(minimum - pad, maximum + pad)
            else:
                pad = max(0.01, (maximum - minimum) * 0.15)
                axis.set_ylim(max(0.0, minimum - pad), min(1.05, maximum + pad))
        if title == "metrics/accuracy_top1":
            axis.legend(loc="lower right", framealpha=0.9)

    figure.tight_layout()
    figure.savefig(run_path / "results.png", dpi=150)
    plt.close(figure)


def _plot_confusion_matrix(
    path: Path,
    *,
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str,
    value_format: str,
) -> None:
    if plt is None:
        return
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
    figure.savefig(path, dpi=150)
    plt.close(figure)


__all__ = ["write_confusion_matrix_artifacts", "write_results_plot"]
