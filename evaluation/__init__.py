"""Evaluation helpers shared across training phases."""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_auc_roc",
    "compute_balanced_accuracy",
    "compute_binary_classification_metrics",
    "compute_f1",
    "plot_confusion_matrix",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from evaluation import eval as eval_module

        return getattr(eval_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
