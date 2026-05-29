"""RF ensemble module for Week 3 Dev 2.

Implements:
  - extract_branch_outputs: run a Phase 3/4 model over a dataloader and collect
    per-branch feature vectors and labels.
  - train_rf_ensemble: fit a RandomForestClassifier on arbitrary feature columns.
  - evaluate_ensemble: compute balanced accuracy / F1 / AUC-ROC for a fitted clf.
  - COMBO_CONFIGS: canonical 7 branch-combination descriptors used by the runner.

Branch feature dimensions (active runtime contract):
  A  — 2048-D  (BranchAEncoder output)
  B  —   32-D  (BranchB expander output)
  C  —   28-D  (BranchC_Physics output)

Usage
-----
  from evaluation.ensemble import extract_branch_outputs, train_rf_ensemble, evaluate_ensemble

  features, labels = extract_branch_outputs(model, dataloader, device)
  clf = train_rf_ensemble(features["b_c"], labels)
  metrics = evaluate_ensemble(clf, features["b_c"], labels)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.eval import (
    compute_auc_roc,
    compute_balanced_accuracy,
    compute_f1,
    plot_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Combo configuration
# ---------------------------------------------------------------------------

class ComboConfig(NamedTuple):
    key: str                  # short id used for filenames / dict keys
    label: str                # human-readable name
    branches: Tuple[str, ...]  # which branch keys to concatenate ("a", "b", "c")
    classifier: str           # "logistic" | "rf"


COMBO_CONFIGS: List[ComboConfig] = [
    ComboConfig("a",     "A only (logistic)",       ("a",),        "logistic"),
    ComboConfig("b",     "B only (logistic)",       ("b",),        "logistic"),
    ComboConfig("c",     "C only (logistic)",       ("c",),        "logistic"),
    ComboConfig("a_b",   "A + B (RF)",              ("a", "b"),    "rf"),
    ComboConfig("a_c",   "A + C (RF)",              ("a", "c"),    "rf"),
    ComboConfig("b_c",   "B + C (RF) ⭐",           ("b", "c"),    "rf"),
    ComboConfig("a_b_c", "A + B + C (RF)",          ("a", "b", "c"), "rf"),
]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

BranchFeatures = Dict[str, np.ndarray]   # keys: "a", "b", "c", "logit"


def extract_branch_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str = "extracting features",
    max_batches: Optional[int] = None,
) -> Tuple[BranchFeatures, np.ndarray]:
    """Run *model* over *dataloader* and collect per-branch feature arrays.

    The model must be a DiscriminatorPhase3 or DiscriminatorPhase4 instance (or
    any object whose ``forward_with_branch_features`` method accepts
    ``(frame_a, frame_b, flow)`` and returns a dict with keys
    ``"a"``, ``"b"``, ``"c"``, ``"logit"``).

    Returns
    -------
    features : dict with keys "a" (N×2048), "b" (N×32), "c" (N×28), "logit" (N,)
    labels   : int64 array of shape (N,)
    """
    model.eval()
    model = model.to(device)

    a_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    logit_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    batches_done = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            frame_a = batch["frame_a"].to(device)
            frame_b = batch["frame_b"].to(device)
            flow    = batch["flow"].to(device)
            labels  = batch["label"].cpu().numpy().astype(np.int64)

            out = model.forward_with_branch_features(frame_a, frame_b, flow)

            a_list.append(out["a"].cpu().numpy())
            b_list.append(out["b"].cpu().numpy())
            c_list.append(out["c"].cpu().numpy())
            logit_list.append(out["logit"].cpu().numpy())
            label_list.append(labels)

            batches_done += 1
            if max_batches is not None and batches_done >= max_batches:
                break

    features: BranchFeatures = {
        "a":     np.concatenate(a_list,     axis=0),
        "b":     np.concatenate(b_list,     axis=0),
        "c":     np.concatenate(c_list,     axis=0),
        "logit": np.concatenate(logit_list, axis=0),
    }
    labels_arr = np.concatenate(label_list, axis=0)
    return features, labels_arr


def build_combo_features(features: BranchFeatures, branches: Tuple[str, ...]) -> np.ndarray:
    """Concatenate the requested branch arrays into a single feature matrix."""
    parts = [features[b] for b in branches]
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_rf_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a RandomForestClassifier on feature matrix *X* and labels *y*.

    Parameters match the plan spec: n_estimators=100, random_state=42.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def train_logistic_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Fit a LogisticRegression classifier (used for single-branch configs)."""
    clf = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        C=1.0,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EnsembleMetrics = Dict[str, float]


def evaluate_ensemble(
    clf: RandomForestClassifier | LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> EnsembleMetrics:
    """Evaluate a fitted classifier.  Returns balanced_accuracy, f1, auc_roc."""
    predictions = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        # use probability of class 1 (fake)
        if proba.shape[1] == 2:
            scores = proba[:, 1]
        else:
            scores = proba[:, 0]
    else:
        scores = clf.decision_function(X)

    return {
        "balanced_accuracy": compute_balanced_accuracy(y, predictions),
        "f1":                compute_f1(y, predictions),
        "auc_roc":           compute_auc_roc(y, scores),
    }


# ---------------------------------------------------------------------------
# Full 7-combo ablation runner
# ---------------------------------------------------------------------------

def run_all_combos(
    train_features: BranchFeatures,
    train_labels: np.ndarray,
    test_features: BranchFeatures,
    test_labels: np.ndarray,
    *,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Train and evaluate all 7 branch-combination classifiers.

    Saves:
      - ``{output_dir}/{key}_confusion_matrix.png``
      - ``{output_dir}/{key}_confusion_matrix_normalized.png``

    Returns
    -------
    results : dict mapping combo key → {metrics, clf, ...}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}

    for combo in COMBO_CONFIGS:
        t0 = time.perf_counter()

        X_train = build_combo_features(train_features, combo.branches)
        X_test  = build_combo_features(test_features,  combo.branches)

        if combo.classifier == "rf":
            clf = train_rf_ensemble(X_train, train_labels)
        else:
            clf = train_logistic_ensemble(X_train, train_labels)

        metrics = evaluate_ensemble(clf, X_test, test_labels)
        elapsed = time.perf_counter() - t0

        # Confusion matrices
        predictions = clf.predict(X_test)
        cm_path     = output_dir / f"{combo.key}_confusion_matrix.png"
        cm_norm_path = output_dir / f"{combo.key}_confusion_matrix_normalized.png"
        plot_confusion_matrix(test_labels, predictions, cm_path)
        plot_confusion_matrix(test_labels, predictions, cm_norm_path, normalize=True)

        results[combo.key] = {
            "label":       combo.label,
            "branches":    combo.branches,
            "classifier":  combo.classifier,
            "metrics":     metrics,
            "clf":         clf,
            "duration_s":  round(elapsed, 3),
            "confusion_matrix_png":            str(cm_path),
            "confusion_matrix_normalized_png": str(cm_norm_path),
        }

        if verbose:
            m = metrics
            gate_mark = " ✓ GATE" if (
                combo.key == "b_c"
                and m["balanced_accuracy"] >= 0.944
                and m["f1"] >= 0.93
            ) else ""
            print(
                f"  [{combo.key:5s}] bal_acc={m['balanced_accuracy']:.4f}  "
                f"f1={m['f1']:.4f}  auc={m['auc_roc']:.4f}  "
                f"({elapsed:.1f}s){gate_mark}"
            )

    return results


# ---------------------------------------------------------------------------
# Per-branch neural ablation (zero other branches)
# ---------------------------------------------------------------------------

def ablate_single_branches(
    features: BranchFeatures,
    labels: np.ndarray,
    *,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, EnsembleMetrics]:
    """Evaluate each branch independently using only its neural logit.

    For single-branch ablation we use the neural discriminator's own logit
    (sigmoid threshold 0.5) rather than a new RF so that we isolate each
    branch's *trained* discriminative power with no additional fitting.

    Returns metrics dict keyed by branch name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    # The model logit already represents A+B+C (Phase 3 / Phase 4 fusion).
    # For true single-branch ablation we use the RF-on-single-branch approach
    # from COMBO_CONFIGS (configs "a", "b", "c") — which are already handled
    # by run_all_combos.  Here we add the complementary *neural logit* view
    # by evaluating sigma(logit) from the full model.
    metrics_out: Dict[str, EnsembleMetrics] = {}

    neural_logits = features["logit"]
    neural_proba  = _sigmoid(neural_logits)
    neural_preds  = (neural_proba >= 0.5).astype(np.int64)

    metrics_out["neural_full"] = {
        "balanced_accuracy": compute_balanced_accuracy(labels, neural_preds),
        "f1":                compute_f1(labels, neural_preds),
        "auc_roc":           compute_auc_roc(labels, neural_proba),
    }

    # Save confusion matrix for the full neural model too
    cm_path = output_dir / "neural_full_confusion_matrix.png"
    cm_norm_path = output_dir / "neural_full_confusion_matrix_normalized.png"
    plot_confusion_matrix(labels, neural_preds, cm_path)
    plot_confusion_matrix(labels, neural_preds, cm_norm_path, normalize=True)

    if verbose:
        m = metrics_out["neural_full"]
        print(
            f"  [neural] bal_acc={m['balanced_accuracy']:.4f}  "
            f"f1={m['f1']:.4f}  auc={m['auc_roc']:.4f}"
        )

    return metrics_out


__all__ = [
    "COMBO_CONFIGS",
    "ComboConfig",
    "BranchFeatures",
    "EnsembleMetrics",
    "ablate_single_branches",
    "build_combo_features",
    "evaluate_ensemble",
    "extract_branch_outputs",
    "run_all_combos",
    "train_logistic_ensemble",
    "train_rf_ensemble",
]
