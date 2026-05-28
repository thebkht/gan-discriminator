"""Evaluate available branch checkpoints and write prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.celeba_loader import PairingMode, create_celeba_dataloader, load_config
from evaluation.eval import compute_binary_classification_metrics, plot_confusion_matrix
from models import (
    BranchABaseline,
    DiscriminatorPhase2,
    DiscriminatorPhase3,
    DiscriminatorPhase4,
    load_phase2_checkpoint,
    load_phase3_checkpoint,
)


def _as_dict(value: object, *, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for {context}")
    return {str(key): item for key, item in value.items()}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0
    probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[~positive_mask])
    probabilities[~positive_mask] = exp_logits / (1.0 + exp_logits)
    return probabilities


def _binary_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels_float = labels.astype(np.float64)
    logits_float = logits.astype(np.float64)
    return np.maximum(logits_float, 0.0) - (logits_float * labels_float) + np.log1p(np.exp(-np.abs(logits_float)))


def _balanced_class_indices(labels: np.ndarray) -> np.ndarray:
    real_indices = np.flatnonzero(labels == 0)
    fake_indices = np.flatnonzero(labels == 1)
    class_count = min(len(real_indices), len(fake_indices))
    if class_count == 0:
        raise ValueError("Balanced eval requires at least one real and one fake example")
    balanced_indices = np.concatenate([real_indices[:class_count], fake_indices[:class_count]])
    return np.sort(balanced_indices)


def _write_predictions(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = [
        "index",
        "anchor_path",
        "pair_path",
        "pair_type",
        "pair_strategy",
        "label",
        "logit",
        "probability",
        "prediction",
        "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_eval(
    *,
    name: str,
    model: nn.Module,
    checkpoint_path: Path,
    config: Dict[str, Any],
    run_dir: Path,
    split: str,
    device: torch.device,
    include_flow: bool,
    pairing_mode: PairingMode,
    limit: Optional[int],
    dataloader_overrides: Dict[str, object],
    balance_classes: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    dataloader = create_celeba_dataloader(
        config,
        split=split,
        shuffle=False,
        limit=limit,
        include_flow=include_flow,
        pairing_mode=pairing_mode,
        dataloader_overrides=dataloader_overrides,
    )

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    prediction_rows: list[Dict[str, object]] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            frame_a = batch["frame_a"].to(device)
            frame_b = batch["frame_b"].to(device)
            labels = batch["label"].float().to(device)
            if include_flow:
                logits = model(frame_a, frame_b, batch["flow"].to(device))
            else:
                logits = model(frame_a, frame_b)

            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()
            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            for item_index, metadata in enumerate(batch["metadata"]):
                label_value = int(labels[item_index].item())
                prediction_value = int(predictions[item_index].item())
                prediction_rows.append(
                    {
                        "index": len(prediction_rows),
                        "anchor_path": metadata.get("anchor_path", ""),
                        "pair_path": metadata.get("pair_path", ""),
                        "pair_type": metadata.get("pair_type", ""),
                        "pair_strategy": metadata.get("pair_strategy", ""),
                        "label": label_value,
                        "logit": f"{float(logits[item_index].item()):.8f}",
                        "probability": f"{float(probabilities[item_index].item()):.8f}",
                        "prediction": prediction_value,
                        "correct": int(prediction_value == label_value),
                    }
                )

            if batch_index == 1 or batch_index % 50 == 0:
                print(f"{name}: processed {total_examples} examples", flush=True)

    if total_examples == 0:
        raise ValueError(f"{name} received an empty dataloader")

    logits_array = np.concatenate(all_logits)
    labels_array = np.concatenate(all_labels).astype(np.int64)
    source_examples = total_examples
    class_counts = {"real": int(np.sum(labels_array == 0)), "fake": int(np.sum(labels_array == 1))}
    if balance_classes:
        selected_indices = _balanced_class_indices(labels_array)
        logits_array = logits_array[selected_indices]
        labels_array = labels_array[selected_indices]
        prediction_rows = [prediction_rows[int(index)] for index in selected_indices]
        for index, row in enumerate(prediction_rows):
            row["index"] = index
        total_examples = int(len(selected_indices))
        total_loss = float(_binary_cross_entropy_with_logits(logits_array, labels_array).mean()) * total_examples

    probabilities_array = _stable_sigmoid(logits_array)
    predictions_array = (probabilities_array >= 0.5).astype(np.int64)
    metrics = compute_binary_classification_metrics(
        logits=logits_array,
        labels=labels_array,
        average_loss=total_loss / total_examples,
    )

    predictions_path = run_dir / f"{name}_predictions.csv"
    confusion_path = run_dir / f"{name}_confusion_matrix.png"
    normalized_confusion_path = run_dir / f"{name}_confusion_matrix_normalized.png"
    _write_predictions(predictions_path, prediction_rows)
    plot_confusion_matrix(labels_array, predictions_array, confusion_path)
    plot_confusion_matrix(labels_array, predictions_array, normalized_confusion_path, normalize=True)

    return {
        "status": "completed",
        "checkpoint_path": str(checkpoint_path),
        "include_flow": include_flow,
        "pairing_mode": pairing_mode,
        "balance_classes": balance_classes,
        "num_examples": total_examples,
        "source_examples": source_examples,
        "source_class_counts": class_counts,
        "metrics": metrics,
        "predictions_csv": str(predictions_path),
        "confusion_matrix_png": str(confusion_path),
        "confusion_matrix_normalized_png": str(normalized_confusion_path),
        "duration_seconds": time.perf_counter() - started,
    }


def _skip(checkpoint_path: Path, reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "checkpoint_path": str(checkpoint_path), "reason": reason}


def _write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = ["# Eval + Prediction Summary", ""]
    for name, result in summary["evaluations"].items():
        lines.append(f"## {name}")
        if result.get("status") == "skipped":
            lines.append("- Status: `skipped`")
            lines.append(f"- Reason: `{result['reason']}`")
        else:
            metrics = result["metrics"]
            lines.append(f"- Checkpoint: `{result['checkpoint_path']}`")
            lines.append(f"- Examples: `{result['num_examples']}`")
            lines.append(f"- Balanced accuracy: `{metrics['balanced_accuracy']:.4f}`")
            lines.append(f"- F1: `{metrics['f1']:.4f}`")
            lines.append(f"- AUC-ROC: `{metrics['auc_roc']:.4f}`")
            lines.append(f"- Loss: `{metrics['loss']:.4f}`")
            if result.get("balance_classes"):
                counts = result["source_class_counts"]
                lines.append(
                    f"- Balanced from source: `{result['source_examples']}` "
                    f"(real `{counts['real']}`, fake `{counts['fake']}`)"
                )
            lines.append(f"- Predictions: `{result['predictions_csv']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval and prediction export for available branch checkpoints")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-dir", default="runs/eval_pred_all_branches")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    config = _as_dict(load_config(args.config), context="config")
    paths_cfg = _as_dict(config["paths"], context="config.paths")
    dataloader_overrides = _as_dict(config["dataloader"], context="config.dataloader")
    dataloader_overrides["num_workers"] = 0
    dataloader_overrides.pop("prefetch_factor", None)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "split": args.split,
        "device": str(device),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "run_dir": str(run_dir),
        "started_at_unix": time.time(),
        "evaluations": {},
    }

    training_cfg = _as_dict(config["training"], context="config.training")
    branch_a_checkpoint = Path(paths_cfg["checkpoints_dir"]) / str(training_cfg["checkpoint_name"])
    if branch_a_checkpoint.exists():
        model = BranchABaseline(dropout=float(training_cfg["dropout"]))
        checkpoint = torch.load(branch_a_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        summary["evaluations"]["branch_a"] = _run_eval(
            name="branch_a",
            model=model,
            checkpoint_path=branch_a_checkpoint,
            config=config,
            run_dir=run_dir,
            split=args.split,
            device=device,
            include_flow=False,
            pairing_mode="default",
            limit=args.limit,
            dataloader_overrides=dataloader_overrides,
        )
    else:
        summary["evaluations"]["branch_a"] = _skip(branch_a_checkpoint, "checkpoint missing")

    phase2_cfg = _as_dict(config["phase2"], context="config.phase2")
    phase2_checkpoint = Path(paths_cfg["checkpoints_dir"]) / str(phase2_cfg["checkpoint_name"])
    if phase2_checkpoint.exists():
        try:
            model = DiscriminatorPhase2(
                dropout=float(phase2_cfg["dropout"]),
                backbone_train_last_n=int(phase2_cfg["backbone_train_last_n"]),
            )
            load_phase2_checkpoint(model, phase2_checkpoint)
            summary["evaluations"]["phase2_a_b"] = _run_eval(
                name="phase2_a_b",
                model=model,
                checkpoint_path=phase2_checkpoint,
                config=config,
                run_dir=run_dir,
                split=args.split,
                device=device,
                include_flow=False,
                pairing_mode="default",
                limit=args.limit,
                dataloader_overrides=dataloader_overrides,
            )
        except Exception as exc:
            summary["evaluations"]["phase2_a_b"] = _skip(phase2_checkpoint, f"{type(exc).__name__}: {exc}")
    else:
        summary["evaluations"]["phase2_a_b"] = _skip(phase2_checkpoint, "checkpoint missing")

    phase3_cfg = _as_dict(config["phase3"], context="config.phase3")
    phase3_checkpoint = Path(paths_cfg["checkpoints_dir"]) / str(phase3_cfg["checkpoint_name"])
    if phase3_checkpoint.exists():
        try:
            model = DiscriminatorPhase3(dropout=float(phase3_cfg["dropout"]))
            load_phase3_checkpoint(model, None, None, phase3_checkpoint)
            summary["evaluations"]["phase3_a_b_c"] = _run_eval(
                name="phase3_a_b_c",
                model=model,
                checkpoint_path=phase3_checkpoint,
                config=config,
                run_dir=run_dir,
                split=args.split,
                device=device,
                include_flow=True,
                pairing_mode="adjacent_cache",
                limit=args.limit,
                dataloader_overrides=dataloader_overrides,
                balance_classes=True,
            )
        except Exception as exc:
            summary["evaluations"]["phase3_a_b_c"] = _skip(phase3_checkpoint, f"{type(exc).__name__}: {exc}")
    else:
        summary["evaluations"]["phase3_a_b_c"] = _skip(phase3_checkpoint, "checkpoint missing")

    phase4_cfg = _as_dict(config.get("phase4", {}), context="config.phase4")
    phase4_checkpoint = Path(paths_cfg["checkpoints_dir"]) / str(phase4_cfg.get("checkpoint_name", "phase4_ensemble.pt"))
    if phase4_checkpoint.exists():
        try:
            model = DiscriminatorPhase4(dropout=float(phase4_cfg["dropout"]))
            checkpoint = torch.load(phase4_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            summary["evaluations"]["phase4_ensemble"] = _run_eval(
                name="phase4_ensemble",
                model=model,
                checkpoint_path=phase4_checkpoint,
                config=config,
                run_dir=run_dir,
                split=args.split,
                device=device,
                include_flow=True,
                pairing_mode="adjacent_cache",
                limit=args.limit,
                dataloader_overrides=dataloader_overrides,
                balance_classes=True,
            )
        except Exception as exc:
            summary["evaluations"]["phase4_ensemble"] = _skip(phase4_checkpoint, f"{type(exc).__name__}: {exc}")
    else:
        summary["evaluations"]["phase4_ensemble"] = _skip(phase4_checkpoint, "checkpoint missing")

    summary["finished_at_unix"] = time.time()
    summary_json_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md_path, summary)
    print(f"Wrote {summary_json_path}", flush=True)
    print(f"Wrote {summary_md_path}", flush=True)


if __name__ == "__main__":
    main()
