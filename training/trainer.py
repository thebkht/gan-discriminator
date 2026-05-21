"""Training loop for the Week 1 Branch A baseline."""

from __future__ import annotations

import json
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.celeba_loader import create_celeba_dataloader, load_config
from evaluation import compute_binary_classification_metrics
from models import BranchABaseline
from training.batch_preview import maybe_save_train_preview, maybe_save_val_previews
from training.overfit_stop import OverfitStopConfig, OverfitStopMonitor
from training.run_artifacts import write_confusion_matrix_artifacts, write_results_plot
from training.tracker import Tracker


TARGET_BALANCED_ACCURACY = 0.77
TARGET_F1 = 0.70
BRANCH_A_VAL_LOSS_CEILING = 0.35

BinaryMetrics = Dict[str, float]
EpochMetrics = Dict[str, Any]


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    rounded_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(rounded_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:d}:{remaining_seconds:02d}"


def _format_progress_bar(current: int, total: int, *, width: int = 12) -> str:
    if total <= 0:
        return "─" * width
    filled = min(width, int(width * current / total))
    return ("━" * filled) + ("─" * (width - filled))


def _format_metric_value(value: float, *, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def _format_epoch_prefix(epoch: int, total_epochs: int, *, show_epoch: bool) -> str:
    formatted_prefix = f"Epoch {epoch:>2d}/{total_epochs:<2d}"
    if not show_epoch:
        return " " * len(formatted_prefix)
    return formatted_prefix


def _format_memory_label(device: torch.device) -> str:
    if device.type == "cuda":
        allocated_gb = torch.cuda.memory_allocated(device=device) / (1024**3)
        return f"{allocated_gb:.2f}G"
    return device.type.upper()


def _print_progress_header(*, split_name: str) -> None:
    if split_name.lower() == "train":
        print(f"\n{'Epoch':>11}{'GPU_mem':>11}{'loss':>11}{'Instances':>11}{'Size':>11}", flush=True)
        return
    print(f"{'split':>11}{'loss':>11}{'bal_acc':>11}{'f1':>11}", flush=True)


def _format_progress_prefix(
    *,
    epoch: int,
    total_epochs: int,
    split_name: str,
    device: torch.device,
    running_loss: float,
    batch_size: int,
    image_size: int,
) -> str:
    if split_name.lower() == "train":
        return (
            f"{f'{epoch}/{total_epochs}':>11}"
            f"{_format_memory_label(device):>11}"
            f"{_format_metric_value(running_loss, precision=6):>11}"
            f"{batch_size:>11d}"
            f"{image_size:>11d}:"
        )
    return (
        f"{split_name:>11}"
        f"{_format_metric_value(running_loss, precision=6):>11}"
        f"{'--':>11}"
        f"{'--':>11}:"
    )


def _print_epoch_result_row(
    *,
    split_name: str,
    loss: float,
    balanced_accuracy: float,
    f1: float,
) -> None:
    print(
        f"{split_name:>11}{_format_metric_value(loss, precision=6):>11}"
        f"{_format_metric_value(balanced_accuracy, precision=4):>11}"
        f"{_format_metric_value(f1, precision=4):>11}",
        flush=True,
    )


@dataclass
class EpochResult:
    epoch: int
    split: str
    loss: float
    balanced_accuracy: float
    f1: float
    learning_rate: float
    duration_seconds: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_str_key_mapping(value: object, *, context: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {context}")
    return {str(key): item for key, item in value.items()}


def _as_float(value: object, *, context: str) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected float-convertible value for {context}")
    return float(value)


def _build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]) -> Adam:
    beta_values = cast(list[object], training_cfg["betas"])
    if len(beta_values) != 2:
        raise ValueError("Adam betas must contain exactly two values")
    betas = (
        _as_float(beta_values[0], context="training.betas[0]"),
        _as_float(beta_values[1], context="training.betas[1]"),
    )
    return Adam(
        model.parameters(),
        lr=_as_float(training_cfg["learning_rate"], context="training.learning_rate"),
        betas=betas,
    )


def _build_scheduler(optimizer: Adam, training_cfg: Dict[str, Any]) -> CosineAnnealingLR:
    scheduler_name = str(training_cfg["scheduler"])
    if scheduler_name != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler for Week 1 baseline: {scheduler_name}")
    return CosineAnnealingLR(optimizer, T_max=int(training_cfg["scheduler_t_max"]))


def _resolve_early_stopping(training_cfg: Dict[str, Any]) -> OverfitStopConfig:
    raw = training_cfg.get("early_stopping")
    if raw is None:
        return OverfitStopConfig(val_loss_ceiling=BRANCH_A_VAL_LOSS_CEILING)
    early_cfg = _as_str_key_mapping(raw, context="training.early_stopping")
    return OverfitStopConfig(
        patience_overfit=int(early_cfg.get("patience_overfit", 5)),
        patience_ceiling=int(early_cfg.get("patience_ceiling", 3)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 3)),
        val_loss_ceiling=_as_float(
            early_cfg.get("val_loss_ceiling", BRANCH_A_VAL_LOSS_CEILING),
            context="training.early_stopping.val_loss_ceiling",
        ),
        enable_loss_ceiling=bool(early_cfg.get("enable_loss_ceiling", True)),
    )


def _resolve_device(device_override: Optional[str] = None) -> torch.device:
    requested = device_override.lower() if device_override is not None else None

    if requested is not None:
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested device 'cuda' but CUDA is not available")
            return torch.device("cuda")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Requested device 'mps' but MPS is not available")
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device override: {device_override}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[Adam] = None,
    epoch: int = 1,
    total_epochs: int = 1,
    split_name: str = "train",
    run_dir: Optional[Path] = None,
    include_predictions: bool = False,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    start = time.perf_counter()

    num_batches = len(dataloader)
    progress_end = "" if sys.stdout.isatty() else "\n"
    _print_progress_header(split_name=split_name)
    for batch_index, batch in enumerate(dataloader, start=1):
        frame_a = batch["frame_a"].to(device)
        frame_b = batch["frame_b"].to(device)
        labels = batch["label"].float().to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(frame_a, frame_b)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        if run_dir is not None:
            preview_batch_index = batch_index - 1
            if is_train:
                maybe_save_train_preview(
                    run_dir=run_dir,
                    batch_index=preview_batch_index,
                    frame_a=frame_a,
                    labels=labels,
                )
            else:
                maybe_save_val_previews(
                    run_dir=run_dir,
                    batch_index=preview_batch_index,
                    frame_a=frame_a,
                    labels=labels,
                    logits=logits,
                )

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        running_loss = total_loss / total_examples
        elapsed_seconds = time.perf_counter() - start
        average_batch_seconds = elapsed_seconds / batch_index
        batches_per_second = batch_index / elapsed_seconds if elapsed_seconds > 0 else 0.0
        eta_seconds = average_batch_seconds * (num_batches - batch_index)
        percent_complete = (100.0 * batch_index / num_batches) if num_batches > 0 else 0.0
        progress_prefix = _format_progress_prefix(
            epoch=epoch,
            total_epochs=total_epochs,
            split_name=split_name,
            device=device,
            running_loss=running_loss,
            batch_size=batch_size,
            image_size=int(frame_a.shape[-1]),
        )
        print(
            f"\r{progress_prefix} {percent_complete:>3.0f}% "
            f"{_format_progress_bar(batch_index, num_batches, width=14)} "
            f"{batch_index:>4d}/{num_batches:<4d} {batches_per_second:>4.1f}it/s "
            f"{_format_duration(elapsed_seconds)} < {_format_duration(eta_seconds)}",
            end=progress_end,
            flush=True,
        )

    if progress_end == "":
        print()

    if total_examples == 0:
        raise ValueError("Received an empty dataloader split; cannot compute metrics")

    average_loss = total_loss / total_examples
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(np.int64)
    metrics: EpochMetrics = compute_binary_classification_metrics(
        logits=logits,
        labels=labels,
        average_loss=average_loss,
    )
    _print_epoch_result_row(
        split_name="all" if split_name.lower() == "val" else split_name,
        loss=float(metrics["loss"]),
        balanced_accuracy=float(metrics["balanced_accuracy"]),
        f1=float(metrics["f1"]),
    )
    metrics["duration_seconds"] = time.perf_counter() - start
    metrics["num_batches"] = float(len(dataloader))
    if include_predictions:
        metrics["logits"] = logits
        metrics["labels"] = labels
    return metrics


def _serialize_history(path: Path, history: list[EpochResult]) -> None:
    payload = [asdict(entry) for entry in history]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _status_label(best_metrics: BinaryMetrics) -> str:
    meets_bal_acc = best_metrics["balanced_accuracy"] >= TARGET_BALANCED_ACCURACY
    meets_f1 = best_metrics["f1"] >= TARGET_F1
    if meets_bal_acc and meets_f1:
        return "met"
    if meets_bal_acc or meets_f1:
        return "partially met"
    return "not met"


def _build_summary_payload(
    *,
    config: Dict[str, Any],
    training_cfg: Dict[str, Any],
    best_metrics: BinaryMetrics,
    best_epoch: int,
    device: torch.device,
    run_dir: Path,
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint_selection_rule": "highest validation balanced accuracy",
        "best_epoch": best_epoch,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "best_validation_metrics": {
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "f1": best_metrics["f1"],
            "loss": best_metrics["loss"],
        },
        "targets": {
            "balanced_accuracy": TARGET_BALANCED_ACCURACY,
            "f1": TARGET_F1,
        },
        "status": _status_label(best_metrics),
        "hyperparameters": {
            "epochs": int(training_cfg["epochs"]),
            "learning_rate": float(training_cfg["learning_rate"]),
            "batch_size": int(config["dataloader"]["batch_size"]),
            "scheduler": str(training_cfg["scheduler"]),
            "scheduler_t_max": int(training_cfg["scheduler_t_max"]),
            "checkpoint_metric": str(training_cfg["checkpoint_metric"]),
        },
        "limitations": [
            "Branch A only.",
            "Fake pairs are cross-identity proxy negatives or distant-index fallbacks, not actual deepfakes.",
            "This benchmark is still a proxy task and likely optimistic relative to real deepfake detection.",
            "No out-of-domain benchmark is included in Week 1.",
            "With a 100-epoch run, the best checkpoint may occur well before epoch 100 if overfitting appears.",
        ],
    }


def _write_summary_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    json_path = run_dir / "benchmark_summary.json"
    md_path = run_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Week 1 Branch A Benchmark Summary",
                "",
                f"- Status: `{payload['status']}`",
                f"- Device: `{payload['device']}`",
                f"- Checkpoint selection: {payload['checkpoint_selection_rule']}",
                f"- Best epoch: `{payload['best_epoch']}`",
                f"- Stopped early: `{payload['stopped_early']}`",
                f"- Best validation balanced accuracy: `{payload['best_validation_metrics']['balanced_accuracy']:.4f}`",
                f"- Best validation F1: `{payload['best_validation_metrics']['f1']:.4f}`",
                f"- Best validation loss: `{payload['best_validation_metrics']['loss']:.4f}`",
                f"- Target balanced accuracy: `>= {payload['targets']['balanced_accuracy']:.2f}`",
                f"- Target F1: `>= {payload['targets']['f1']:.2f}`",
                *([f"- Stop reason: {payload['stop_reason']}"] if payload["stop_reason"] else []),
                "",
                "## Hyperparameters",
                "",
                f"- Epochs: `{payload['hyperparameters']['epochs']}`",
                f"- Batch size: `{payload['hyperparameters']['batch_size']}`",
                f"- Learning rate: `{payload['hyperparameters']['learning_rate']}`",
                f"- Scheduler: `{payload['hyperparameters']['scheduler']}`",
                f"- Scheduler T_max: `{payload['hyperparameters']['scheduler_t_max']}`",
                "",
                "## Limitations",
                "",
                *[f"- {item}" for item in payload["limitations"]],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _print_run_header(
    *,
    run_name: str,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_cfg: Dict[str, Any],
) -> None:
    print(
        (
            f"Starting Branch A baseline run '{run_name}' on {device}."
            f"epochs={training_cfg['epochs']} "
            f"train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} "
            f"batch_size={train_loader.batch_size} \n"
        ),
        flush=True,
    )
    if device.type == "cpu":
        print(
            "Warning: training is running on CPU. A full 100-epoch CelebA run may take a long time.",
            flush=True,
        )


def _print_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    current_lr: float,
    best_epoch: int,
    best_metrics: BinaryMetrics,
) -> None:
    print(
        (
            f"\n{_format_epoch_prefix(epoch, total_epochs, show_epoch=False)}   "
            f"train_loss {_format_metric_value(train_metrics['loss'], precision=6):>8}   "
            f"val_loss {_format_metric_value(val_metrics['loss'], precision=6):>8}   "
            f"bal_acc {_format_metric_value(val_metrics['balanced_accuracy'], precision=4):>6}   "
            f"f1 {_format_metric_value(val_metrics['f1'], precision=4):>6}   "
            f"lr {current_lr:>8.6f}   "
            f"best {best_epoch:>2d}/{total_epochs:<2d} {_format_metric_value(best_metrics['balanced_accuracy'], precision=4):>6}\n"
        ),
        flush=True,
    )


def _print_training_footer(
    *,
    total_epochs: int,
    total_duration_seconds: float,
    run_dir: Path,
    checkpoint_path: Path,
    summary_payload: Dict[str, Any],
    completed_epochs: int,
) -> None:
    print(f"\n{completed_epochs}/{total_epochs} epochs completed in {total_duration_seconds / 3600:.3f} hours.", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    print("\nTraining complete", flush=True)
    print(f"Best checkpoint : {checkpoint_path}", flush=True)
    print(f"Selected ckpt  : {checkpoint_path}", flush=True)
    print(f"Top-1 accuracy : {summary_payload['best_validation_metrics']['balanced_accuracy']}", flush=True)
    print(f"Top-5 accuracy : {summary_payload['best_validation_metrics']['f1']}", flush=True)
    if summary_payload["stop_reason"]:
        print(f"Stop reason    : {summary_payload['stop_reason']}", flush=True)
    print(f"Report saved   : {run_dir / 'benchmark_summary.json'}", flush=True)


def train_branch_a(
    config_path: str | Path,
    *,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    run_name: str = "branch_a_baseline",
    tracker_backend: Optional[str] = None,
    epochs_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    training_start = time.perf_counter()
    config = _as_str_key_mapping(load_config(config_path), context="config")
    training_cfg = _as_str_key_mapping(config["training"], context="config.training")
    if epochs_override is not None:
        training_cfg["epochs"] = int(epochs_override)
    effective_config = dict(config)
    effective_config["training"] = training_cfg
    _set_seed(int(training_cfg["seed"]))

    device = _resolve_device(device_override)
    model = BranchABaseline(dropout=float(training_cfg["dropout"])).to(device)
    optimizer = _build_optimizer(model, training_cfg)
    scheduler = _build_scheduler(optimizer, training_cfg)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = create_celeba_dataloader(config, split="train", limit=train_limit)
    val_loader = create_celeba_dataloader(config, split="val", shuffle=False, limit=val_limit)

    paths_cfg = _as_str_key_mapping(config["paths"], context="config.paths")
    checkpoints_dir = Path(str(paths_cfg["checkpoints_dir"]))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / str(training_cfg["checkpoint_name"])
    run_dir = Path(str(paths_cfg["runs_dir"])) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker: Optional[Tracker] = None
    if tracker_backend is not None:
        tracker = Tracker(run_dir / "tensorboard", backend=tracker_backend)

    history: list[EpochResult] = []
    best_epoch = 0
    best_metrics: Optional[BinaryMetrics] = None
    best_val_logits: Optional[np.ndarray[Any, Any]] = None
    best_val_labels: Optional[np.ndarray[Any, Any]] = None
    stop_reason: Optional[str] = None
    completed_epochs = 0
    overfit_monitor = OverfitStopMonitor(_resolve_early_stopping(training_cfg))
    _print_run_header(
        run_name=run_name,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
    )
    try:
        for epoch in range(1, int(training_cfg["epochs"]) + 1):
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                epoch=epoch,
                total_epochs=int(training_cfg["epochs"]),
                split_name="train",
                run_dir=run_dir,
            )
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                total_epochs=int(training_cfg["epochs"]),
                split_name="val",
                run_dir=run_dir,
                include_predictions=True,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            history.extend(
                [
                    EpochResult(
                        epoch=epoch,
                        split="train",
                        loss=train_metrics["loss"],
                        balanced_accuracy=train_metrics["balanced_accuracy"],
                        f1=train_metrics["f1"],
                        learning_rate=current_lr,
                        duration_seconds=train_metrics["duration_seconds"],
                    ),
                    EpochResult(
                        epoch=epoch,
                        split="val",
                        loss=val_metrics["loss"],
                        balanced_accuracy=val_metrics["balanced_accuracy"],
                        f1=val_metrics["f1"],
                        learning_rate=current_lr,
                        duration_seconds=val_metrics["duration_seconds"],
                    ),
                ]
            )

            if tracker is not None:
                tracker.log_scalar("loss/train", train_metrics["loss"], epoch)
                tracker.log_scalar("loss/val", val_metrics["loss"], epoch)
                tracker.log_scalar("balanced_accuracy/train", train_metrics["balanced_accuracy"], epoch)
                tracker.log_scalar("balanced_accuracy/val", val_metrics["balanced_accuracy"], epoch)
                tracker.log_scalar("f1/train", train_metrics["f1"], epoch)
                tracker.log_scalar("f1/val", val_metrics["f1"], epoch)
                tracker.log_scalar("lr", current_lr, epoch)

            should_replace = best_metrics is None or (
                val_metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]
            )
            if should_replace:
                best_metrics = {
                    "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                    "f1": float(val_metrics["f1"]),
                    "loss": float(val_metrics["loss"]),
                }
                best_epoch = epoch
                best_val_logits = np.asarray(val_metrics["logits"]).copy()
                best_val_labels = np.asarray(val_metrics["labels"]).copy()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_validation_metrics": {
                            "balanced_accuracy": val_metrics["balanced_accuracy"],
                            "f1": val_metrics["f1"],
                            "loss": val_metrics["loss"],
                        },
                        "config": effective_config,
                    },
                    checkpoint_path,
                )

            _print_epoch_summary(
                epoch=epoch,
                total_epochs=int(training_cfg["epochs"]),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                current_lr=current_lr,
                best_epoch=best_epoch,
                best_metrics=cast(BinaryMetrics, best_metrics),
            )
            completed_epochs = epoch
            stop_decision = overfit_monitor.update(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
            )
            if stop_decision.should_stop:
                stop_reason = stop_decision.reason
                print(stop_reason, flush=True)
                break
    finally:
        if tracker is not None:
            tracker.flush()
            tracker.close()

    if best_metrics is None:
        raise RuntimeError("Training completed without producing validation metrics")
    if best_val_logits is None or best_val_labels is None:
        raise RuntimeError("Training completed without preserving best validation predictions")

    _serialize_history(run_dir / "metrics_history.json", history)
    write_confusion_matrix_artifacts(
        run_dir,
        labels=best_val_labels,
        logits=best_val_logits,
        class_names=("real", "fake"),
    )
    write_results_plot(run_dir, history)
    summary_payload = _build_summary_payload(
        config=effective_config,
        training_cfg=training_cfg,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        device=device,
        run_dir=run_dir,
        stop_reason=stop_reason,
    )
    _write_summary_files(run_dir, summary_payload)
    _print_training_footer(
        total_epochs=int(training_cfg["epochs"]),
        total_duration_seconds=time.perf_counter() - training_start,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_payload=summary_payload,
        completed_epochs=completed_epochs,
    )
    return summary_payload
