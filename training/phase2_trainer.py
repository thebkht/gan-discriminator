"""Training loop for the Week 2 Phase 2 A+B discriminator."""

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
from models import DiscriminatorPhase2, load_pretrained_branch_a
from training.branch_a_trainer import (
    _as_float,
    _as_str_key_mapping,
    _format_duration,
    _format_epoch_prefix,
    _format_metric_value,
    _format_progress_bar,
    _resolve_device,
)
from training.tracker import Tracker


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EpochResult:
    epoch: int
    split: str
    loss: float
    balanced_accuracy: float
    f1: float
    learning_rate: float
    duration_seconds: float


def _build_optimizer(model: DiscriminatorPhase2, phase2_cfg: Dict[str, Any]) -> Adam:
    beta_values = cast(list[object], phase2_cfg["betas"])
    if len(beta_values) != 2:
        raise ValueError("Adam betas must contain exactly two values")
    betas = (
        _as_float(beta_values[0], context="phase2.betas[0]"),
        _as_float(beta_values[1], context="phase2.betas[1]"),
    )
    trainable_params = list(model.branch_b.parameters()) + list(model.fusion.parameters())
    return Adam(
        trainable_params,
        lr=_as_float(phase2_cfg["learning_rate"], context="phase2.learning_rate"),
        betas=betas,
    )


def _build_scheduler(optimizer: Adam, phase2_cfg: Dict[str, Any]) -> CosineAnnealingLR:
    scheduler_name = str(phase2_cfg["scheduler"])
    if scheduler_name != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler for Phase 2: {scheduler_name}")
    return CosineAnnealingLR(optimizer, T_max=int(phase2_cfg["scheduler_t_max"]))


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[Adam] = None,
    epoch: int = 1,
    total_epochs: int = 1,
    split_name: str = "train",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    start = time.perf_counter()

    num_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    progress_end = "" if sys.stdout.isatty() else "\n"
    for batch_index, batch in enumerate(dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        frame_a = batch["frame_a"].to(device)
        frame_b = batch["frame_b"].to(device)
        labels = batch["label"].float().to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(frame_a=frame_a, frame_b=frame_b)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

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
        epoch_prefix = _format_epoch_prefix(epoch, total_epochs, show_epoch=split_name.lower() != "val")
        print(
            f"\r{epoch_prefix}   "
            f"loss {_format_metric_value(running_loss, precision=6):>8}   "
            f"{split_name:<5} {percent_complete:>3.0f}% {_format_progress_bar(batch_index, num_batches, width=14)} "
            f"{batch_index:>3d}/{num_batches:<3d} {batches_per_second:>3.1f}it/s {_format_duration(elapsed_seconds)} "
            f"< {_format_duration(eta_seconds)}",
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
    metrics = compute_binary_classification_metrics(logits=logits, labels=labels, average_loss=average_loss)
    metrics["duration_seconds"] = time.perf_counter() - start
    metrics["num_batches"] = float(num_batches)
    return metrics


def _serialize_history(path: Path, history: list[EpochResult]) -> None:
    payload = [asdict(entry) for entry in history]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _status_label(best_metrics: Dict[str, float], targets: Dict[str, float]) -> str:
    meets_bal_acc = best_metrics["balanced_accuracy"] >= targets["balanced_accuracy"]
    meets_f1 = best_metrics["f1"] >= targets["f1"]
    if meets_bal_acc and meets_f1:
        return "met"
    if meets_bal_acc or meets_f1:
        return "partially met"
    return "not met"


def _build_summary_payload(
    *,
    config: Dict[str, Any],
    phase2_cfg: Dict[str, Any],
    best_metrics: Dict[str, float],
    best_epoch: int,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, Any]:
    targets = _as_str_key_mapping(phase2_cfg["targets"], context="phase2.targets")
    return {
        "phase": 2,
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint_selection_rule": "highest validation balanced accuracy",
        "best_epoch": best_epoch,
        "best_validation_metrics": {
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "f1": best_metrics["f1"],
            "loss": best_metrics["loss"],
        },
        "targets": {
            "balanced_accuracy": float(targets["balanced_accuracy"]),
            "f1": float(targets["f1"]),
        },
        "status": _status_label(
            best_metrics,
            {
                "balanced_accuracy": float(targets["balanced_accuracy"]),
                "f1": float(targets["f1"]),
            },
        ),
        "hyperparameters": {
            "epochs": int(phase2_cfg["epochs"]),
            "learning_rate": float(phase2_cfg["learning_rate"]),
            "batch_size": int(config["dataloader"]["batch_size"]),
            "scheduler": str(phase2_cfg["scheduler"]),
            "scheduler_t_max": int(phase2_cfg["scheduler_t_max"]),
            "checkpoint_metric": str(phase2_cfg["checkpoint_metric"]),
        },
        "limitations": [
            "Branch A is frozen from the Phase 1 checkpoint and only consumes frame_a.",
            "Fake pairs are noise-duplicate samples, not GAN-generated fakes.",
            "This benchmark is likely optimistic relative to real deepfake detection.",
            "The local dataset currently lacks identity_CelebA.txt, so real pairs use adjacent fallback.",
            "No out-of-domain benchmark is included in Week 2.",
        ],
    }


def _write_summary_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    json_path = run_dir / "benchmark_summary.json"
    md_path = run_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Week 2 Phase 2 A+B Benchmark Summary",
                "",
                f"- Phase: `{payload['phase']}`",
                f"- Status: `{payload['status']}`",
                f"- Device: `{payload['device']}`",
                f"- Checkpoint selection: {payload['checkpoint_selection_rule']}",
                f"- Best epoch: `{payload['best_epoch']}`",
                f"- Best validation balanced accuracy: `{payload['best_validation_metrics']['balanced_accuracy']:.4f}`",
                f"- Best validation F1: `{payload['best_validation_metrics']['f1']:.4f}`",
                f"- Best validation loss: `{payload['best_validation_metrics']['loss']:.4f}`",
                f"- Target balanced accuracy: `>= {payload['targets']['balanced_accuracy']:.2f}`",
                f"- Target F1: `>= {payload['targets']['f1']:.2f}`",
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
    phase2_cfg: Dict[str, Any],
) -> None:
    print(
        (
            f"Starting Phase 2 A+B run '{run_name}' on {device}. "
            f"epochs={phase2_cfg['epochs']} "
            f"train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)} "
            f"batch_size={train_loader.batch_size} \n"
        ),
        flush=True,
    )
    if device.type == "cpu":
        print(
            "Warning: training is running on CPU. A full 20-epoch CelebA run may take a long time.",
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
    best_metrics: Dict[str, float],
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
) -> None:
    print(f"\n{total_epochs} epochs completed in {total_duration_seconds / 3600:.3f} hours.", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    print("\nTraining complete", flush=True)
    print(f"Best checkpoint : {checkpoint_path}", flush=True)
    print(f"Selected ckpt  : {checkpoint_path}", flush=True)
    print(f"Best bal acc   : {summary_payload['best_validation_metrics']['balanced_accuracy']}", flush=True)
    print(f"Best f1        : {summary_payload['best_validation_metrics']['f1']}", flush=True)
    print(f"Report saved   : {run_dir / 'benchmark_summary.json'}", flush=True)


def train_phase2(
    config_path: str | Path,
    *,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    run_name: str = "phase2_a_b",
    tracker_backend: Optional[str] = None,
    epochs_override: Optional[int] = None,
    device_override: Optional[str] = None,
    max_batches: Optional[int] = None,
    checkpoint_name_override: Optional[str] = None,
) -> Dict[str, Any]:
    training_start = time.perf_counter()
    config = _as_str_key_mapping(load_config(config_path), context="config")
    phase2_cfg = _as_str_key_mapping(config["phase2"], context="config.phase2")
    if epochs_override is not None:
        phase2_cfg["epochs"] = int(epochs_override)
    effective_config = dict(config)
    effective_config["phase2"] = phase2_cfg
    _set_seed(int(phase2_cfg["seed"]))

    device = _resolve_device(device_override)
    model = DiscriminatorPhase2(dropout=float(phase2_cfg["dropout"]))

    paths_cfg = _as_str_key_mapping(config["paths"], context="config.paths")
    pretrained_path = Path(str(paths_cfg["checkpoints_dir"])) / str(phase2_cfg["pretrained_branch_a"])
    load_pretrained_branch_a(model, pretrained_path)
    model = model.to(device)

    optimizer = _build_optimizer(model, phase2_cfg)
    scheduler = _build_scheduler(optimizer, phase2_cfg)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = create_celeba_dataloader(config, split="train", limit=train_limit)
    val_loader = create_celeba_dataloader(config, split="val", shuffle=False, limit=val_limit)

    checkpoints_dir = Path(str(paths_cfg["checkpoints_dir"]))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        str(checkpoint_name_override)
        if checkpoint_name_override is not None
        else str(phase2_cfg["checkpoint_name"])
    )
    checkpoint_path = checkpoints_dir / checkpoint_name
    run_dir = Path(str(paths_cfg["runs_dir"])) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker: Optional[Tracker] = None
    if tracker_backend is not None:
        tracker = Tracker(run_dir / "tensorboard", backend=tracker_backend)

    history: list[EpochResult] = []
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None
    _print_run_header(
        run_name=run_name,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        phase2_cfg=phase2_cfg,
    )
    try:
        for epoch in range(1, int(phase2_cfg["epochs"]) + 1):
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                epoch=epoch,
                total_epochs=int(phase2_cfg["epochs"]),
                split_name="train",
                max_batches=max_batches,
            )
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                total_epochs=int(phase2_cfg["epochs"]),
                split_name="val",
                max_batches=max_batches,
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
                best_metrics = val_metrics
                best_epoch = epoch
                torch.save(
                    {
                        "phase": 2,
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
                total_epochs=int(phase2_cfg["epochs"]),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                current_lr=current_lr,
                best_epoch=best_epoch,
                best_metrics=cast(Dict[str, float], best_metrics),
            )
    finally:
        if tracker is not None:
            tracker.flush()
            tracker.close()

    if best_metrics is None:
        raise RuntimeError("Training completed without producing validation metrics")

    _serialize_history(run_dir / "metrics_history.json", history)
    summary_payload = _build_summary_payload(
        config=config,
        phase2_cfg=phase2_cfg,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        device=device,
        run_dir=run_dir,
    )
    _write_summary_files(run_dir, summary_payload)

    total_duration_seconds = time.perf_counter() - training_start
    _print_training_footer(
        total_epochs=int(phase2_cfg["epochs"]),
        total_duration_seconds=total_duration_seconds,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_payload=summary_payload,
    )
    return summary_payload


__all__ = ["train_phase2"]
