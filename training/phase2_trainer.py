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
from models.branch_b import SIGN_CONSISTENCY_IDX
from training.trainer import (
    _as_float,
    _as_str_key_mapping,
    _format_duration,
    _format_epoch_prefix,
    _format_metric_value,
    _format_progress_bar,
    _format_progress_prefix,
    _resolve_device,
    _print_epoch_result_row,
    _print_progress_header,
    suppress_console_input,
)
from training.batch_preview import maybe_save_train_preview, maybe_save_val_previews
from training.overfit_stop import (
    OverfitStopConfig,
    OverfitStopMonitor,
    ValMetricEarlyStop,
    ValMetricStopConfig,
)
from training.run_artifacts import write_confusion_matrix_artifacts, write_results_plot
from training.tracker import Tracker


PHASE2_VAL_LOSS_CEILING = 0.40


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
    base_lr = _as_float(phase2_cfg["learning_rate"], context="phase2.learning_rate")
    tail_lr = base_lr * _as_float(
        phase2_cfg.get("backbone_lr_scale", 0.1),
        context="phase2.backbone_lr_scale",
    )
    train_last_n = int(phase2_cfg.get("backbone_train_last_n", 0))
    param_groups: list[dict[str, Any]] = [
        {"params": list(model.branch_b.expander.parameters()), "lr": base_lr},
        {"params": list(model.fusion.parameters()), "lr": base_lr},
    ]
    if train_last_n > 0:
        start_index = len(model.branch_a.features) - train_last_n
        for block_index in range(start_index, len(model.branch_a.features)):
            param_groups.append(
                {"params": list(model.branch_a.features[block_index].parameters()), "lr": tail_lr}
            )
    return Adam(
        param_groups,
        lr=base_lr,
        betas=betas,
        weight_decay=_as_float(phase2_cfg.get("weight_decay", 0.0), context="phase2.weight_decay"),
    )


def _build_scheduler(optimizer: Adam, phase2_cfg: Dict[str, Any]) -> CosineAnnealingLR:
    scheduler_name = str(phase2_cfg["scheduler"])
    if scheduler_name != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler for Phase 2: {scheduler_name}")
    return CosineAnnealingLR(optimizer, T_max=int(phase2_cfg["scheduler_t_max"]))


def _resolve_early_stopping(phase2_cfg: Dict[str, Any]) -> OverfitStopConfig:
    raw = phase2_cfg.get("early_stopping")
    if raw is None:
        return OverfitStopConfig(val_loss_ceiling=PHASE2_VAL_LOSS_CEILING)
    early_cfg = _as_str_key_mapping(raw, context="phase2.early_stopping")
    return OverfitStopConfig(
        patience_overfit=int(early_cfg.get("patience_overfit", 5)),
        patience_ceiling=int(early_cfg.get("patience_ceiling", 3)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 3)),
        val_loss_ceiling=_as_float(
            early_cfg.get("val_loss_ceiling", PHASE2_VAL_LOSS_CEILING),
            context="phase2.early_stopping.val_loss_ceiling",
        ),
        enable_loss_ceiling=bool(early_cfg.get("enable_loss_ceiling", True)),
    )


def _resolve_val_metric_early_stopping(phase2_cfg: Dict[str, Any]) -> ValMetricStopConfig:
    raw = phase2_cfg.get("early_stopping")
    if raw is None:
        return ValMetricStopConfig()
    early_cfg = _as_str_key_mapping(raw, context="phase2.early_stopping")
    return ValMetricStopConfig(
        metric_name=str(early_cfg.get("val_metric", "balanced_accuracy")),
        patience=int(early_cfg.get("patience_val_metric", 4)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 3)),
    )


def _branch_a_batch_norm(model: DiscriminatorPhase2, block_index: int) -> nn.BatchNorm2d:
    block = cast(nn.Sequential, model.branch_a.features[block_index])
    batch_norm = block[1]
    if not isinstance(batch_norm, nn.BatchNorm2d):
        raise TypeError(f"Expected BatchNorm2d in branch_a.features[{block_index}][1]")
    return batch_norm


def _snapshot_bn_stats(model: DiscriminatorPhase2) -> dict[str, torch.Tensor]:
    frozen_bn = _branch_a_batch_norm(model, 1)
    tail_bn = _branch_a_batch_norm(model, 3)
    if frozen_bn.running_mean is None or frozen_bn.running_var is None:
        raise RuntimeError("Frozen Branch A batch norm buffers are not initialized")
    if tail_bn.running_mean is None or tail_bn.running_var is None:
        raise RuntimeError("Tail Branch A batch norm buffers are not initialized")
    return {
        "frozen_mean": frozen_bn.running_mean.detach().clone(),
        "frozen_var": frozen_bn.running_var.detach().clone(),
        "tail_mean": tail_bn.running_mean.detach().clone(),
        "tail_var": tail_bn.running_var.detach().clone(),
    }


def _print_bn_drift(model: DiscriminatorPhase2, baseline: dict[str, torch.Tensor]) -> None:
    frozen_bn = _branch_a_batch_norm(model, 1)
    tail_bn = _branch_a_batch_norm(model, 3)
    if frozen_bn.running_mean is None or frozen_bn.running_var is None:
        raise RuntimeError("Frozen Branch A batch norm buffers are not initialized")
    if tail_bn.running_mean is None or tail_bn.running_var is None:
        raise RuntimeError("Tail Branch A batch norm buffers are not initialized")
    frozen_mean_delta = (frozen_bn.running_mean - baseline["frozen_mean"]).abs().max().item()
    frozen_var_delta = (frozen_bn.running_var - baseline["frozen_var"]).abs().max().item()
    tail_mean_delta = (tail_bn.running_mean - baseline["tail_mean"]).abs().max().item()
    tail_var_delta = (tail_bn.running_var - baseline["tail_var"]).abs().max().item()
    print(
        (
            "BN drift after epoch 1 "
            f"frozen_mean_max={frozen_mean_delta:.6f} "
            f"frozen_var_max={frozen_var_delta:.6f} "
            f"tail_mean_max={tail_mean_delta:.6f} "
            f"tail_var_max={tail_var_delta:.6f}"
        ),
        flush=True,
    )


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
    run_dir: Optional[Path] = None,
    include_predictions: bool = False,
    diagnostics_model: Optional[DiscriminatorPhase2] = None,
    emit_branch_b_diagnostics: bool = False,
    branch_b_diagnostic_batches: int = 1,
) -> tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    branch_b_diag_rows: list[tuple[int, float, float, float]] = []
    start = time.perf_counter()

    num_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    progress_end = "" if sys.stdout.isatty() else "\n"
    _print_progress_header(split_name=split_name)
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

        if (
            emit_branch_b_diagnostics
            and batch_index <= branch_b_diagnostic_batches
            and diagnostics_model is not None
        ):
            with torch.no_grad():
                summary = diagnostics_model.branch_b._summary_features(frame_a, frame_b)
                feat_b = diagnostics_model.branch_b(frame_a, frame_b)
            sign_std = summary[:, SIGN_CONSISTENCY_IDX].std(unbiased=False).item()
            feat_std = feat_b.std(unbiased=False).item()
            branch_b_diag_rows.append((batch_index, feat_b.mean().item(), feat_std, sign_std))
            if feat_std < 0.01:
                print(
                    f"Warning: Branch B feature std is below 0.01 on diagnostic batch {batch_index}.",
                    flush=True,
                )
            if sign_std < 1e-4:
                print(
                    f"Warning: sign_consistency appears nearly constant on diagnostic batch {batch_index}.",
                    flush=True,
                )

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

    if branch_b_diag_rows:
        summary_text = " | ".join(
            (
                f"b{batch_index}: mean={feat_mean:.4f} std={feat_std:.4f} sign_std={sign_std:.4f}"
                for batch_index, feat_mean, feat_std, sign_std in branch_b_diag_rows
            )
        )
        print(f"Branch B diagnostics ({split_name}): {summary_text}", flush=True)

    if total_examples == 0:
        raise ValueError("Received an empty dataloader split; cannot compute metrics")

    average_loss = total_loss / total_examples
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(np.int64)
    metrics = compute_binary_classification_metrics(logits=logits, labels=labels, average_loss=average_loss)
    _print_epoch_result_row(
        split_name="all" if split_name.lower() == "val" else split_name,
        loss=float(metrics["loss"]),
        balanced_accuracy=float(metrics["balanced_accuracy"]),
        f1=float(metrics["f1"]),
    )
    metrics["duration_seconds"] = time.perf_counter() - start
    metrics["num_batches"] = float(num_batches)
    if include_predictions:
        return metrics, logits, labels
    return metrics, None, None


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
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    targets = _as_str_key_mapping(phase2_cfg["targets"], context="phase2.targets")
    return {
        "phase": 2,
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
            "weight_decay": float(phase2_cfg.get("weight_decay", 0.0)),
            "batch_size": int(config["dataloader"]["batch_size"]),
            "scheduler": str(phase2_cfg["scheduler"]),
            "scheduler_t_max": int(phase2_cfg["scheduler_t_max"]),
            "checkpoint_metric": str(phase2_cfg["checkpoint_metric"]),
        },
        "limitations": [
            "Branch A's early blocks remain frozen while only the shared encoder tail is finetuned through Branch B.",
            "Fake pairs are cross-identity proxy negatives or distant-index fallbacks, not actual deepfakes.",
            "This benchmark is still a proxy task and likely optimistic relative to real deepfake detection.",
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
                f"- Weight decay: `{payload['hyperparameters']['weight_decay']}`",
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
    completed_epochs: int,
) -> None:
    print(f"\n{completed_epochs}/{total_epochs} epochs completed in {total_duration_seconds / 3600:.3f} hours.", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    print("\nTraining complete", flush=True)
    print(f"Best checkpoint : {checkpoint_path}", flush=True)
    print(f"Selected ckpt  : {checkpoint_path}", flush=True)
    print(f"Best bal acc   : {summary_payload['best_validation_metrics']['balanced_accuracy']}", flush=True)
    print(f"Best f1        : {summary_payload['best_validation_metrics']['f1']}", flush=True)
    if summary_payload["stop_reason"]:
        print(f"Stop reason    : {summary_payload['stop_reason']}", flush=True)
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
    model = DiscriminatorPhase2(
        dropout=float(phase2_cfg["dropout"]),
        backbone_train_last_n=int(phase2_cfg.get("backbone_train_last_n", 0)),
    )

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
    best_val_logits: Optional[np.ndarray] = None
    best_val_labels: Optional[np.ndarray] = None
    stop_reason: Optional[str] = None
    completed_epochs = 0
    overfit_monitor = OverfitStopMonitor(_resolve_early_stopping(phase2_cfg))
    val_metric_monitor = ValMetricEarlyStop(_resolve_val_metric_early_stopping(phase2_cfg))
    bn_epoch1_baseline = _snapshot_bn_stats(model)
    _print_run_header(
        run_name=run_name,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        phase2_cfg=phase2_cfg,
    )
    try:
        with suppress_console_input():
            for epoch in range(1, int(phase2_cfg["epochs"]) + 1):
                train_metrics, _, _ = _run_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_epochs=int(phase2_cfg["epochs"]),
                    split_name="train",
                    max_batches=max_batches,
                    run_dir=run_dir,
                    diagnostics_model=model,
                    emit_branch_b_diagnostics=epoch == 1,
                    branch_b_diagnostic_batches=5,
                )
                val_metrics, val_logits, val_labels = _run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    epoch=epoch,
                    total_epochs=int(phase2_cfg["epochs"]),
                    split_name="val",
                    max_batches=max_batches,
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
                    best_metrics = val_metrics
                    best_epoch = epoch
                    if val_logits is None or val_labels is None:
                        raise RuntimeError("Validation predictions were requested but not returned")
                    best_val_logits = val_logits.copy()
                    best_val_labels = val_labels.copy()
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
                completed_epochs = epoch
                if epoch == 1:
                    _print_bn_drift(model, bn_epoch1_baseline)
                stop_decision = overfit_monitor.update(
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                )
                metric_name = val_metric_monitor.config.metric_name
                if metric_name not in val_metrics:
                    raise KeyError(f"Validation metrics do not contain '{metric_name}'")
                metric_stop_decision = val_metric_monitor.update(
                    epoch=epoch,
                    metric_value=float(val_metrics[metric_name]),
                )
                if stop_decision.should_stop:
                    stop_reason = stop_decision.reason
                    print(stop_reason, flush=True)
                    break
                if metric_stop_decision.should_stop:
                    stop_reason = metric_stop_decision.reason
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
        config=config,
        phase2_cfg=phase2_cfg,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        device=device,
        run_dir=run_dir,
        stop_reason=stop_reason,
    )
    _write_summary_files(run_dir, summary_payload)

    total_duration_seconds = time.perf_counter() - training_start
    _print_training_footer(
        total_epochs=int(phase2_cfg["epochs"]),
        total_duration_seconds=total_duration_seconds,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_payload=summary_payload,
        completed_epochs=completed_epochs,
    )
    return summary_payload


__all__ = ["train_phase2"]
