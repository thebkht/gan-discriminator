"""Training loop for the Week 3 Phase 4 end-to-end fine-tuning pass."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.celeba_loader import create_celeba_dataloader, load_config, verify_flow_cache
from evaluation import compute_binary_classification_metrics
from evaluation.inference_handoff import write_inference_contract
from models import DiscriminatorPhase4, load_phase3_into_phase4
from training.batch_preview import maybe_save_train_preview, maybe_save_val_previews
from training.checkpointing import CheckpointPayload, load_checkpoint, save_checkpoint
from training.losses import AsymmetricCombinedLoss
from training.overfit_stop import OverfitStopConfig, OverfitStopMonitor, ValMetricEarlyStop, ValMetricStopConfig
from training.run_artifacts import write_confusion_matrix_artifacts, write_results_plot
from training.tracker import Tracker
from training.trainer import (
    _as_float,
    _as_str_key_mapping,
    _format_duration,
    _format_epoch_prefix,
    _format_metric_value,
    _format_progress_bar,
    _format_progress_prefix,
    _print_epoch_result_row,
    _print_progress_header,
    _resolve_device,
    _set_seed,
    suppress_console_input,
)


@dataclass
class EpochResult:
    epoch: int
    stage: str
    stage_epoch: int
    split: str
    loss: float
    balanced_accuracy: float
    f1: float
    learning_rate: float
    duration_seconds: float


@dataclass(frozen=True)
class Phase4Stage:
    name: str
    planned_epochs: int
    branch_a_train_last_n: int
    branch_b_expander: bool
    branch_c: bool
    learning_rate: float


def _build_optimizer(model: DiscriminatorPhase4, phase4_cfg: Dict[str, Any]) -> Adam:
    beta_values = cast(list[object], phase4_cfg["betas"])
    if len(beta_values) != 2:
        raise ValueError("Adam betas must contain exactly two values")
    betas = (
        _as_float(beta_values[0], context="phase4.betas[0]"),
        _as_float(beta_values[1], context="phase4.betas[1]"),
    )
    return Adam(
        model.parameters(),
        lr=_as_float(phase4_cfg["learning_rate"], context="phase4.learning_rate"),
        betas=betas,
    )


def _build_scheduler(optimizer: Adam, phase4_cfg: Dict[str, Any]) -> CosineAnnealingLR:
    scheduler_name = str(phase4_cfg["scheduler"])
    if scheduler_name != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler for Phase 4: {scheduler_name}")
    return CosineAnnealingLR(optimizer, T_max=int(phase4_cfg["scheduler_t_max"]))


def _resolve_early_stopping(phase4_cfg: Dict[str, Any]) -> OverfitStopConfig:
    early_cfg = _as_str_key_mapping(phase4_cfg.get("early_stopping", {}), context="phase4.early_stopping")
    return OverfitStopConfig(
        patience_overfit=int(early_cfg.get("patience_overfit", 5)),
        patience_ceiling=int(early_cfg.get("patience_ceiling", 3)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 3)),
        val_loss_ceiling=_as_float(early_cfg.get("val_loss_ceiling", 0.45), context="phase4.early_stopping.val_loss_ceiling"),
        enable_loss_ceiling=bool(early_cfg.get("enable_loss_ceiling", True)),
    )


def _resolve_val_metric_early_stopping(phase4_cfg: Dict[str, Any]) -> ValMetricStopConfig:
    early_cfg = _as_str_key_mapping(phase4_cfg.get("early_stopping", {}), context="phase4.early_stopping")
    return ValMetricStopConfig(
        metric_name=str(early_cfg.get("val_metric", "balanced_accuracy")),
        patience=int(early_cfg.get("patience_val_metric", 4)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 3)),
    )


def _serialize_history(path: Path, history: list[EpochResult]) -> None:
    path.write_text(json.dumps([asdict(entry) for entry in history], indent=2), encoding="utf-8")


def _phase4_stage_sequence(phase4_cfg: Dict[str, Any], *, total_branch_a_blocks: int) -> list[Phase4Stage]:
    stage_cfg = _as_str_key_mapping(phase4_cfg.get("staged_unfreezing", {}), context="phase4.staged_unfreezing")
    base_lr = _as_float(phase4_cfg["learning_rate"], context="phase4.learning_rate")
    total_epochs = int(phase4_cfg["epochs"])
    if not bool(stage_cfg.get("enabled", True)):
        return [
            Phase4Stage(
                name="all_branches",
                planned_epochs=total_epochs,
                branch_a_train_last_n=total_branch_a_blocks,
                branch_b_expander=True,
                branch_c=True,
                learning_rate=base_lr,
            )
        ]

    fusion_epochs = int(stage_cfg.get("fusion_epochs", 10))
    branches_bc_epochs = int(stage_cfg.get("branches_bc_epochs", 10))
    branch_a_epochs = total_epochs - fusion_epochs - branches_bc_epochs
    branch_a_train_last_n = int(stage_cfg.get("branch_a_train_last_n", 2))
    if fusion_epochs <= 0 or branches_bc_epochs <= 0 or branch_a_epochs <= 0:
        raise ValueError(
            "Phase 4 staged_unfreezing requires positive stage lengths; "
            f"got fusion={fusion_epochs}, branches_bc={branches_bc_epochs}, branch_a={branch_a_epochs}"
        )
    return [
        Phase4Stage(
            name="fusion_only",
            planned_epochs=fusion_epochs,
            branch_a_train_last_n=0,
            branch_b_expander=False,
            branch_c=False,
            learning_rate=_as_float(stage_cfg.get("fusion_lr", base_lr), context="phase4.staged_unfreezing.fusion_lr"),
        ),
        Phase4Stage(
            name="branches_bc",
            planned_epochs=branches_bc_epochs,
            branch_a_train_last_n=0,
            branch_b_expander=True,
            branch_c=True,
            learning_rate=_as_float(stage_cfg.get("branches_bc_lr", base_lr), context="phase4.staged_unfreezing.branches_bc_lr"),
        ),
        Phase4Stage(
            name="branch_a_tail",
            planned_epochs=branch_a_epochs,
            branch_a_train_last_n=branch_a_train_last_n,
            branch_b_expander=True,
            branch_c=True,
            learning_rate=_as_float(stage_cfg.get("branch_a_lr", base_lr), context="phase4.staged_unfreezing.branch_a_lr"),
        ),
    ]


def _phase4_stage_for_epoch(epoch: int, phase4_cfg: Dict[str, Any], *, total_branch_a_blocks: int) -> Phase4Stage:
    remaining_epoch = epoch
    for stage in _phase4_stage_sequence(phase4_cfg, total_branch_a_blocks=total_branch_a_blocks):
        if remaining_epoch <= stage.planned_epochs:
            return stage
        remaining_epoch -= stage.planned_epochs
    raise ValueError(f"Epoch {epoch} exceeds configured Phase 4 epochs")


def _apply_phase4_stage(model: DiscriminatorPhase4, optimizer: Adam, stage: Phase4Stage) -> None:
    model.set_phase4_trainability(
        branch_a_train_last_n=stage.branch_a_train_last_n,
        branch_b_expander=stage.branch_b_expander,
        branch_c=stage.branch_c,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = stage.learning_rate


def _load_upstream_summary(runs_dir: Path, *, phase: int, preferred_runs: tuple[str, ...]) -> dict[str, Any]:
    for run_name in preferred_runs:
        path = runs_dir / run_name / "benchmark_summary.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if int(data.get("phase", -1)) == phase:
                return cast(dict[str, Any], data)
    for path in sorted(runs_dir.glob("*/benchmark_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("phase", -1)) == phase:
            return cast(dict[str, Any], data)
    return {"phase": phase, "status": "unknown", "best_validation_metrics": {}}


def _run_epoch(
    model: DiscriminatorPhase4,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optional[Adam] = None,
    epoch: int = 1,
    total_epochs: int = 1,
    split_name: str = "train",
    max_batches: Optional[int] = None,
    run_dir: Optional[Path] = None,
    include_predictions: bool = False,
) -> tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    start = time.perf_counter()

    num_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    _print_progress_header(split_name=split_name)
    progress_end = "" if torch.cuda.is_available() or torch.backends.mps.is_available() else "\n"
    for batch_index, batch in enumerate(dataloader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        frame_a = cast(torch.Tensor, batch["frame_a"]).to(device)
        frame_b = cast(torch.Tensor, batch["frame_b"]).to(device)
        flow = cast(torch.Tensor, batch["flow"]).to(device)
        labels = cast(torch.Tensor, batch["label"]).float().to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(frame_a, frame_b, flow)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        if run_dir is not None:
            preview_batch_index = batch_index - 1
            if is_train:
                maybe_save_train_preview(run_dir=run_dir, batch_index=preview_batch_index, frame_a=frame_a, labels=labels)
            else:
                maybe_save_val_previews(run_dir=run_dir, batch_index=preview_batch_index, frame_a=frame_a, labels=labels, logits=logits)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        running_loss = total_loss / total_examples
        elapsed_seconds = time.perf_counter() - start
        avg_batch_seconds = elapsed_seconds / batch_index
        eta_seconds = avg_batch_seconds * (num_batches - batch_index)
        batches_per_second = batch_index / elapsed_seconds if elapsed_seconds > 0 else 0.0
        progress_prefix = _format_progress_prefix(
            epoch=epoch,
            total_epochs=total_epochs,
            split_name=split_name,
            device=device,
            running_loss=running_loss,
            batch_size=batch_size,
            image_size=int(frame_a.shape[-1]),
        )
        percent_complete = (100.0 * batch_index / num_batches) if num_batches > 0 else 0.0
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


def _status_label(best_metrics: Dict[str, float], targets: Dict[str, float]) -> str:
    meets_bal_acc = best_metrics["balanced_accuracy"] >= targets["balanced_accuracy"]
    meets_f1 = best_metrics["f1"] >= targets["f1"]
    if meets_bal_acc and meets_f1:
        return "met"
    if meets_bal_acc or meets_f1:
        return "partially met"
    return "not met"


def train_phase4(
    config_path: str | Path,
    *,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    run_name: str = "phase4_ensemble",
    tracker_backend: Optional[str] = None,
    epochs_override: Optional[int] = None,
    device_override: Optional[str] = None,
    max_batches: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    checkpoint_name_override: Optional[str] = None,
    resume: Optional[str | Path] = None,
) -> Dict[str, Any]:
    training_start = time.perf_counter()
    config = _as_str_key_mapping(load_config(config_path), context="config")
    phase4_cfg = _as_str_key_mapping(config["phase4"], context="config.phase4")
    dataloader_cfg = dict(_as_str_key_mapping(config["dataloader"], context="config.dataloader"))
    dataloader_cfg.update(_as_str_key_mapping(phase4_cfg.get("dataloader", {}), context="config.phase4.dataloader"))
    if epochs_override is not None:
        phase4_cfg["epochs"] = int(epochs_override)
    if num_workers_override is not None:
        dataloader_cfg["num_workers"] = int(num_workers_override)
    effective_config = dict(config)
    effective_config["phase4"] = phase4_cfg
    _set_seed(int(phase4_cfg["seed"]))

    device = _resolve_device(device_override if device_override is not None else cast(Optional[str], phase4_cfg.get("device")))
    model = DiscriminatorPhase4(dropout=float(phase4_cfg["dropout"]))
    paths_cfg = _as_str_key_mapping(config["paths"], context="config.paths")
    checkpoints_dir = Path(str(paths_cfg["checkpoints_dir"]))
    runs_dir = Path(str(paths_cfg["runs_dir"]))
    phase3_path = checkpoints_dir / str(phase4_cfg["pretrained_phase3"])
    phase3_metadata = load_phase3_into_phase4(model, phase3_path)
    model = model.to(device)
    stage_sequence = _phase4_stage_sequence(phase4_cfg, total_branch_a_blocks=len(model.branch_a.features))
    initial_stage = stage_sequence[0]
    model.set_phase4_trainability(
        branch_a_train_last_n=initial_stage.branch_a_train_last_n,
        branch_b_expander=initial_stage.branch_b_expander,
        branch_c=initial_stage.branch_c,
    )

    optimizer = _build_optimizer(model, phase4_cfg)
    scheduler = _build_scheduler(optimizer, phase4_cfg)
    loss_cfg = _as_str_key_mapping(phase4_cfg["loss"], context="phase4.loss")
    criterion = AsymmetricCombinedLoss(
        bce_weight=_as_float(loss_cfg["bce_weight"], context="phase4.loss.bce_weight"),
        hinge_weight=_as_float(loss_cfg["hinge_weight"], context="phase4.loss.hinge_weight"),
        real_weight=_as_float(loss_cfg.get("real_weight", 1.5), context="phase4.loss.real_weight"),
        fake_weight=_as_float(loss_cfg.get("fake_weight", 1.0), context="phase4.loss.fake_weight"),
        margin=_as_float(loss_cfg.get("margin", 0.8), context="phase4.loss.margin"),
    )

    flow_cache_report = verify_flow_cache(paths_cfg["image_dir"], paths_cfg["flow_cache_dir"])
    missing_flow_count = flow_cache_report["missing_count"]
    extra_flow_count = flow_cache_report["extra_count"]
    if not isinstance(missing_flow_count, int) or not isinstance(extra_flow_count, int):
        raise TypeError("Flow cache report counts must be integers")
    if missing_flow_count != 0 or extra_flow_count != 0:
        raise RuntimeError("Flow cache verification failed for Phase 4")

    train_loader = create_celeba_dataloader(
        config,
        split="train",
        limit=train_limit,
        include_flow=True,
        pairing_mode="adjacent_cache",
        dataloader_overrides=dataloader_cfg,
    )
    val_loader = create_celeba_dataloader(
        config,
        split="val",
        shuffle=False,
        limit=val_limit,
        include_flow=True,
        pairing_mode="adjacent_cache",
        dataloader_overrides=dataloader_cfg,
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(checkpoint_name_override) if checkpoint_name_override is not None else str(phase4_cfg["checkpoint_name"])
    checkpoint_path = checkpoints_dir / checkpoint_name
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker: Optional[Tracker] = None
    if tracker_backend is not None:
        tracker = Tracker(run_dir / "tensorboard", backend=tracker_backend)

    history: list[EpochResult] = []
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None
    best_val_logits: Optional[np.ndarray] = None
    best_val_labels: Optional[np.ndarray] = None
    start_epoch = 1
    if resume is not None:
        payload = load_checkpoint(Path(resume), model, optimizer, scheduler)
        start_epoch = payload.epoch + 1
        best_metrics = payload.best_validation_metrics or None
        best_epoch = payload.epoch

    overfit_monitor = OverfitStopMonitor(_resolve_early_stopping(phase4_cfg))
    val_metric_monitor = ValMetricEarlyStop(_resolve_val_metric_early_stopping(phase4_cfg))
    stop_reason: Optional[str] = None
    stage_events: list[dict[str, Any]] = []
    targets = _as_str_key_mapping(phase4_cfg["targets"], context="phase4.targets")
    upstream_phase2 = _load_upstream_summary(runs_dir, phase=2, preferred_runs=("phase2_a_b_30ep", "phase2_a_b_run3", "phase2_a_b"))
    upstream_phase3 = _load_upstream_summary(runs_dir, phase=3, preferred_runs=("phase3_a_b_c_w2",))

    try:
        with suppress_console_input():
            stage_index = 0
            consumed_epochs = 0
            while stage_index < len(stage_sequence) and start_epoch > consumed_epochs + stage_sequence[stage_index].planned_epochs:
                consumed_epochs += stage_sequence[stage_index].planned_epochs
                stage_index += 1
            stage_epoch = max(1, start_epoch - consumed_epochs)

            for epoch in range(start_epoch, int(phase4_cfg["epochs"]) + 1):
                if stage_index >= len(stage_sequence):
                    break
                stage = stage_sequence[stage_index]
                _apply_phase4_stage(model, optimizer, stage)
                train_metrics, _, _ = _run_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_epochs=int(phase4_cfg["epochs"]),
                    split_name="train",
                    max_batches=max_batches,
                    run_dir=run_dir,
                )
                val_metrics, val_logits, val_labels = _run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    epoch=epoch,
                    total_epochs=int(phase4_cfg["epochs"]),
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
                            epoch,
                            stage.name,
                            stage_epoch,
                            "train",
                            train_metrics["loss"],
                            train_metrics["balanced_accuracy"],
                            train_metrics["f1"],
                            current_lr,
                            train_metrics["duration_seconds"],
                        ),
                        EpochResult(
                            epoch,
                            stage.name,
                            stage_epoch,
                            "val",
                            val_metrics["loss"],
                            val_metrics["balanced_accuracy"],
                            val_metrics["f1"],
                            current_lr,
                            val_metrics["duration_seconds"],
                        ),
                    ]
                )
                if tracker is not None:
                    for metric_name in ("loss", "balanced_accuracy", "f1"):
                        tracker.log_scalar(f"{metric_name}/train", train_metrics[metric_name], epoch)
                        tracker.log_scalar(f"{metric_name}/val", val_metrics[metric_name], epoch)
                    tracker.log_scalar("auc_roc/val", val_metrics["auc_roc"], epoch)
                    tracker.log_scalar("lr", current_lr, epoch)

                if (
                    best_metrics is None
                    or best_val_logits is None
                    or best_val_labels is None
                    or val_metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]
                ):
                    best_metrics = val_metrics
                    best_epoch = epoch
                    if val_logits is None or val_labels is None:
                        raise RuntimeError("Validation predictions were requested but not returned")
                    best_val_logits = val_logits.copy()
                    best_val_labels = val_labels.copy()
                    save_checkpoint(
                        checkpoint_path,
                        CheckpointPayload(
                            epoch=epoch,
                            phase=4,
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            best_validation_metrics={
                                "balanced_accuracy": val_metrics["balanced_accuracy"],
                                "f1": val_metrics["f1"],
                                "auc_roc": val_metrics["auc_roc"],
                                "loss": val_metrics["loss"],
                            },
                            config=effective_config,
                            metadata={
                                "fusion_contract": str(phase4_cfg["fusion_contract"]),
                                "fusion_dim": int(phase4_cfg["fusion_dim"]),
                                "stage": stage.name,
                                "stage_epoch": stage_epoch,
                                "branch_dims": _as_str_key_mapping(phase4_cfg["branch_dims"], context="phase4.branch_dims"),
                                "upstream": {
                                    "phase2_gate_met": str(upstream_phase2.get("status", "")).lower() == "met",
                                    "phase2_balanced_accuracy": float(cast(dict[str, Any], upstream_phase2.get("best_validation_metrics", {})).get("balanced_accuracy", float("nan"))),
                                    "phase3_balanced_accuracy": float(cast(dict[str, Any], upstream_phase3.get("best_validation_metrics", {})).get("balanced_accuracy", float("nan"))),
                                },
                                "phase3_metadata": phase3_metadata,
                            },
                        ),
                    )

                overfit_decision = overfit_monitor.update(epoch=stage_epoch, train_loss=train_metrics["loss"], val_loss=val_metrics["loss"])
                val_metric_decision = val_metric_monitor.update(epoch=stage_epoch, metric_value=val_metrics["balanced_accuracy"])
                if overfit_decision.should_stop:
                    stop_reason = overfit_decision.reason
                elif val_metric_decision.should_stop:
                    stop_reason = val_metric_decision.reason

                print(
                    (
                        f"\n{_format_epoch_prefix(epoch, int(phase4_cfg['epochs']), show_epoch=False)}   "
                        f"stage {stage.name} {stage_epoch:>2d}/{stage.planned_epochs:<2d}   "
                        f"train_loss {_format_metric_value(train_metrics['loss'], precision=6):>8}   "
                        f"val_loss {_format_metric_value(val_metrics['loss'], precision=6):>8}   "
                        f"bal_acc {_format_metric_value(val_metrics['balanced_accuracy'], precision=4):>6}   "
                        f"f1 {_format_metric_value(val_metrics['f1'], precision=4):>6}   "
                        f"lr {current_lr:>8.6f}   "
                        f"best {best_epoch:>2d}/{int(phase4_cfg['epochs']):<2d} "
                        f"{_format_metric_value(cast(Dict[str, float], best_metrics)['balanced_accuracy'], precision=4):>6}\n"
                    ),
                    flush=True,
                )
                if stop_reason is not None:
                    stage_events.append(
                        {
                            "stage": stage.name,
                            "start_epoch": epoch - stage_epoch + 1,
                            "end_epoch": epoch,
                            "planned_epochs": stage.planned_epochs,
                            "completed": False,
                            "reason": stop_reason,
                        }
                    )
                    if stage_index == len(stage_sequence) - 1:
                        print(stop_reason, flush=True)
                        break
                    print(f"{stop_reason} Advancing to next Phase 4 stage.", flush=True)
                    stage_index += 1
                    stage_epoch = 1
                    overfit_monitor = OverfitStopMonitor(_resolve_early_stopping(phase4_cfg))
                    val_metric_monitor = ValMetricEarlyStop(_resolve_val_metric_early_stopping(phase4_cfg))
                    stop_reason = None
                    continue

                if stage_epoch >= stage.planned_epochs:
                    stage_events.append(
                        {
                            "stage": stage.name,
                            "start_epoch": epoch - stage_epoch + 1,
                            "end_epoch": epoch,
                            "planned_epochs": stage.planned_epochs,
                            "completed": True,
                            "reason": None,
                        }
                    )
                    stage_index += 1
                    stage_epoch = 1
                    overfit_monitor = OverfitStopMonitor(_resolve_early_stopping(phase4_cfg))
                    val_metric_monitor = ValMetricEarlyStop(_resolve_val_metric_early_stopping(phase4_cfg))
                else:
                    stage_epoch += 1
    finally:
        if tracker is not None:
            tracker.flush()
            tracker.close()

    if best_metrics is None or best_val_logits is None or best_val_labels is None:
        raise RuntimeError("Training completed without a best validation checkpoint")

    _serialize_history(run_dir / "metrics_history.json", history)
    write_confusion_matrix_artifacts(run_dir, labels=best_val_labels, logits=best_val_logits, class_names=("real", "fake"))
    write_results_plot(run_dir, history)

    summary_payload = {
        "phase": 4,
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint_selection_rule": "highest validation balanced accuracy",
        "best_epoch": best_epoch,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "stage_events": stage_events,
        "stage_early_exits": [event for event in stage_events if not bool(event["completed"])],
        "fusion_contract": str(phase4_cfg["fusion_contract"]),
        "fusion_dim": int(phase4_cfg["fusion_dim"]),
        "best_validation_metrics": {
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "f1": best_metrics["f1"],
            "auc_roc": best_metrics["auc_roc"],
            "loss": best_metrics["loss"],
        },
        "targets": {
            "balanced_accuracy": float(targets["balanced_accuracy"]),
            "f1": float(targets["f1"]),
        },
        "status": _status_label(
            best_metrics,
            {"balanced_accuracy": float(targets["balanced_accuracy"]), "f1": float(targets["f1"])},
        ),
        "hyperparameters": {
            "epochs": int(phase4_cfg["epochs"]),
            "learning_rate": float(phase4_cfg["learning_rate"]),
            "batch_size": int(dataloader_cfg["batch_size"]),
            "scheduler": str(phase4_cfg["scheduler"]),
            "scheduler_t_max": int(phase4_cfg["scheduler_t_max"]),
            "checkpoint_metric": "balanced_accuracy",
            "loss": {
                "name": "AsymmetricCombinedLoss",
                "bce_weight": float(loss_cfg["bce_weight"]),
                "hinge_weight": float(loss_cfg["hinge_weight"]),
                "real_weight": float(loss_cfg.get("real_weight", 1.5)),
                "fake_weight": float(loss_cfg.get("fake_weight", 1.0)),
                "margin": float(loss_cfg.get("margin", 0.8)),
            },
            "staged_unfreezing": _as_str_key_mapping(
                phase4_cfg.get("staged_unfreezing", {}),
                context="phase4.staged_unfreezing",
            ),
        },
        "early_stopping_config": _as_str_key_mapping(phase4_cfg["early_stopping"], context="phase4.early_stopping"),
        "flow_cache": flow_cache_report,
        "upstream": {
            "phase2_gate_met": str(upstream_phase2.get("status", "")).lower() == "met",
            "phase2_balanced_accuracy": float(cast(dict[str, Any], upstream_phase2.get("best_validation_metrics", {})).get("balanced_accuracy", float("nan"))),
            "phase3_balanced_accuracy": float(cast(dict[str, Any], upstream_phase3.get("best_validation_metrics", {})).get("balanced_accuracy", float("nan"))),
        },
    }
    (run_dir / "benchmark_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (run_dir / "benchmark_summary.md").write_text(
        "\n".join(
            [
                "# Week 3 Phase 4 Benchmark Summary",
                "",
                f"- Phase: `{summary_payload['phase']}`",
                f"- Status: `{summary_payload['status']}`",
                f"- Fusion contract: `{summary_payload['fusion_contract']}`",
                f"- Best epoch: `{summary_payload['best_epoch']}`",
                f"- Best validation balanced accuracy: `{summary_payload['best_validation_metrics']['balanced_accuracy']:.4f}`",
                f"- Best validation F1: `{summary_payload['best_validation_metrics']['f1']:.4f}`",
                f"- Best validation AUC-ROC: `{summary_payload['best_validation_metrics']['auc_roc']:.4f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_inference_contract(
        run_dir,
        checkpoint_path=checkpoint_path,
        fusion_contract=str(phase4_cfg["fusion_contract"]),
        branch_dims={str(key): int(value) for key, value in _as_str_key_mapping(phase4_cfg["branch_dims"], context="phase4.branch_dims").items()},
        pairing_mode=str(phase4_cfg["pairing_mode"]),
        include_flow=bool(phase4_cfg["include_flow"]),
        recommended_combo="B+C",
        recommended_combo_gate_met=False,
    )

    print(f"\nResults saved to {run_dir}", flush=True)
    print(f"Best checkpoint : {checkpoint_path}", flush=True)
    print(f"Best bal acc   : {summary_payload['best_validation_metrics']['balanced_accuracy']}", flush=True)
    print(f"Best f1        : {summary_payload['best_validation_metrics']['f1']}", flush=True)
    print(f"Duration       : {_format_duration(time.perf_counter() - training_start)}", flush=True)
    return summary_payload


__all__ = ["train_phase4"]
