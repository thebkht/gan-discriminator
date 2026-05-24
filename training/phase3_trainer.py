"""Training loop for the Week 2 Phase 3 A+B+C discriminator."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.celeba_loader import _load_flow_tensor, create_celeba_dataloader, load_config, verify_flow_cache
from evaluation import compute_binary_classification_metrics
from models import DiscriminatorPhase3, load_phase2_into_phase3
from training.batch_preview import maybe_save_train_preview, maybe_save_val_previews
from training.checkpointing import CheckpointPayload, load_checkpoint, save_checkpoint
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
    suppress_console_input,
)


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


def _build_optimizer(model: DiscriminatorPhase3, phase3_cfg: Dict[str, Any]) -> Adam:
    beta_values = cast(list[object], phase3_cfg["betas"])
    if len(beta_values) != 2:
        raise ValueError("Adam betas must contain exactly two values")
    betas = (
        _as_float(beta_values[0], context="phase3.betas[0]"),
        _as_float(beta_values[1], context="phase3.betas[1]"),
    )
    params = list(model.branch_c.parameters()) + list(model.fusion.parameters())
    return Adam(
        params,
        lr=_as_float(phase3_cfg["learning_rate"], context="phase3.learning_rate"),
        betas=betas,
    )


def _build_scheduler(optimizer: Adam, phase3_cfg: Dict[str, Any]) -> CosineAnnealingLR:
    scheduler_name = str(phase3_cfg["scheduler"])
    if scheduler_name != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler for Phase 3: {scheduler_name}")
    return CosineAnnealingLR(optimizer, T_max=int(phase3_cfg["scheduler_t_max"]))


def _serialize_history(path: Path, history: list[EpochResult]) -> None:
    path.write_text(json.dumps([asdict(entry) for entry in history], indent=2), encoding="utf-8")


def _parameter_checksums(module: nn.Module, prefixes: tuple[str, ...]) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for name, tensor in module.state_dict().items():
        if name.startswith(prefixes):
            checksums[name] = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
    return checksums


def _detect_flow_load_method(sample_path: Path) -> str:
    try:
        torch.load(sample_path, map_location="cpu", weights_only=True)
        return "weights_only"
    except Exception:
        _load_flow_tensor(sample_path)
        return "fallback"


def _as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected int value for {context}, got {type(value).__name__}")
    return value


def _run_epoch(
    model: DiscriminatorPhase3,
    dataloader: DataLoader,
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
    progress_end = "" if sys.stdout.isatty() else "\n"
    _print_progress_header(split_name=split_name)
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


def _build_summary_payload(
    *,
    config: Dict[str, Any],
    phase3_cfg: Dict[str, Any],
    dataloader_cfg: Dict[str, Any],
    best_metrics: Dict[str, float],
    best_epoch: int,
    device: torch.device,
    run_dir: Path,
    stop_reason: Optional[str],
    flow_cache_report: Dict[str, object],
    flow_load_method: str,
    ab_checksums: Dict[str, str],
) -> Dict[str, Any]:
    targets = _as_str_key_mapping(phase3_cfg["targets"], context="phase3.targets")
    return {
        "phase": 3,
        "proxy_task": "adjacent_cache_identity_match",
        "run_dir": str(run_dir),
        "device": str(device),
        "checkpoint_selection_rule": "highest validation balanced accuracy",
        "best_epoch": best_epoch,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "fusion_reinitialized": True,
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
            {
                "balanced_accuracy": float(targets["balanced_accuracy"]),
                "f1": float(targets["f1"]),
            },
        ),
        "hyperparameters": {
            "epochs": int(phase3_cfg["epochs"]),
            "learning_rate": float(phase3_cfg["learning_rate"]),
            "batch_size": int(config["dataloader"]["batch_size"]),
            "scheduler": str(phase3_cfg["scheduler"]),
            "scheduler_t_max": int(phase3_cfg["scheduler_t_max"]),
            "checkpoint_metric": "balanced_accuracy",
        },
        "dataloader": {
            "num_workers": int(dataloader_cfg["num_workers"]),
            **(
                {"prefetch_factor": int(dataloader_cfg["prefetch_factor"])}
                if "prefetch_factor" in dataloader_cfg and int(dataloader_cfg["num_workers"]) > 0
                else {}
            ),
            "note": "bump num_workers if Phase 4 becomes IO-bound",
        },
        "flow_cache": flow_cache_report,
        "flow_cache_load": {
            "torch_version": torch.__version__,
            "load_method": flow_load_method,
        },
        "branch_ab_checksums": ab_checksums,
        "limitations": [
            "Phase 3 proxy differs from Phase 2 cross-identity proxy; metrics are not directly comparable.",
            "Branch A and Branch B stay frozen during Phase 3 while only Branch C features and fusion are optimized.",
            "Flow tensors are loaded from trusted local cache files keyed to adjacent-index partners.",
            "No out-of-domain benchmark is included in Week 2.",
        ],
    }


def _write_summary_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    (run_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "benchmark_summary.md").write_text(
        "\n".join(
            [
                "# Week 2 Phase 3 A+B+C Benchmark Summary",
                "",
                f"- Phase: `{payload['phase']}`",
                f"- Status: `{payload['status']}`",
                f"- Device: `{payload['device']}`",
                f"- Proxy task: `{payload['proxy_task']}`",
                f"- Best epoch: `{payload['best_epoch']}`",
                f"- Best validation balanced accuracy: `{payload['best_validation_metrics']['balanced_accuracy']:.4f}`",
                f"- Best validation F1: `{payload['best_validation_metrics']['f1']:.4f}`",
                f"- Best validation AUC-ROC: `{payload['best_validation_metrics']['auc_roc']:.4f}`",
                f"- Flow cache load path: `{payload['flow_cache_load']['load_method']}`",
                "",
                "## Limitations",
                "",
                *[f"- {item}" for item in payload["limitations"]],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def train_phase3(
    config_path: str | Path,
    *,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    run_name: str = "phase3_a_b_c",
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
    phase3_cfg = _as_str_key_mapping(config["phase3"], context="config.phase3")
    phase3_dataloader_cfg = dict(_as_str_key_mapping(config["dataloader"], context="config.dataloader"))
    phase3_dataloader_cfg.update(
        _as_str_key_mapping(phase3_cfg.get("dataloader", {}), context="config.phase3.dataloader")
    )
    if epochs_override is not None:
        phase3_cfg["epochs"] = int(epochs_override)
    if num_workers_override is not None:
        phase3_dataloader_cfg["num_workers"] = int(num_workers_override)
    effective_config = dict(config)
    effective_config["phase3"] = phase3_cfg
    _set_seed(int(phase3_cfg["seed"]))
    resolved_device_override = device_override if device_override is not None else cast(Optional[str], phase3_cfg.get("device"))
    device = _resolve_device(resolved_device_override)
    model = DiscriminatorPhase3(dropout=float(phase3_cfg["dropout"]))
    paths_cfg = _as_str_key_mapping(config["paths"], context="config.paths")
    phase2_path = Path(str(paths_cfg["checkpoints_dir"])) / str(phase3_cfg["pretrained_phase2"])
    load_phase2_into_phase3(model, phase2_path)
    ab_checksums_before = _parameter_checksums(model, ("branch_a.", "branch_b."))
    model = model.to(device)

    optimizer = _build_optimizer(model, phase3_cfg)
    scheduler = _build_scheduler(optimizer, phase3_cfg)
    criterion = nn.BCEWithLogitsLoss()

    flow_cache_report = verify_flow_cache(paths_cfg["image_dir"], paths_cfg["flow_cache_dir"])
    missing_count = _as_int(flow_cache_report["missing_count"], context="flow_cache_report.missing_count")
    extra_count = _as_int(flow_cache_report["extra_count"], context="flow_cache_report.extra_count")
    if missing_count != 0 or extra_count != 0:
        raise RuntimeError(
            f"Flow cache verification failed: missing={missing_count} extra={extra_count}"
        )
    flow_sample_path = Path(str(paths_cfg["flow_cache_dir"])) / "000001_flow.pt"
    if not flow_sample_path.exists():
        first_stem = sorted(Path(str(paths_cfg["flow_cache_dir"])).glob("*_flow.pt"))
        if not first_stem:
            raise FileNotFoundError("No cached flow tensors found for Phase 3 pre-flight")
        flow_sample_path = first_stem[0]
    flow_load_method = _detect_flow_load_method(flow_sample_path)

    train_loader = create_celeba_dataloader(
        config,
        split="train",
        limit=train_limit,
        include_flow=True,
        pairing_mode="adjacent_cache",
        dataloader_overrides=phase3_dataloader_cfg,
    )
    val_loader = create_celeba_dataloader(
        config,
        split="val",
        shuffle=False,
        limit=val_limit,
        include_flow=True,
        pairing_mode="adjacent_cache",
        dataloader_overrides=phase3_dataloader_cfg,
    )

    checkpoints_dir = Path(str(paths_cfg["checkpoints_dir"]))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(checkpoint_name_override) if checkpoint_name_override is not None else str(phase3_cfg["checkpoint_name"])
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
    start_epoch = 1
    if resume is not None:
        payload = load_checkpoint(Path(resume), model, optimizer, scheduler)
        start_epoch = payload.epoch + 1
        best_metrics = payload.best_validation_metrics or None
        best_epoch = payload.epoch

    try:
        with suppress_console_input():
            for epoch in range(start_epoch, int(phase3_cfg["epochs"]) + 1):
                train_metrics, _, _ = _run_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_epochs=int(phase3_cfg["epochs"]),
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
                    total_epochs=int(phase3_cfg["epochs"]),
                    split_name="val",
                    max_batches=max_batches,
                    run_dir=run_dir,
                    include_predictions=True,
                )
                current_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step()
                history.extend(
                    [
                        EpochResult(epoch, "train", train_metrics["loss"], train_metrics["balanced_accuracy"], train_metrics["f1"], current_lr, train_metrics["duration_seconds"]),
                        EpochResult(epoch, "val", val_metrics["loss"], val_metrics["balanced_accuracy"], val_metrics["f1"], current_lr, val_metrics["duration_seconds"]),
                    ]
                )
                if tracker is not None:
                    tracker.log_scalar("loss/train", train_metrics["loss"], epoch)
                    tracker.log_scalar("loss/val", val_metrics["loss"], epoch)
                    tracker.log_scalar("balanced_accuracy/train", train_metrics["balanced_accuracy"], epoch)
                    tracker.log_scalar("balanced_accuracy/val", val_metrics["balanced_accuracy"], epoch)
                    tracker.log_scalar("f1/train", train_metrics["f1"], epoch)
                    tracker.log_scalar("f1/val", val_metrics["f1"], epoch)
                    tracker.log_scalar("auc_roc/val", val_metrics["auc_roc"], epoch)
                    tracker.log_scalar("lr", current_lr, epoch)

                should_replace = (
                    best_metrics is None
                    or best_val_logits is None
                    or best_val_labels is None
                    or val_metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]
                )
                if should_replace:
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
                            phase=3,
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
                        ),
                    )
                print(
                    (
                        f"\n{_format_epoch_prefix(epoch, int(phase3_cfg['epochs']), show_epoch=False)}   "
                        f"train_loss {_format_metric_value(train_metrics['loss'], precision=6):>8}   "
                        f"val_loss {_format_metric_value(val_metrics['loss'], precision=6):>8}   "
                        f"bal_acc {_format_metric_value(val_metrics['balanced_accuracy'], precision=4):>6}   "
                        f"f1 {_format_metric_value(val_metrics['f1'], precision=4):>6}   "
                        f"lr {current_lr:>8.6f}   "
                        f"best {best_epoch:>2d}/{int(phase3_cfg['epochs']):<2d} "
                        f"{_format_metric_value(cast(Dict[str, float], best_metrics)['balanced_accuracy'], precision=4):>6}\n"
                    ),
                    flush=True,
                )
    finally:
        if tracker is not None:
            tracker.flush()
            tracker.close()

    if best_metrics is None or best_val_logits is None or best_val_labels is None:
        raise RuntimeError("Training completed without a best validation checkpoint")
    ab_checksums_after = _parameter_checksums(model, ("branch_a.", "branch_b."))
    if ab_checksums_before != ab_checksums_after:
        raise AssertionError("Branch A+B checksums changed during Phase 3 training")

    _serialize_history(run_dir / "metrics_history.json", history)
    write_confusion_matrix_artifacts(run_dir, labels=best_val_labels, logits=best_val_logits, class_names=("real", "fake"))
    write_results_plot(run_dir, history)
    summary_payload = _build_summary_payload(
        config=config,
        phase3_cfg=phase3_cfg,
        dataloader_cfg=phase3_dataloader_cfg,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
        device=device,
        run_dir=run_dir,
        stop_reason=None,
        flow_cache_report=flow_cache_report,
        flow_load_method=flow_load_method,
        ab_checksums=ab_checksums_after,
    )
    _write_summary_files(run_dir, summary_payload)
    print(f"\nResults saved to {run_dir}", flush=True)
    print(f"Best checkpoint : {checkpoint_path}", flush=True)
    print(f"Best bal acc   : {summary_payload['best_validation_metrics']['balanced_accuracy']}", flush=True)
    print(f"Best f1        : {summary_payload['best_validation_metrics']['f1']}", flush=True)
    print(f"Duration       : {_format_duration(time.perf_counter() - training_start)}", flush=True)
    return summary_payload


__all__ = ["train_phase3"]
