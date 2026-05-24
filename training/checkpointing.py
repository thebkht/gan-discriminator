"""Checkpoint helpers shared across training phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import torch


class _SchedulerWithStateDict(Protocol):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


@dataclass
class CheckpointPayload:
    epoch: int
    phase: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]]
    scheduler_state_dict: Optional[Dict[str, Any]]
    best_validation_metrics: Dict[str, float]
    config: Optional[Dict[str, Any]] = None


def save_checkpoint(path: str | Path, payload: CheckpointPayload) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": payload.epoch,
            "phase": payload.phase,
            "model_state_dict": payload.model_state_dict,
            "optimizer_state_dict": payload.optimizer_state_dict,
            "scheduler_state_dict": payload.scheduler_state_dict,
            "best_validation_metrics": payload.best_validation_metrics,
            "config": payload.config,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_SchedulerWithStateDict] = None,
) -> CheckpointPayload:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint load result: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return CheckpointPayload(
        epoch=int(checkpoint["epoch"]),
        phase=int(checkpoint["phase"]),
        model_state_dict=state_dict,
        optimizer_state_dict=optimizer_state,
        scheduler_state_dict=scheduler_state,
        best_validation_metrics=dict(checkpoint.get("best_validation_metrics", {})),
        config=checkpoint.get("config"),
    )


__all__ = ["CheckpointPayload", "load_checkpoint", "save_checkpoint"]
