"""Helpers for stopping training when validation loss shows sustained overfitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverfitStopConfig:
    patience_overfit: int = 5
    patience_ceiling: int = 3
    warmup_epochs: int = 3
    val_loss_ceiling: float | None = None
    enable_loss_ceiling: bool = True


@dataclass(frozen=True)
class OverfitStopDecision:
    should_stop: bool
    reason: str | None
    overfit_streak: int
    ceiling_streak: int


@dataclass(frozen=True)
class ValMetricStopConfig:
    metric_name: str = "balanced_accuracy"
    patience: int = 4
    warmup_epochs: int = 3


@dataclass(frozen=True)
class ValMetricStopDecision:
    should_stop: bool
    reason: str | None
    best_value: float
    stale_streak: int


class OverfitStopMonitor:
    """Track train/val loss trends and emit a stop decision when overfitting persists."""

    def __init__(self, config: OverfitStopConfig) -> None:
        self.config = config
        self._previous_train_loss: float | None = None
        self._previous_val_loss: float | None = None
        self._overfit_streak = 0
        self._ceiling_streak = 0

    def update(self, *, epoch: int, train_loss: float, val_loss: float) -> OverfitStopDecision:
        train_improved = (
            self._previous_train_loss is not None and train_loss < self._previous_train_loss
        )
        val_worsened = self._previous_val_loss is not None and val_loss > self._previous_val_loss

        if train_improved and val_worsened:
            self._overfit_streak += 1
        else:
            self._overfit_streak = 0

        if (
            self.config.enable_loss_ceiling
            and self.config.val_loss_ceiling is not None
            and epoch > self.config.warmup_epochs
            and val_loss > self.config.val_loss_ceiling
            and train_loss < val_loss
        ):
            self._ceiling_streak += 1
        else:
            self._ceiling_streak = 0

        reason: str | None = None
        if self._overfit_streak >= self.config.patience_overfit:
            reason = (
                "Stopped early: validation loss worsened while train loss improved for "
                f"{self._overfit_streak} consecutive epochs."
            )
        elif self._ceiling_streak >= self.config.patience_ceiling:
            reason = (
                "Stopped early: validation loss exceeded the configured ceiling "
                f"({self.config.val_loss_ceiling:.3f}) for {self._ceiling_streak} consecutive epochs "
                "after warmup."
            )

        self._previous_train_loss = train_loss
        self._previous_val_loss = val_loss
        return OverfitStopDecision(
            should_stop=reason is not None,
            reason=reason,
            overfit_streak=self._overfit_streak,
            ceiling_streak=self._ceiling_streak,
        )


class ValMetricEarlyStop:
    """Stop when a validation metric fails to improve after the warmup period."""

    def __init__(self, config: ValMetricStopConfig) -> None:
        self.config = config
        self._best_value = float("-inf")
        self._stale_streak = 0

    def update(self, *, epoch: int, metric_value: float) -> ValMetricStopDecision:
        if epoch <= self.config.warmup_epochs:
            if metric_value > self._best_value:
                self._best_value = metric_value
            return ValMetricStopDecision(
                should_stop=False,
                reason=None,
                best_value=self._best_value,
                stale_streak=0,
            )

        if metric_value > self._best_value:
            self._best_value = metric_value
            self._stale_streak = 0
        else:
            self._stale_streak += 1

        reason: str | None = None
        if self._stale_streak >= self.config.patience:
            reason = (
                f"Stopped early: validation {self.config.metric_name} did not improve for "
                f"{self._stale_streak} consecutive epochs after warmup."
            )

        return ValMetricStopDecision(
            should_stop=reason is not None,
            reason=reason,
            best_value=self._best_value,
            stale_streak=self._stale_streak,
        )
