from __future__ import annotations

import unittest

from training.overfit_stop import (
    OverfitStopConfig,
    OverfitStopMonitor,
    ValMetricEarlyStop,
    ValMetricStopConfig,
)


class OverfitStopMonitorTestCase(unittest.TestCase):
    def test_stops_after_sustained_overfit_trend(self) -> None:
        monitor = OverfitStopMonitor(
            OverfitStopConfig(
                patience_overfit=3,
                patience_ceiling=99,
                warmup_epochs=0,
                val_loss_ceiling=None,
                enable_loss_ceiling=False,
            )
        )

        history = [
            (0.50, 0.50),
            (0.40, 0.55),
            (0.35, 0.60),
            (0.30, 0.65),
        ]
        decisions = [
            monitor.update(epoch=idx, train_loss=train_loss, val_loss=val_loss)
            for idx, (train_loss, val_loss) in enumerate(history, start=1)
        ]

        self.assertFalse(decisions[1].should_stop)
        self.assertTrue(decisions[-1].should_stop)
        self.assertIn("validation loss worsened", decisions[-1].reason or "")

    def test_stops_after_ceiling_breach_post_warmup(self) -> None:
        monitor = OverfitStopMonitor(
            OverfitStopConfig(
                patience_overfit=99,
                patience_ceiling=2,
                warmup_epochs=1,
                val_loss_ceiling=0.35,
                enable_loss_ceiling=True,
            )
        )

        decisions = [
            monitor.update(epoch=1, train_loss=0.50, val_loss=0.36),
            monitor.update(epoch=2, train_loss=0.30, val_loss=0.40),
            monitor.update(epoch=3, train_loss=0.25, val_loss=0.41),
        ]

        self.assertFalse(decisions[0].should_stop)
        self.assertFalse(decisions[1].should_stop)
        self.assertTrue(decisions[2].should_stop)
        self.assertIn("configured ceiling", decisions[2].reason or "")

    def test_val_metric_early_stop_after_warmup(self) -> None:
        monitor = ValMetricEarlyStop(
            ValMetricStopConfig(metric_name="balanced_accuracy", patience=2, warmup_epochs=1)
        )

        decisions = [
            monitor.update(epoch=1, metric_value=0.70),
            monitor.update(epoch=2, metric_value=0.72),
            monitor.update(epoch=3, metric_value=0.71),
            monitor.update(epoch=4, metric_value=0.71),
        ]

        self.assertFalse(decisions[1].should_stop)
        self.assertFalse(decisions[2].should_stop)
        self.assertTrue(decisions[3].should_stop)
        self.assertIn("did not improve", decisions[3].reason or "")
