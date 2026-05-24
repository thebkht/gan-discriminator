"""Loss modules shared across later training phases."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HingeLoss(nn.Module):
    """
    Master-plan §8 GAN hinge on discriminator logits.

    Convention (pinned — do not invert in Phase 4):
      - real samples (dataset label 0): loss = max(0, 1 - logit)
      - fake samples (dataset label 1): loss = max(0, 1 + logit)

    Dataset labels remain 0=real, 1=fake (same as BCEWithLogitsLoss).
    """

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.shape != labels.shape:
            raise ValueError(
                f"HingeLoss expects logits and labels with identical shape, got {tuple(logits.shape)} and {tuple(labels.shape)}"
            )
        labels_long = labels.long()
        real_loss = torch.relu(1.0 - logits)
        fake_loss = torch.relu(1.0 + logits)
        loss = torch.where(labels_long == 0, real_loss, fake_loss)
        return loss.mean()


__all__ = ["HingeLoss"]
