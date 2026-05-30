"""Test-time augmentation helpers for forensics inference."""

from __future__ import annotations

import torch
from torch import Tensor


def horizontal_flip_batch(tensor: Tensor) -> Tensor:
    """Flip a BCHW batch horizontally."""
    return torch.flip(tensor, dims=(-1,))


def predict_probabilities_with_flip_tta(
    model: torch.nn.Module,
    frame_a: Tensor,
    frame_b: Tensor,
    flow: Tensor,
    *,
    degenerate_pairing: bool,
) -> Tensor:
    """Average original and horizontal-flip sigmoid probabilities."""
    logits = model(frame_a, frame_b, flow)
    flipped_a = horizontal_flip_batch(frame_a)
    flipped_b = flipped_a if degenerate_pairing else horizontal_flip_batch(frame_b)
    flipped_flow = horizontal_flip_batch(flow.clone())
    if flipped_flow.shape[1] >= 1:
        flipped_flow[:, 0] = -flipped_flow[:, 0]
    flipped_logits = model(flipped_a, flipped_b, flipped_flow)
    return torch.stack((torch.sigmoid(logits), torch.sigmoid(flipped_logits)), dim=0).mean(dim=0)


def probabilities_to_logits(probabilities: Tensor, eps: float = 1e-6) -> Tensor:
    clipped = probabilities.clamp(eps, 1.0 - eps)
    return torch.logit(clipped)


__all__ = [
    "horizontal_flip_batch",
    "predict_probabilities_with_flip_tta",
    "probabilities_to_logits",
]
