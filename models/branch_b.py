"""Branch B spatiotemporal features for Week 2 Phase 2.

Committed base 8-D layout per batch row:
- velocity: mean, std, max
- curvature: mean, std, max
- acceleration: mean, max

The acceleration proxy is intentionally fixed to
`acceleration = velocity * velocity.sign()` because Phase 2 only sees two
frames and cannot compute a true second derivative. The golden regression
test snapshots this committed implementation; regenerate that snapshot only
when this formula or the stat layout changes intentionally.

All scalar reductions operate over the 64-D embedding axis only. Standard
deviation is pinned to the population definition (`unbiased=False`) so the
result stays stable across PyTorch versions and small batch sizes.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


def _conv_block(in_channels: int, out_channels: int, *, use_batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _scalar_stats(x: Tensor, stats: tuple[str, ...]) -> Tensor:
    """Reduce a `(B, 64)` tensor over `dim=1` only and return `(B, len(stats))`."""

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor shaped (B, 64); received {tuple(x.shape)}")

    reduced: list[Tensor] = []
    for stat_name in stats:
        if stat_name == "mean":
            reduced.append(x.mean(dim=1))
            continue
        if stat_name == "std":
            reduced.append(x.std(dim=1, unbiased=False))
            continue
        if stat_name == "max":
            reduced.append(x.max(dim=1).values)
            continue
        raise ValueError(f"Unsupported scalar stat: {stat_name}")
    return torch.stack(reduced, dim=1)


class EmbedCNN(nn.Module):
    """Lightweight tied encoder that maps one 64x64 RGB frame into a 64-D embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, use_batch_norm=False),
            _conv_block(32, 64, use_batch_norm=True),
            _conv_block(64, 128, use_batch_norm=True),
            _conv_block(128, 128, use_batch_norm=True),
        )
        self.projection = nn.Linear(4 * 4 * 128, 64)

    def forward(self, frame: Tensor) -> Tensor:
        encoded = self.features(frame)
        flattened = encoded.flatten(start_dim=1)
        return self.projection(flattened)


class BranchB_Spatiotemporal(nn.Module):
    """Two-frame spatiotemporal summary branch with a 32-D learned output."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 32
        self.embed = EmbedCNN()
        self.expander = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _summary_features(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Return the committed 8-D temporal summary before learned expansion."""

        e_t = self.embed(frame_a)
        e_t1 = self.embed(frame_b)

        velocity = e_t1 - e_t
        curvature = velocity / (velocity.norm(dim=1, keepdim=True) + 1e-8)
        acceleration = velocity * velocity.sign()

        stats: Iterable[Tensor] = (
            _scalar_stats(velocity, ("mean", "std", "max")),
            _scalar_stats(curvature, ("mean", "std", "max")),
            _scalar_stats(acceleration, ("mean", "max")),
        )
        return torch.cat(list(stats), dim=1)

    def forward(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        summary = self._summary_features(frame_a, frame_b)
        return self.expander(summary)


__all__ = ["BranchB_Spatiotemporal", "EmbedCNN", "_scalar_stats"]
