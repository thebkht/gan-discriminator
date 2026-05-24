"""Branch C physics features for Phase 3.

Cardinal direction bins are defined directly in radians from `torch.atan2(dy, dx)`:

| Bin | Radian range | Compass |
| --- | ------------ | ------- |
| E | `[-pi/4, pi/4)` | East (+x) |
| N | `[pi/4, 3pi/4)` | North (+y) |
| S | `[-3pi/4, -pi/4)` | South (-y) |
| W | `[3pi/4, pi]` U `[-pi, -3pi/4)` | West (-x) |
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _field_stats(field: Tensor) -> Tensor:
    flattened = field.flatten(start_dim=1)
    maximum = flattened.max(dim=1).values
    minimum = flattened.min(dim=1).values
    return torch.stack(
        [
            flattened.mean(dim=1),
            flattened.std(dim=1, unbiased=False),
            maximum,
            minimum,
            maximum - minimum,
        ],
        dim=1,
    )


def _finite_difference_x(field: Tensor) -> Tensor:
    right = torch.roll(field, shifts=-1, dims=2)
    gradient = right - field
    gradient[:, :, -1] = field[:, :, -1] - field[:, :, -2]
    return gradient


def _finite_difference_y(field: Tensor) -> Tensor:
    down = torch.roll(field, shifts=-1, dims=1)
    gradient = down - field
    gradient[:, -1, :] = field[:, -1, :] - field[:, -2, :]
    return gradient


def _rgb_to_hsv_torch(rgb: Tensor) -> Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor shaped (B, 3, H, W), got {tuple(rgb.shape)}")
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    delta = maxc - minc

    safe_delta = torch.where(delta > 0, delta, torch.ones_like(delta))
    hue = torch.zeros_like(maxc)
    red_mask = (maxc == r) & (delta > 0)
    green_mask = (maxc == g) & (delta > 0)
    blue_mask = (maxc == b) & (delta > 0)
    hue[red_mask] = torch.remainder(((g - b) / safe_delta)[red_mask], 6.0)
    hue[green_mask] = (((b - r) / safe_delta) + 2.0)[green_mask]
    hue[blue_mask] = (((r - g) / safe_delta) + 4.0)[blue_mask]
    hue = hue / 6.0

    saturation = torch.where(maxc > 0, delta / torch.clamp(maxc, min=1e-12), torch.zeros_like(maxc))
    value = maxc
    return torch.stack([hue, saturation, value], dim=1)


def _hsv_summary(frame: Tensor) -> Tensor:
    hsv = _rgb_to_hsv_torch(torch.clamp((frame + 1.0) / 2.0, 0.0, 1.0))
    hue = hsv[:, 0].flatten(start_dim=1)
    saturation = hsv[:, 1].flatten(start_dim=1)
    value = hsv[:, 2].flatten(start_dim=1)
    return torch.stack(
        [
            hue.mean(dim=1),
            hue.std(dim=1, unbiased=False),
            saturation.mean(dim=1),
            value.mean(dim=1),
        ],
        dim=1,
    )


class BranchC_Physics(nn.Module):
    """Deterministic Phase 3 physics branch with a fixed 28-D output."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 28

    def _flow_summary(self, flow: Tensor) -> Tensor:
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"Expected flow tensor shaped (B, 2, H, W), got {tuple(flow.shape)}")
        dx = flow[:, 0]
        dy = flow[:, 1]
        divergence = _finite_difference_x(dx) + _finite_difference_y(dy)
        curl = _finite_difference_x(dy) - _finite_difference_y(dx)
        magnitude = torch.sqrt(torch.clamp(dx.square() + dy.square(), min=1e-12))

        theta = torch.atan2(dy, dx)
        east = ((theta >= (-math.pi / 4.0)) & (theta < (math.pi / 4.0))).float().flatten(start_dim=1).mean(dim=1)
        north = ((theta >= (math.pi / 4.0)) & (theta < (3.0 * math.pi / 4.0))).float().flatten(start_dim=1).mean(dim=1)
        south = ((theta >= (-3.0 * math.pi / 4.0)) & (theta < (-math.pi / 4.0))).float().flatten(start_dim=1).mean(dim=1)
        west = (
            ((theta >= (3.0 * math.pi / 4.0)) & (theta <= math.pi))
            | ((theta >= -math.pi) & (theta < (-3.0 * math.pi / 4.0)))
        ).float().flatten(start_dim=1).mean(dim=1)

        parts = [
            _field_stats(divergence),
            _field_stats(curl),
            _field_stats(magnitude),
            torch.stack(
                [
                    magnitude.flatten(start_dim=1).mean(dim=1),
                    east,
                    north,
                    south,
                    west,
                ],
                dim=1,
            ),
        ]
        return torch.cat(parts, dim=1)

    def forward(self, frame_a: Tensor, frame_b: Tensor, flow: Tensor) -> Tensor:
        flow_features = self._flow_summary(flow)
        hsv_features = torch.cat([_hsv_summary(frame_a), _hsv_summary(frame_b)], dim=1)
        features = torch.cat([flow_features, hsv_features], dim=1)
        if features.shape[1] != self.output_dim:
            raise AssertionError(f"Expected {self.output_dim} Branch C features, got {features.shape[1]}")
        return features


__all__ = ["BranchC_Physics"]
