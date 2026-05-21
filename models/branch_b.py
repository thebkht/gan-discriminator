"""Branch B spatiotemporal features for Week 2 Phase 2.

Committed base 8-D layout per batch row:
- velocity: mean, std, max, min
- cosine similarity between the two shared-encoder embeddings
- L2 distance between the two shared-encoder embeddings
- sign consistency between velocity and the first-frame embedding
- mean absolute velocity

The golden regression test snapshots this committed implementation;
regenerate that snapshot only when this formula or the stat layout changes
intentionally.

All scalar reductions operate over the shared encoder embedding axis only.
Standard deviation is pinned to the population definition (`unbiased=False`)
so the result stays stable across PyTorch versions and small batch sizes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SUMMARY_FEATURE_NAMES = (
    "vel_mean",
    "vel_std",
    "vel_max",
    "vel_min",
    "cos_sim",
    "l2_dist",
    "sign_consistency",
    "abs_vel_mean",
)
SIGN_CONSISTENCY_IDX = SUMMARY_FEATURE_NAMES.index("sign_consistency")


def _scalar_stats(x: Tensor, stats: tuple[str, ...]) -> Tensor:
    """Reduce a `(B, D)` tensor over `dim=1` only and return `(B, len(stats))`."""

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor shaped (B, D); received {tuple(x.shape)}")

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
        if stat_name == "min":
            reduced.append(x.min(dim=1).values)
            continue
        raise ValueError(f"Unsupported scalar stat: {stat_name}")
    return torch.stack(reduced, dim=1)


class BranchB_Spatiotemporal(nn.Module):
    """Two-frame spatiotemporal summary branch with a 32-D learned output."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.output_dim = 32
        self.backbone = backbone
        self.expander = nn.Sequential(
            nn.LayerNorm(len(SUMMARY_FEATURE_NAMES)),
            nn.Linear(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _encode(self, frame: Tensor) -> Tensor:
        return self.backbone(frame)

    def _summary_features(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Return the committed 8-D temporal summary before learned expansion."""

        e_t = self._encode(frame_a)
        e_t1 = self._encode(frame_b)

        velocity = e_t1 - e_t
        cos_sim = F.cosine_similarity(e_t, e_t1, dim=1, eps=1e-8).unsqueeze(1)
        l2_dist = velocity.norm(dim=1, keepdim=True)
        sign_consistency = (velocity.sign() == e_t.sign()).float().mean(dim=1, keepdim=True)
        parts = [
            _scalar_stats(velocity, ("mean", "std", "max", "min")),
            cos_sim,
            l2_dist,
            sign_consistency,
            velocity.abs().mean(dim=1, keepdim=True),
        ]
        summary = torch.cat(parts, dim=1)
        if summary.shape[1] != len(SUMMARY_FEATURE_NAMES):
            raise AssertionError(
                f"Expected {len(SUMMARY_FEATURE_NAMES)} summary features, got {summary.shape[1]}"
            )
        return summary

    def forward(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        summary = self._summary_features(frame_a, frame_b)
        return self.expander(summary)


__all__ = [
    "BranchB_Spatiotemporal",
    "SIGN_CONSISTENCY_IDX",
    "SUMMARY_FEATURE_NAMES",
    "_scalar_stats",
]
