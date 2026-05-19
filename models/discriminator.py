"""Phase 2 discriminator and Phase 1 Branch A checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn

from models.branch_a import BranchAEncoder
from models.branch_b import BranchB_Spatiotemporal


def _remap_phase1_encoder_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            remapped[key[len("encoder.") :]] = value
    return remapped


class DiscriminatorPhase2(nn.Module):
    """Phase 2 discriminator with a frozen Branch A encoder and trainable Branch B."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.branch_a = BranchAEncoder()
        self.branch_b = BranchB_Spatiotemporal()
        self.fusion = nn.Sequential(
            nn.Linear(2048 + self.branch_b.output_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )
        self.freeze_branch_a()

    def freeze_branch_a(self) -> None:
        for parameter in self.branch_a.parameters():
            parameter.requires_grad = False
        self.branch_a.eval()

    def train(self, mode: bool = True) -> "DiscriminatorPhase2":
        super().train(mode)
        self.branch_a.eval()
        return self

    def forward(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        with torch.no_grad():
            feat_a = self.branch_a(frame_a)
        feat_b = self.branch_b(frame_a, frame_b)
        logits = self.fusion(torch.cat([feat_a, feat_b], dim=1))
        return logits.squeeze(1)


def load_pretrained_branch_a(model: DiscriminatorPhase2, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    if not isinstance(model_state, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict")
    remapped = _remap_phase1_encoder_keys(model_state)
    incompatible = model.branch_a.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Branch A load result: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.freeze_branch_a()

    fresh_branch_a = BranchAEncoder()
    loaded_parameter = next(model.branch_a.parameters()).detach()
    fresh_parameter = next(fresh_branch_a.parameters()).detach()
    if torch.allclose(loaded_parameter, fresh_parameter):
        raise AssertionError("branch_a appears uninitialized after checkpoint load")


__all__ = ["DiscriminatorPhase2", "load_pretrained_branch_a", "_remap_phase1_encoder_keys"]
