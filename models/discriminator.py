"""Phase 2 and Phase 3 discriminator checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol

import torch
from torch import Tensor, nn

from models.branch_a import BranchAEncoder
from models.branch_b import BranchB_Spatiotemporal
from models.branch_c import BranchC_Physics


class _SchedulerWithStateDict(Protocol):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


def _remap_phase1_encoder_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            remapped[key[len("encoder.") :]] = value
    return remapped


def _reject_legacy_branch_b_keys(state_dict: dict[str, Tensor]) -> None:
    legacy = [key for key in state_dict if key.startswith("branch_b.embed.")]
    if legacy:
        raise RuntimeError(
            "Checkpoint contains legacy EmbedCNN keys (branch_b.embed.*). "
            "Use a Run 3 checkpoint trained with shared BranchAEncoder."
        )


class DiscriminatorPhase2(nn.Module):
    """Phase 2 discriminator with a shared Branch A encoder and trainable fusion."""

    def __init__(self, dropout: float = 0.4, backbone_train_last_n: int = 0) -> None:
        super().__init__()
        self.branch_a = BranchAEncoder()
        self.branch_b = BranchB_Spatiotemporal(self.branch_a)
        self.backbone_train_last_n = backbone_train_last_n
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048 + self.branch_b.output_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )
        self.freeze_branch_a(train_last_n=backbone_train_last_n)

    def freeze_branch_a(self, train_last_n: int = 0) -> None:
        self.backbone_train_last_n = train_last_n
        self.branch_a.set_trainable_blocks(train_last_n=train_last_n)
        self.branch_a.eval()

    def train(self, mode: bool = True) -> "DiscriminatorPhase2":
        super().train(mode)
        if not mode:
            self.branch_a.eval()
            return self
        frozen_until = max(0, len(self.branch_a.features) - self.backbone_train_last_n)
        # Frozen early blocks keep BN in eval mode so their running stats do not drift.
        for block_index in range(frozen_until):
            self.branch_a.features[block_index].eval()
        # Unfrozen tail blocks keep BN in train mode for the Branch B finetuning pass.
        for block_index in range(frozen_until, len(self.branch_a.features)):
            self.branch_a.features[block_index].train()
        return self

    def forward(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        # feat_a is intentionally detached; Conv4/Conv5 gradients only flow through Branch B.
        with torch.no_grad():
            feat_a = self.branch_a(frame_a)
        # When the tail blocks are trainable, their BN stats see frame_a twice per step:
        # once here via feat_a's no_grad path and once again through Branch B's grad path.
        feat_b = self.branch_b(frame_a, frame_b)
        logits = self.fusion(torch.cat([feat_a, feat_b], dim=1))
        return logits.squeeze(1)


def load_pretrained_branch_a(model: DiscriminatorPhase2, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    if not isinstance(model_state, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict")
    _reject_legacy_branch_b_keys(model_state)
    remapped = _remap_phase1_encoder_keys(model_state)
    incompatible = model.branch_a.load_state_dict(remapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Branch A load result: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.freeze_branch_a(train_last_n=model.backbone_train_last_n)

    fresh_branch_a = BranchAEncoder()
    loaded_parameter = next(model.branch_a.parameters()).detach()
    fresh_parameter = next(fresh_branch_a.parameters()).detach()
    if torch.allclose(loaded_parameter, fresh_parameter):
        raise AssertionError("branch_a appears uninitialized after checkpoint load")


def load_phase2_checkpoint(model: DiscriminatorPhase2, path: Path) -> None:
    """Load a full Phase 2 checkpoint. Resume wiring remains a trainer TODO."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict")
    _reject_legacy_branch_b_keys(state)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Phase 2 load result: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
        # TODO: wire this helper into a future train_phase2 --resume path.


class DiscriminatorPhase3(nn.Module):
    """Phase 3 discriminator with frozen A+B features and Branch C flow/HSV fusion."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.branch_a = BranchAEncoder()
        self.branch_b = BranchB_Spatiotemporal(self.branch_a)
        self.branch_c = BranchC_Physics()
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048 + self.branch_b.output_dim + self.branch_c.output_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )
        self.freeze_branches_ab()

    def freeze_branches_ab(self) -> None:
        self.branch_a.set_trainable_blocks(train_last_n=0)
        self.branch_a.eval()
        self.branch_b.eval()
        for parameter in self.branch_a.parameters():
            parameter.requires_grad = False
        for parameter in self.branch_b.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "DiscriminatorPhase3":
        super().train(mode)
        if mode:
            self.branch_a.eval()
            self.branch_b.eval()
        return self

    def forward(self, frame_a: Tensor, frame_b: Tensor, flow: Tensor) -> Tensor:
        with torch.no_grad():
            feat_a = self.branch_a(frame_a).detach()
            feat_b = self.branch_b(frame_a, frame_b).detach()
        feat_c = self.branch_c(frame_a, frame_b, flow)
        logits = self.fusion(torch.cat([feat_a, feat_b, feat_c], dim=1))
        return logits.squeeze(1)


def load_phase2_into_phase3(model: DiscriminatorPhase3, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict")
    _reject_legacy_branch_b_keys(state)

    branch_a_state = {
        key[len("branch_a.") :]: value for key, value in state.items() if key.startswith("branch_a.")
    }
    branch_b_state = {
        key[len("branch_b.") :]: value for key, value in state.items() if key.startswith("branch_b.")
    }
    incompatible_a = model.branch_a.load_state_dict(branch_a_state, strict=True)
    incompatible_b = model.branch_b.load_state_dict(branch_b_state, strict=True)
    if incompatible_a.missing_keys or incompatible_a.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Branch A Phase 3 load result: missing={incompatible_a.missing_keys}, unexpected={incompatible_a.unexpected_keys}"
        )
    if incompatible_b.missing_keys or incompatible_b.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Branch B Phase 3 load result: missing={incompatible_b.missing_keys}, unexpected={incompatible_b.unexpected_keys}"
        )
    model.freeze_branches_ab()


def load_phase3_checkpoint(
    model: DiscriminatorPhase3,
    optimizer: torch.optim.Optimizer | None,
    scheduler: _SchedulerWithStateDict | None,
    path: Path,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model_state_dict must be a dict")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Phase 3 load result: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
    model.freeze_branches_ab()
    return int(checkpoint["epoch"]) + 1


__all__ = [
    "DiscriminatorPhase2",
    "DiscriminatorPhase3",
    "_reject_legacy_branch_b_keys",
    "_remap_phase1_encoder_keys",
    "load_phase2_checkpoint",
    "load_phase2_into_phase3",
    "load_phase3_checkpoint",
    "load_pretrained_branch_a",
]
