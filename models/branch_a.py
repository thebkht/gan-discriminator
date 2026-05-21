"""Branch A baseline model for pair-labelled CelebA training."""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, *, use_batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ),
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class BranchAEncoder(nn.Module):
    """Spatial encoder that maps a 64x64 RGB frame into a 2048-D feature vector."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64, use_batch_norm=False),
            _conv_block(64, 128, use_batch_norm=True),
            _conv_block(128, 256, use_batch_norm=True),
            _conv_block(256, 512, use_batch_norm=True),
            _conv_block(512, 512, use_batch_norm=False),
        )

    def set_trainable_blocks(self, train_last_n: int = 0) -> None:
        total_blocks = len(self.features)
        for index, block in enumerate(self.features):
            requires_grad = index >= total_blocks - train_last_n
            for parameter in block.parameters():
                parameter.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.features(x)
        return encoded.flatten(start_dim=1)


class BranchABaseline(nn.Module):
    """Week 1 baseline that applies the Branch A encoder to both frames."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = BranchAEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        feature_a = self.encoder(frame_a)
        feature_b = self.encoder(frame_b)
        logits = self.classifier(torch.cat([feature_a, feature_b], dim=1))
        return logits.squeeze(1)
