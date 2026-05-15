"""Train/eval augmentation builders for CelebA image tensors."""

from __future__ import annotations

from typing import Callable

from torchvision import transforms


def build_transforms(image_size: int = 64, train: bool = True) -> Callable:
    """Build the documented Week 1 transform pipeline.

    Output tensors are normalized to the `[-1, 1]` range expected by later training
    stages.
    """

    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)
