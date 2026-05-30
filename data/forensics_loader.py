"""Forensics GAN frame-pair loader for Week 4 OOD evaluation.

The forensics folders contain still images, not guaranteed video sequences.  To
preserve the Phase 3/4 runtime contract without inventing temporal labels, the
default pairing mode uses the adjacent image in the same class folder:
``images[i]`` pairs with ``images[(i + 1) % n]``.  Flow is computed on the fly
from grayscale 64x64 frames with the same Farneback implementation used by the
CelebA cache precompute path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, TypedDict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.augmentations import build_transforms
from data.precompute_flow import _load_grayscale, compute_farneback_flow


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ForensicsPairingMode = Literal["adjacent_same_class", "degenerate"]


class ForensicsFramePairSample(TypedDict):
    frame_a: torch.Tensor
    frame_b: torch.Tensor
    flow: torch.Tensor
    label: torch.Tensor
    path_a: str
    path_b: str
    metadata: Dict[str, object]


def normalize_split(name: str) -> str:
    split = name.lower().strip()
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    split = aliases.get(split, split)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported forensics split: {name}")
    return split


def resolve_forensics_root(path: str | Path) -> Path:
    """Return the concrete root containing ``train/validation/test`` folders."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Forensics path does not exist: {root}")

    if _looks_like_dataset_root(root):
        return root

    nested = root / root.name
    if _looks_like_dataset_root(nested):
        return nested

    raise FileNotFoundError(
        f"Forensics dataset root must contain train/validation/test folders: {root}"
    )


def discover_forensics_datasets(forensics_root: str | Path) -> List[Path]:
    """Discover Data Set N roots under the top-level forensics directory."""
    root = Path(forensics_root)
    if _looks_like_dataset_root(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Forensics root does not exist: {root}")

    datasets = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if candidate.is_dir() and candidate.name.lower().startswith("data set"):
            datasets.append(resolve_forensics_root(candidate))
    if not datasets:
        raise FileNotFoundError(f"No Data Set N folders found under: {root}")
    return datasets


class ForensicsFramePairDataset(Dataset):
    """Dataset yielding Phase 3/4-compatible frame pairs and flow tensors."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "test",
        *,
        image_size: int = 64,
        pairing_mode: ForensicsPairingMode = "adjacent_same_class",
        aligned_root: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.dataset_root = resolve_forensics_root(dataset_root)
        self.source_dataset_root = self.dataset_root
        self.split = normalize_split(split)
        self.aligned_root = Path(aligned_root) if aligned_root is not None else None
        if self.aligned_root is not None:
            self.dataset_root = _resolve_aligned_dataset_root(
                self.source_dataset_root,
                self.aligned_root,
                self.split,
            )
        self.image_size = int(image_size)
        self.pairing_mode = pairing_mode
        if pairing_mode not in {"adjacent_same_class", "degenerate"}:
            raise ValueError(f"Unsupported pairing mode: {pairing_mode}")
        self.transform = transform or build_transforms(image_size=image_size, train=False)

        class_items: List[List[tuple[Path, Path, int, str]]] = []
        for class_name, label in (("real", 0), ("fake", 1)):
            class_dir = self.dataset_root / self.split / class_name
            images = _image_paths(class_dir)
            items: List[tuple[Path, Path, int, str]] = []
            for index, image_path in enumerate(images):
                if pairing_mode == "degenerate" or len(images) == 1:
                    pair_path = image_path
                else:
                    pair_path = images[(index + 1) % len(images)]
                items.append((image_path, pair_path, label, class_name))
            class_items.append(items)

        self.items = _apply_balanced_limit(class_items, limit)
        if not self.items:
            raise FileNotFoundError(
                f"No forensics images found at {self.dataset_root / self.split}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ForensicsFramePairSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path_a, path_b, label, class_name = self.items[index]
        frame_a = self._load_tensor(path_a)
        frame_b = self._load_tensor(path_b)
        if self.pairing_mode == "degenerate":
            flow = torch.zeros(2, self.image_size, self.image_size, dtype=torch.float32)
        else:
            gray_a = _load_grayscale(path_a, image_size=self.image_size)
            gray_b = _load_grayscale(path_b, image_size=self.image_size)
            flow = compute_farneback_flow(gray_a, gray_b)
        if torch.isnan(flow).any():
            raise ValueError(f"Flow tensor contains NaN values for {path_a}")
        return {
            "frame_a": frame_a,
            "frame_b": frame_b,
            "flow": flow,
            "label": torch.tensor(label, dtype=torch.long),
            "path_a": str(path_a),
            "path_b": str(path_b),
            "metadata": {
                "dataset_root": str(self.dataset_root),
                "source_dataset_root": str(self.source_dataset_root),
                "aligned_root": str(self.aligned_root) if self.aligned_root else None,
                "split": self.split,
                "class_name": class_name,
                "pairing_mode": self.pairing_mode,
            },
        }

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB"))


def collate_forensics_frame_pair_batch(
    batch: Sequence[ForensicsFramePairSample],
) -> Dict[str, object]:
    return {
        "frame_a": torch.stack([sample["frame_a"] for sample in batch]),
        "frame_b": torch.stack([sample["frame_b"] for sample in batch]),
        "flow": torch.stack([sample["flow"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "path_a": [sample["path_a"] for sample in batch],
        "path_b": [sample["path_b"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }


def create_forensics_dataloader(
    dataset_root: str | Path,
    split: str = "test",
    pairing_mode: ForensicsPairingMode = "adjacent_same_class",
    batch_size: int = 64,
    num_workers: int = 0,
    limit: Optional[int] = None,
    *,
    aligned_root: Optional[str | Path] = None,
    image_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    dataset = ForensicsFramePairDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        pairing_mode=pairing_mode,
        aligned_root=aligned_root,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_forensics_frame_pair_batch,
    )


def _looks_like_dataset_root(path: Path) -> bool:
    return all((path / split).is_dir() for split in ("train", "validation", "test"))


def _image_paths(class_dir: Path) -> List[Path]:
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Forensics class directory missing: {class_dir}")
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _resolve_aligned_dataset_root(source_dataset_root: Path, aligned_root: Path, split: str) -> Path:
    """Resolve a cached aligned root for nested or flat Data Set N layouts."""
    dataset_name = (
        source_dataset_root.parent.name
        if source_dataset_root.name == source_dataset_root.parent.name
        else source_dataset_root.name
    )
    candidates = [
        aligned_root / dataset_name,
        aligned_root / dataset_name / dataset_name,
    ]
    if _looks_like_dataset_root(aligned_root) or (aligned_root / split).is_dir():
        candidates.insert(0, aligned_root)
    for candidate in candidates:
        if _looks_like_dataset_root(candidate) or (candidate / split).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Aligned forensics cache for {dataset_name!r} not found under: {aligned_root}"
    )


def _apply_balanced_limit(
    class_items: Sequence[Sequence[tuple[Path, Path, int, str]]],
    limit: Optional[int],
) -> List[tuple[Path, Path, int, str]]:
    """Apply a smoke-test limit without dropping an entire class first."""
    if limit is None:
        return [item for items in class_items for item in items]
    if limit <= 0:
        return []

    per_class = max(1, limit // max(1, len(class_items)))
    remainder = limit - per_class * len(class_items)
    limited: List[tuple[Path, Path, int, str]] = []
    for class_index, items in enumerate(class_items):
        take = per_class + (1 if class_index < remainder else 0)
        limited.extend(list(items[:take]))
    return limited[:limit]


__all__ = [
    "ForensicsFramePairDataset",
    "ForensicsFramePairSample",
    "collate_forensics_frame_pair_batch",
    "create_forensics_dataloader",
    "discover_forensics_datasets",
    "normalize_split",
    "resolve_forensics_root",
]
