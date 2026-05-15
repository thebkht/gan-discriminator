"""CelebA frame-pair dataset and dataloader factory for Week 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.augmentations import build_transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _resolve_image_dir(path_like: Path) -> Path:
    if (path_like / "img_align_celeba").is_dir():
        return path_like / "img_align_celeba"
    return path_like


def discover_celeba_images(image_dir: str | Path) -> List[Path]:
    root = _resolve_image_dir(Path(image_dir))
    if not root.exists():
        raise FileNotFoundError(f"CelebA image directory does not exist: {root}")
    images = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image files found under: {root}")
    return images


def load_config(config_path: str | Path) -> Mapping[str, object]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = dict(config.get("paths", {}))
    project_root_value = paths.get("project_root", ".")
    project_root = (config_path.parent / project_root_value).resolve()
    paths["project_root"] = str(project_root)

    for key, value in list(paths.items()):
        if key == "project_root":
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            paths[key] = str((project_root / candidate).resolve())

    config["paths"] = paths
    return config


def validate_celeba_dataset(
    image_dir: str | Path,
    expected_count: Optional[int] = None,
    expected_resolution: Optional[Tuple[int, int]] = None,
    sample_count: int = 3,
) -> Dict[str, object]:
    images = discover_celeba_images(image_dir)
    if expected_count is not None and len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} images, found {len(images)}")

    sampled = images[: max(1, min(sample_count, len(images)))]
    resolutions = []
    for sample in sampled:
        with Image.open(sample) as img:
            resolutions.append(img.size)
            if expected_resolution is not None and img.size != expected_resolution:
                raise ValueError(
                    f"Expected resolution {expected_resolution} but found {img.size} in {sample.name}"
                )

    return {
        "image_dir": str(_resolve_image_dir(Path(image_dir))),
        "count": len(images),
        "sampled_resolutions": resolutions,
    }


@dataclass(frozen=True)
class PairMetadata:
    anchor_path: str
    pair_path: str
    pair_type: str
    pair_strategy: str
    identity: Optional[int]


class CelebAFramePairDataset(Dataset):
    """Stable Week 1 contract for real/fake frame-pair sampling.

    Real pairs:
    - With `identity_CelebA.txt` present, sample two images for the same identity.
    - Without the identity file, fall back to adjacent-index sampling.

    Fake pairs:
    - Duplicate a single image and inject small Gaussian noise into the second tensor.
    """

    def __init__(
        self,
        image_dir: str | Path,
        identity_file: str | Path | None = None,
        image_size: int = 64,
        fake_ratio: float = 0.5,
        gaussian_noise_std: float = 0.05,
        transform: Optional[Callable] = None,
        train: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.image_paths = discover_celeba_images(image_dir)
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        self.index_by_name = {path.name: idx for idx, path in enumerate(self.image_paths)}
        self.image_size = image_size
        self.fake_ratio = fake_ratio
        self.gaussian_noise_std = gaussian_noise_std
        self.transform = transform or build_transforms(image_size=image_size, train=train)
        self.identity_file = Path(identity_file) if identity_file else None
        self.identity_lookup: Dict[str, int] = {}
        self.identity_groups: Dict[int, List[int]] = {}
        self.has_identity_file = False
        self._fake_fraction = Fraction(str(fake_ratio)).limit_denominator(1000)
        self._load_identity_pairs()

    def _load_identity_pairs(self) -> None:
        if not self.identity_file or not self.identity_file.exists():
            return

        groups: Dict[int, List[int]] = defaultdict(list)
        valid_names = {path.name for path in self.image_paths}
        with open(self.identity_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, identity_value = parts
                if filename not in valid_names:
                    continue
                identity = int(identity_value)
                self.identity_lookup[filename] = identity
                groups[identity].append(self._index_for_filename(filename))

        self.identity_groups = {identity: sorted(indices) for identity, indices in groups.items()}
        self.has_identity_file = bool(self.identity_lookup)

    def _index_for_filename(self, filename: str) -> int:
        return self.index_by_name[filename]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        anchor_path = self.image_paths[index]
        is_fake = self._is_fake_index(index)

        if is_fake:
            frame_a = self._load_tensor(anchor_path)
            frame_b = self._make_fake_pair(frame_a)
            metadata = PairMetadata(
                anchor_path=str(anchor_path),
                pair_path=str(anchor_path),
                pair_type="fake",
                pair_strategy="gaussian_noise_duplicate",
                identity=self.identity_lookup.get(anchor_path.name),
            )
            label = 1
        else:
            pair_index, identity_value, strategy = self._select_real_pair(index)
            pair_path = self.image_paths[pair_index]
            frame_a = self._load_tensor(anchor_path)
            frame_b = self._load_tensor(pair_path)
            metadata = PairMetadata(
                anchor_path=str(anchor_path),
                pair_path=str(pair_path),
                pair_type="real",
                pair_strategy=strategy,
                identity=identity_value,
            )
            label = 0

        return {
            "frame_a": frame_a,
            "frame_b": frame_b,
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": metadata.__dict__,
        }

    def _is_fake_index(self, index: int) -> bool:
        if self._fake_fraction.numerator == 0:
            return False
        if self._fake_fraction.numerator >= self._fake_fraction.denominator:
            return True
        return (index % self._fake_fraction.denominator) < self._fake_fraction.numerator

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            return self.transform(img.convert("RGB"))

    def _make_fake_pair(self, anchor: torch.Tensor) -> torch.Tensor:
        noisy = anchor + torch.randn_like(anchor) * self.gaussian_noise_std
        return noisy.clamp(-1.0, 1.0)

    def _select_real_pair(self, index: int) -> Tuple[int, Optional[int], str]:
        anchor_name = self.image_paths[index].name
        identity_value = self.identity_lookup.get(anchor_name)
        if self.has_identity_file and identity_value is not None:
            group = self.identity_groups.get(identity_value, [])
            if len(group) >= 2:
                offset = group.index(index)
                pair_index = group[(offset + 1) % len(group)]
                return pair_index, identity_value, "same_identity"
            return index, identity_value, "identity_singleton"

        pair_index = min(index + 1, len(self.image_paths) - 1)
        if pair_index == index:
            pair_index = max(0, index - 1)
        return pair_index, None, "adjacent_fallback"


def collate_frame_pair_batch(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "frame_a": torch.stack([sample["frame_a"] for sample in batch]),
        "frame_b": torch.stack([sample["frame_b"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


def create_celeba_dataloader(
    config: Mapping[str, object] | str | Path,
    split: str = "train",
    shuffle: Optional[bool] = None,
    limit: Optional[int] = None,
) -> DataLoader:
    if isinstance(config, (str, Path)):
        config = load_config(config)

    paths = config["paths"]
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    dataset = CelebAFramePairDataset(
        image_dir=paths["image_dir"],
        identity_file=paths.get("identity_file"),
        image_size=int(dataset_cfg["image_size"]),
        fake_ratio=float(dataset_cfg["fake_ratio"]),
        gaussian_noise_std=float(dataset_cfg["gaussian_noise_std"]),
        train=(split == "train"),
        limit=limit,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=int(dataloader_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(dataloader_cfg["num_workers"]),
        pin_memory=bool(dataloader_cfg["pin_memory"]),
        drop_last=bool(dataloader_cfg["drop_last"]),
        collate_fn=collate_frame_pair_batch,
    )
