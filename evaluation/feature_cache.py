"""Branch feature cache helpers for transfer ensemble evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.ensemble import BranchFeatures, extract_branch_outputs


def extract_and_save_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    out_path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    max_batches: Optional[int] = None,
) -> Path:
    """Extract Phase 3/4 branch features and save them as a compressed NPZ."""
    features, labels = extract_branch_outputs(
        model,
        dataloader,
        device,
        desc=f"feature cache -> {Path(out_path).name}",
        max_batches=max_batches,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": int(labels.shape[0]),
        "feature_dims": {
            "a": int(features["a"].shape[1]),
            "b": int(features["b"].shape[1]),
            "c": int(features["c"].shape[1]),
        },
        **dict(metadata or {}),
    }
    np.savez_compressed(
        path,
        a=features["a"],
        b=features["b"],
        c=features["c"],
        logit=features["logit"],
        labels=labels.astype(np.int64),
        metadata=json.dumps(merged_metadata, sort_keys=True),
    )
    (path.parent / "metadata.json").write_text(
        json.dumps(merged_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_feature_cache(path: str | Path) -> Tuple[BranchFeatures, np.ndarray, dict[str, Any]]:
    """Load an NPZ branch feature cache."""
    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as data:
        features: BranchFeatures = {
            "a": np.asarray(data["a"]),
            "b": np.asarray(data["b"]),
            "c": np.asarray(data["c"]),
            "logit": np.asarray(data["logit"]),
        }
        labels = np.asarray(data["labels"]).astype(np.int64)
        raw_metadata = str(data["metadata"]) if "metadata" in data.files else "{}"
    metadata = json.loads(raw_metadata)
    expected = labels.shape[0]
    for key, value in features.items():
        if value.shape[0] != expected:
            raise ValueError(
                f"Feature cache row mismatch for {key}: {value.shape[0]} != {expected}"
            )
    return features, labels, metadata


__all__ = ["extract_and_save_features", "load_feature_cache"]
