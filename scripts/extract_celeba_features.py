"""Extract CelebA train-split Phase 3 branch features for transfer RF.

Full transfer cache:
  python scripts/extract_celeba_features.py \
    --config config/config.yaml \
    --checkpoint checkpoints/phase3_a_b_c.pt \
    --split train \
    --pairing-mode adjacent_cache \
    --out runs/celeba_features/phase3_train_adjacent_cache.npz

Smoke cache:
  python scripts/extract_celeba_features.py --device cpu --limit 256 \
    --out runs/celeba_features/phase3_train_adjacent_cache_smoke.npz
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.celeba_loader import create_celeba_dataloader, load_config
from evaluation.feature_cache import extract_and_save_features
from models.discriminator import DiscriminatorPhase3, load_phase3_checkpoint


def _resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"mps", "cuda"}:
        print(f"WARNING: {name} unavailable, falling back to CPU")
    return torch.device("cpu")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_dict(value: object, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict for {context}, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CelebA branch feature cache")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/phase3_a_b_c.pt")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--pairing-mode", default="adjacent_cache", choices=("default", "adjacent_cache"))
    parser.add_argument("--out", default="runs/celeba_features/phase3_train_adjacent_cache.npz")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _as_dict(load_config(args.config), "config")
    phase3_cfg = _as_dict(config.get("phase3", {}), "config.phase3")
    dl_cfg = _as_dict(config.get("dataloader", {}), "config.dataloader")
    dl_overrides = dict(dl_cfg)
    dl_overrides["num_workers"] = args.num_workers
    dl_overrides.pop("prefetch_factor", None)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Phase 3 checkpoint not found: {checkpoint}")
    device = _resolve_device(args.device)

    model = DiscriminatorPhase3(dropout=float(phase3_cfg.get("dropout", 0.3)))
    load_phase3_checkpoint(model, None, None, checkpoint)
    loader = create_celeba_dataloader(
        config,
        split=args.split,
        shuffle=False,
        limit=args.limit,
        include_flow=True,
        pairing_mode=args.pairing_mode,
        dataloader_overrides=dl_overrides,
    )
    out = extract_and_save_features(
        model,
        loader,
        device,
        args.out,
        metadata={
            "config": args.config,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "split": args.split,
            "pairing_mode": args.pairing_mode,
            "limit": args.limit,
            "max_batches": args.max_batches,
        },
        max_batches=args.max_batches,
    )
    print(f"Saved feature cache: {out}")


if __name__ == "__main__":
    main()
