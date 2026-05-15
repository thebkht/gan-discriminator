"""Offline optical-flow precompute script for Week 1."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

import cv2
import torch
from tqdm import tqdm

from data.celeba_loader import discover_celeba_images


def _load_grayscale(path: Path, image_size: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized)


def compute_farneback_flow(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    flow = cv2.calcOpticalFlowFarneback(
        frame_a.numpy(),
        frame_b.numpy(),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return torch.from_numpy(flow).permute(2, 0, 1).float()


def precompute_flow(
    img_dir: str | Path,
    out_dir: str | Path,
    method: str = "farneback",
    image_size: int = 64,
) -> int:
    if method.lower() != "farneback":
        raise ValueError("Only --method farneback is supported in Week 1")

    images = discover_celeba_images(img_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(tqdm(images, desc="Precomputing optical flow")):
        partner_index = min(idx + 1, len(images) - 1)
        if partner_index == idx:
            partner_index = max(0, idx - 1)
        frame_a = _load_grayscale(image_path, image_size=image_size)
        frame_b = _load_grayscale(images[partner_index], image_size=image_size)
        flow = compute_farneback_flow(frame_a, frame_b)
        torch.save(flow, out_path / f"{image_path.stem}_flow.pt")

    return len(images)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute CelebA optical flow tensors")
    parser.add_argument("--img-dir", required=True, help="Directory containing CelebA images")
    parser.add_argument("--out-dir", required=True, help="Directory for cached .pt flow tensors")
    parser.add_argument("--method", default="farneback", help="Optical flow method")
    parser.add_argument("--image-size", type=int, default=64, help="Output flow size")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = time.time()
    count = precompute_flow(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        method=args.method,
        image_size=args.image_size,
    )
    duration = time.time() - start
    print(f"Precomputed {count} flow tensors in {duration:.2f}s")


if __name__ == "__main__":
    main()
