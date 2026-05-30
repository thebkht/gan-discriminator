"""Build an MTCNN-aligned forensics face cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.face_align import align_face, align_face_or_fallback
from data.forensics_loader import discover_forensics_datasets, normalize_split


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def preprocess_forensics_faces(
    *,
    forensics_root: Path,
    output_root: Path,
    splits: list[str],
    margin: float,
    image_size: int,
    fallback: str,
    device: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"output_root": str(output_root), "datasets": {}}
    for dataset_root in discover_forensics_datasets(forensics_root):
        dataset_name = dataset_root.parent.name if dataset_root.name == dataset_root.parent.name else dataset_root.name
        dataset_report: dict[str, Any] = {}
        for split in splits:
            split = normalize_split(split)
            split_report: dict[str, Any] = {"detected": 0, "fallback": 0, "failed": 0, "total": 0}
            for class_name in ("real", "fake"):
                source_dir = dataset_root / split / class_name
                output_dir = output_root / dataset_name / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                images = sorted(
                    path for path in source_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
                )
                for image_path in tqdm(images, desc=f"{dataset_name} {split} {class_name}", leave=False):
                    split_report["total"] += 1
                    try:
                        with Image.open(image_path) as image:
                            aligned = align_face(
                                image,
                                margin=margin,
                                image_size=image_size,
                                device=device,
                            )
                            if aligned is None:
                                split_report["fallback"] += 1
                                aligned = align_face_or_fallback(
                                    image,
                                    margin=margin,
                                    image_size=image_size,
                                    fallback=fallback,
                                    device=device,
                                )
                            else:
                                split_report["detected"] += 1
                            aligned.resize((image_size, image_size), Image.Resampling.BILINEAR).save(output_dir / image_path.name)
                    except Exception as error:  # noqa: BLE001 - report and continue cache build
                        split_report["failed"] += 1
                        print(f"WARNING: failed to align {image_path}: {error}", file=sys.stderr)
            total = max(1, split_report["total"])
            split_report["detection_rate"] = split_report["detected"] / total
            split_report["fallback_rate"] = split_report["fallback"] / total
            dataset_report[split] = split_report
        report["datasets"][dataset_name] = dataset_report
    report_path = output_root / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess forensics images into an aligned face cache")
    parser.add_argument("--forensics-root", default="data/forensics")
    parser.add_argument("--output-root", default="data/forensics_aligned")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--fallback", default="center_crop", choices=("center_crop",))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = preprocess_forensics_faces(
        forensics_root=Path(args.forensics_root),
        output_root=Path(args.output_root),
        splits=list(args.splits),
        margin=args.margin,
        image_size=args.image_size,
        fallback=args.fallback,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
