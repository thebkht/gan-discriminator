"""Forensics validation threshold calibration for the frozen Phase 3 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.celeba_loader import load_config
from data.forensics_loader import create_forensics_dataloader, discover_forensics_datasets, normalize_split
from evaluation.ood_eval import _apply_branch_b_inversion, _dataset_key, _extract_features_with_paths
from evaluation.threshold_sweep import best_threshold_by, sweep_thresholds, write_sweep_json, write_sweep_markdown
from models.discriminator import DiscriminatorPhase3, load_phase3_checkpoint


def run_forensics_threshold_sweep(
    *,
    config: dict[str, Any],
    forensics_root: Path,
    aligned_root: Optional[Path],
    checkpoint: Path,
    split: str,
    pairing: str,
    run_dir: Path,
    device: torch.device,
    steps: int,
    batch_size: int,
    num_workers: int,
    limit: Optional[int],
    max_batches: Optional[int],
    branch_b_invert_logits: bool,
    tta: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model = DiscriminatorPhase3(dropout=float(dict(config.get("phase3", {})).get("dropout", 0.3)))
    load_phase3_checkpoint(model, None, None, checkpoint)
    model.eval().to(device)

    pooled_logits: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    per_dataset: dict[str, Any] = {}
    branch_b_check: dict[str, Any] = {}

    for dataset_root in discover_forensics_datasets(forensics_root):
        dataset_key = _dataset_key(dataset_root)
        loader = create_forensics_dataloader(
            dataset_root,
            split=split,
            pairing_mode=pairing,  # type: ignore[arg-type]
            batch_size=batch_size,
            num_workers=num_workers,
            limit=limit,
            aligned_root=aligned_root,
            shuffle=False,
        )
        features, labels, _ = _extract_features_with_paths(
            model,
            loader,
            device,
            desc=f"{dataset_key} threshold",
            max_batches=max_batches,
            pairing=pairing,  # type: ignore[arg-type]
            tta=tta,
        )
        if branch_b_invert_logits:
            features = _apply_branch_b_inversion(features)
        logits = features["logit"]
        records = sweep_thresholds(logits, labels, steps=steps)
        best = best_threshold_by(records, "balanced_accuracy")
        per_dataset[dataset_key] = {
            "threshold": float(best["threshold"]),
            "best": best,
            "n_images": int(labels.shape[0]),
            "class_counts": {"real": int((labels == 0).sum()), "fake": int((labels == 1).sum())},
        }
        branch_b_check[dataset_key] = _branch_b_histogram_summary(features["b"], labels)
        pooled_logits.append(logits)
        pooled_labels.append(labels)

    logits = np.concatenate(pooled_logits, axis=0)
    labels = np.concatenate(pooled_labels, axis=0)
    pooled_records = sweep_thresholds(logits, labels, steps=steps)
    pooled_best = best_threshold_by(pooled_records, "balanced_accuracy")

    write_sweep_json(
        {"best": pooled_best, "records": pooled_records},  # type: ignore[arg-type]
        run_dir / "threshold_sweep.json",
    )
    write_sweep_markdown(pooled_records, run_dir / "threshold_sweep.md", pooled_best)
    (run_dir / "per_dataset_thresholds.json").write_text(
        json.dumps({"thresholds": per_dataset}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "branch_b_posterior_check.json").write_text(
        json.dumps(branch_b_check, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "split": split,
        "pairing": pairing,
        "aligned_root": str(aligned_root) if aligned_root else None,
        "checkpoint": str(checkpoint),
        "branch_b_invert_logits": branch_b_invert_logits,
        "tta": tta,
        "pooled": {"threshold": float(pooled_best["threshold"]), "best": pooled_best},
        "per_dataset": per_dataset,
        "branch_b_posterior_check": branch_b_check,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _branch_b_histogram_summary(features_b: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scores = features_b.mean(axis=1)
    real = scores[labels == 0]
    fake = scores[labels == 1]
    real_mean = float(real.mean()) if real.size else float("nan")
    fake_mean = float(fake.mean()) if fake.size else float("nan")
    return {
        "score": "mean_branch_b_feature",
        "real_mean": real_mean,
        "fake_mean": fake_mean,
        "polarity": "inverted" if fake_mean > real_mean else "correct_or_unseparated",
        "real_histogram": _histogram(real),
        "fake_histogram": _histogram(fake),
    }


def _histogram(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=20)
    return {"bins": [float(edge) for edge in edges], "counts": [int(count) for count in counts]}


def _resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Phase 3 thresholds on forensics validation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--forensics-root", default="data/forensics")
    parser.add_argument("--aligned-root", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/phase3_a_b_c.pt")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--pairing", default="degenerate", choices=("adjacent_same_class", "degenerate"))
    parser.add_argument("--run-dir", default="runs/forensics_threshold")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--steps", type=int, default=99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--branch-b-invert-logits", action="store_true")
    parser.add_argument("--tta", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_forensics_threshold_sweep(
        config=dict(load_config(args.config)),
        forensics_root=Path(args.forensics_root),
        aligned_root=Path(args.aligned_root) if args.aligned_root else None,
        checkpoint=Path(args.checkpoint),
        split=normalize_split(args.split),
        pairing=args.pairing,
        run_dir=Path(args.run_dir),
        device=_resolve_device(args.device),
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        max_batches=args.max_batches,
        branch_b_invert_logits=args.branch_b_invert_logits,
        tta=args.tta,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
