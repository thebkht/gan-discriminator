"""Compatibility wrapper for the Week 4 forensics evaluator.

Use ``python -m evaluation.ood_eval`` or ``scripts/run_forensics_eval.py`` for
new runs.  This module delegates Phase 3 single-checkpoint evaluation to the
same OOD path so validation split naming, frame pairing, and flow are shared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from data.celeba_loader import load_config
from evaluation.ood_eval import main as ood_main
from evaluation.ood_eval import run_forensics_eval


def evaluate_forensics(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str = "test",
    device: str | torch.device = "cpu",
    *,
    config: str | Path = "config/config.yaml",
) -> Mapping[str, Any]:
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    root = Path(data_root)
    forensics_root = root.parent if root.name.lower().startswith("data set") else root
    dataset = root.name if root.name.lower().startswith("data set") else None
    return run_forensics_eval(
        load_config(config),
        run_dir=Path("runs/forensics_eval_compat"),
        checkpoint=Path(checkpoint_path),
        forensics_root=forensics_root,
        dataset=dataset,
        split=split,
        pairing="adjacent_same_class",
        mode="neural",
        device=torch_device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for evaluation.ood_eval")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/phase3_a_b_c.pt")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--run-dir", default="runs/forensics_eval")
    parser.add_argument("--pairing", default="adjacent_same_class", choices=("adjacent_same_class", "degenerate"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ood_main(
        [
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--forensics-root", args.data_root,
            "--split", args.split,
            "--device", args.device,
            "--run-dir", args.run_dir,
            "--pairing", args.pairing,
            "--mode", "neural",
        ]
    )


if __name__ == "__main__":
    main()
