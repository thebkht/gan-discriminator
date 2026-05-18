"""CLI entrypoint for Week 2 Phase 2 training."""

from __future__ import annotations

import argparse

from training.phase2_trainer import train_phase2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Week 2 Phase 2 A+B discriminator")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap for train samples")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional cap for validation samples")
    parser.add_argument("--run-name", default="phase2_a_b", help="Run directory name")
    parser.add_argument("--epochs-override", type=int, default=None, help="Optional epoch override for dry runs")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for batches per split")
    parser.add_argument(
        "--checkpoint-name-override",
        default=None,
        help="Optional checkpoint filename override for smoke runs or alternate outputs.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Optional device override. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--tracker-backend",
        default=None,
        help="Optional tracking backend. Set to tensorboard to emit TensorBoard logs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_phase2(
        args.config,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        run_name=args.run_name,
        tracker_backend=args.tracker_backend,
        epochs_override=args.epochs_override,
        device_override=args.device,
        max_batches=args.max_batches,
        checkpoint_name_override=args.checkpoint_name_override,
    )


if __name__ == "__main__":
    main()
