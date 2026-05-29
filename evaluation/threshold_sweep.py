"""Phase 3 decision-threshold sweep — Week 3 Dev 2.

Sweeps the sigmoid decision threshold from 0.01 to 0.99 on the Phase 3
checkpoint's test-split logits and reports balanced accuracy, F1, TPR, and TNR
at every step.  The sweep writes:

  {run_dir}/threshold_sweep.json   — full per-threshold records
  {run_dir}/threshold_sweep.md     — human-readable summary table
  {run_dir}/threshold_sweep.png    — curve plot (if matplotlib is available)

Usage (standalone)
------------------
  python -m evaluation.threshold_sweep \
      --config config/config.yaml \
      --checkpoint checkpoints/phase3_a_b_c.pt \
      --run-dir runs/threshold_sweep_phase3 \
      --device cpu
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ── optional matplotlib ──────────────────────────────────────────────────────
try:
    _mpl = importlib.import_module("matplotlib")
    _mpl.use("Agg")
    _plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:
    _mpl = None  # type: ignore[assignment]
    _plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------

def sweep_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 99,
    start: float = 0.01,
    stop: float = 0.99,
) -> List[Dict[str, float]]:
    """Return per-threshold metrics for every threshold in [start, stop].

    Parameters
    ----------
    logits : raw model logits (before sigmoid), shape (N,)
    labels : int64 binary labels (0=real, 1=fake), shape (N,)
    steps  : number of evenly-spaced thresholds between start and stop

    Returns
    -------
    List of dicts, one per threshold, with keys:
      threshold, balanced_accuracy, f1, tpr, tnr, precision, recall
    """
    proba = _stable_sigmoid(logits)

    thresholds = np.linspace(start, stop, steps)
    records: List[Dict[str, float]] = []

    real_mask = labels == 0
    fake_mask = labels == 1
    n_real = int(real_mask.sum())
    n_fake = int(fake_mask.sum())

    for t in thresholds:
        preds = (proba >= t).astype(np.int64)

        tp = int(((preds == 1) & fake_mask).sum())
        tn = int(((preds == 0) & real_mask).sum())
        fp = int(((preds == 1) & real_mask).sum())
        fn = int(((preds == 0) & fake_mask).sum())

        tpr = tp / n_fake if n_fake > 0 else float("nan")
        tnr = tn / n_real if n_real > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tpr
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else float("nan")
        )
        balanced_acc = (tpr + tnr) / 2

        records.append({
            "threshold":         round(float(t), 4),
            "balanced_accuracy": round(balanced_acc, 6),
            "f1":                round(f1, 6),
            "tpr":               round(tpr, 6),
            "tnr":               round(tnr, 6),
            "precision":         round(precision, 6),
            "recall":            round(recall, 6),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    return records


def best_threshold_by(
    records: List[Dict[str, float]],
    metric: str = "balanced_accuracy",
) -> Dict[str, float]:
    """Return the record with the highest value for *metric*."""
    return max(records, key=lambda r: r[metric])


# ---------------------------------------------------------------------------
# Collect logits from the model
# ---------------------------------------------------------------------------

def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def collect_logits_and_labels(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    include_flow: bool = True,
    max_batches: Optional[int] = None,
    desc: str = "collecting logits",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (logits, labels) arrays."""
    model.eval()
    model = model.to(device)

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    done = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            frame_a = batch["frame_a"].to(device)
            frame_b = batch["frame_b"].to(device)
            labels  = batch["label"].cpu().numpy().astype(np.int64)

            if include_flow:
                flow = batch["flow"].to(device)
                logits = model(frame_a, frame_b, flow)
            else:
                logits = model(frame_a, frame_b)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels)

            done += 1
            if max_batches is not None and done >= max_batches:
                break

    return np.concatenate(all_logits), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def write_sweep_json(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_sweep_markdown(records: List[Dict], path: Path, best: Dict) -> None:
    lines = [
        "# Phase 3 Threshold Sweep",
        "",
        f"Best balanced accuracy: **{best['balanced_accuracy']:.4f}** at "
        f"threshold **{best['threshold']:.2f}**  "
        f"(F1={best['f1']:.4f}, TPR={best['tpr']:.4f}, TNR={best['tnr']:.4f})",
        "",
        "| threshold | bal_acc | f1 | tpr | tnr |",
        "| --------- | ------- | -- | --- | --- |",
    ]
    # print every 5th step for readability
    for r in records[::5]:
        lines.append(
            f"| {r['threshold']:.2f} "
            f"| {r['balanced_accuracy']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['tpr']:.4f} "
            f"| {r['tnr']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_plot(records: List[Dict], path: Path) -> None:
    if _plt is None:
        return
    thresholds  = [r["threshold"]         for r in records]
    bal_accs    = [r["balanced_accuracy"]  for r in records]
    f1s         = [r["f1"]                 for r in records]
    tprs        = [r["tpr"]               for r in records]
    tnrs        = [r["tnr"]               for r in records]

    fig, ax = _plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, bal_accs, label="Balanced Accuracy", linewidth=2)
    ax.plot(thresholds, f1s,      label="F1",                linewidth=2)
    ax.plot(thresholds, tprs,     label="TPR (Recall)",      linestyle="--")
    ax.plot(thresholds, tnrs,     label="TNR (Specificity)", linestyle="--")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default (0.5)")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Phase 3 Threshold Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _plt.close(fig)
    print(f"  Saved sweep plot → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _resolve_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    import sys
    from pathlib import Path as _Path

    PROJECT_ROOT = _Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from data.celeba_loader import create_celeba_dataloader, load_config
    from models.discriminator import DiscriminatorPhase3, load_phase3_checkpoint

    parser = argparse.ArgumentParser(description="Threshold sweep on Phase 3 checkpoint")
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (default: from config)")
    parser.add_argument("--run-dir",    default="runs/threshold_sweep_phase3")
    parser.add_argument("--split",      default="test", choices=("train", "val", "test"))
    parser.add_argument("--steps",      type=int, default=99)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--device",     default="cpu", choices=("mps", "cuda", "cpu"))
    args = parser.parse_args()

    device     = _resolve_device(args.device)
    config     = load_config(args.config)
    run_dir    = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    phase3_cfg      = config["phase3"]  # type: ignore[index]
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])  # type: ignore[index]
    ckpt_path       = Path(args.checkpoint) if args.checkpoint else (
        checkpoints_dir / phase3_cfg["checkpoint_name"]  # type: ignore[index]
    )

    print(f"Loading Phase 3 checkpoint: {ckpt_path}")
    model = DiscriminatorPhase3(dropout=float(phase3_cfg["dropout"]))  # type: ignore[index]
    load_phase3_checkpoint(model, None, None, ckpt_path)

    print(f"Building '{args.split}' dataloader (adjacent_cache)…")
    dl_overrides: dict = dict(config["dataloader"])  # type: ignore[index]
    dl_overrides["num_workers"] = 0
    dl_overrides.pop("prefetch_factor", None)

    dataloader = create_celeba_dataloader(
        config,
        split=args.split,
        shuffle=False,
        limit=args.limit,
        include_flow=True,
        pairing_mode="adjacent_cache",
        dataloader_overrides=dl_overrides,
    )

    print("Collecting logits…")
    logits, labels = collect_logits_and_labels(
        model, dataloader, device, include_flow=True
    )

    # balance classes so threshold sweep isn't distorted by class imbalance
    real_idx = np.flatnonzero(labels == 0)
    fake_idx = np.flatnonzero(labels == 1)
    n = min(len(real_idx), len(fake_idx))
    idx = np.sort(np.concatenate([real_idx[:n], fake_idx[:n]]))
    logits = logits[idx]
    labels = labels[idx]
    print(f"  Using {len(labels)} balanced examples (real={n}, fake={n})")

    print(f"Sweeping {args.steps} thresholds…")
    records = sweep_thresholds(logits, labels, steps=args.steps)

    best_ba = best_threshold_by(records, "balanced_accuracy")
    best_f1 = best_threshold_by(records, "f1")

    print(f"\n  Best balanced accuracy: {best_ba['balanced_accuracy']:.4f} "
          f"@ threshold={best_ba['threshold']:.2f}  "
          f"(F1={best_ba['f1']:.4f}, TPR={best_ba['tpr']:.4f}, TNR={best_ba['tnr']:.4f})")
    print(f"  Best F1:                {best_f1['f1']:.4f} "
          f"@ threshold={best_f1['threshold']:.2f}  "
          f"(bal_acc={best_f1['balanced_accuracy']:.4f}, "
          f"TPR={best_f1['tpr']:.4f}, TNR={best_f1['tnr']:.4f})")

    write_sweep_json(records,                       run_dir / "threshold_sweep.json")
    write_sweep_markdown(records, run_dir / "threshold_sweep.md", best_ba)
    write_sweep_plot(records,                       run_dir / "threshold_sweep.png")

    print(f"\nOutputs written to: {run_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "best_threshold_by",
    "collect_logits_and_labels",
    "sweep_thresholds",
    "write_sweep_json",
    "write_sweep_markdown",
    "write_sweep_plot",
]
