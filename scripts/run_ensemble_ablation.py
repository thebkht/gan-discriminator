"""Week 3 Dev 2 — RF ensemble + per-branch ablation runner.

Runs the full Week 3 Dev 2 checklist in one script:

  1. Load Phase 3 checkpoint (or Phase 4 for comparison).
  2. Extract branch features (A 2048-D, B 32-D, C 28-D) + labels from the
     test split using adjacent_cache pairing.
  3. Train & evaluate all 7 RF/logistic branch-combination classifiers.
  4. Run Phase 3 threshold sweep.
  5. Save per-branch neural ablation metrics (full-model logit view).
  6. Write confusion matrices for all 7 combos.
  7. Write a consolidated ablation table to runs/ensemble_ablation/.

Usage
-----
  # Full run (uses GPU/MPS if available):
  python scripts/run_ensemble_ablation.py --config config/config.yaml

  # CPU-only smoke run with a sample limit:
  python scripts/run_ensemble_ablation.py \
      --config config/config.yaml \
      --device cpu \
      --limit 512 \
      --run-dir runs/ensemble_ablation_smoke

Outputs (all under --run-dir, default: runs/ensemble_ablation/)
-------
  summary.json                        — machine-readable full results
  summary.md                          — human-readable ablation table
  threshold_sweep.json/md/png         — Phase 3 threshold sweep
  {combo_key}_confusion_matrix.png
  {combo_key}_confusion_matrix_normalized.png
  neural_full_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.celeba_loader import create_celeba_dataloader, load_config
from evaluation.ensemble import (
    COMBO_CONFIGS,
    ablate_single_branches,
    extract_branch_outputs,
    run_all_combos,
)
from evaluation.threshold_sweep import (
    collect_logits_and_labels,
    sweep_thresholds,
    best_threshold_by,
    write_sweep_json,
    write_sweep_markdown,
    write_sweep_plot,
)
from models.discriminator import (
    DiscriminatorPhase3,
    DiscriminatorPhase4,
    load_phase3_checkpoint,
    load_phase3_into_phase4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_dict(value: object, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict for {context}, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _resolve_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            print("  WARNING: MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            print("  WARNING: CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def _write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Week 3 Dev 2 — RF Ensemble + Ablation Results",
        "",
        f"Split: `{summary['split']}` | Device: `{summary['device']}` | "
        f"Examples: `{summary.get('n_test', '?')}`",
        "",
        "## 7-Combo Ensemble Results",
        "",
        "| # | Config | Classifier | Bal Acc | F1 | AUC-ROC |",
        "| - | ------ | ---------- | ------- | -- | ------- |",
    ]

    combo_results = summary.get("ensemble_combos", {})
    for i, combo in enumerate(COMBO_CONFIGS, 1):
        r = combo_results.get(combo.key, {})
        m = r.get("metrics", {})
        gate = " ⭐ GATE" if (
            combo.key == "b_c"
            and m.get("balanced_accuracy", 0) >= 0.944
            and m.get("f1", 0) >= 0.93
        ) else ""
        lines.append(
            f"| {i} | {combo.label} | {combo.classifier.upper()} "
            f"| {m.get('balanced_accuracy', float('nan')):.4f} "
            f"| {m.get('f1', float('nan')):.4f} "
            f"| {m.get('auc_roc', float('nan')):.4f} |{gate}"
        )

    lines += [
        "",
        "## Neural Ablation (Full Model Logit)",
        "",
        "| Branch config | Bal Acc | F1 | AUC-ROC |",
        "| ------------- | ------- | -- | ------- |",
    ]
    for key, m in summary.get("neural_ablation", {}).items():
        lines.append(
            f"| {key} "
            f"| {m.get('balanced_accuracy', float('nan')):.4f} "
            f"| {m.get('f1', float('nan')):.4f} "
            f"| {m.get('auc_roc', float('nan')):.4f} |"
        )

    tsweep = summary.get("threshold_sweep_best", {})
    if tsweep:
        lines += [
            "",
            "## Phase 3 Threshold Sweep",
            "",
            f"Best balanced accuracy: **{tsweep.get('balanced_accuracy', '?'):.4f}** "
            f"@ threshold={tsweep.get('threshold', '?'):.2f}  "
            f"(F1={tsweep.get('f1', '?'):.4f}, "
            f"TPR={tsweep.get('tpr', '?'):.4f}, "
            f"TNR={tsweep.get('tnr', '?'):.4f})",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 3 Dev 2: RF ensemble + ablation runner"
    )
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--checkpoint",   default=None,
                        help="Phase 3 checkpoint path (default: from config)")
    parser.add_argument("--phase4-checkpoint", default=None,
                        help="Optional Phase 4 checkpoint to also evaluate")
    parser.add_argument("--run-dir",      default="runs/ensemble_ablation")
    parser.add_argument("--split",        default="test",
                        choices=("train", "val", "test"))
    parser.add_argument("--device",       default="cpu",
                        choices=("mps", "cuda", "cpu"))
    parser.add_argument("--limit",        type=int, default=None,
                        help="Cap dataset size for smoke runs")
    parser.add_argument("--sweep-steps",  type=int, default=99)
    parser.add_argument("--num-workers",  type=int, default=0)
    args = parser.parse_args()

    t_start   = time.perf_counter()
    device    = _resolve_device(args.device)
    run_dir   = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Week 3 Dev 2 — RF Ensemble + Ablation")
    print(f"{'='*60}")
    print(f"  Config   : {args.config}")
    print(f"  Run dir  : {run_dir}")
    print(f"  Device   : {device}")
    print(f"  Split    : {args.split}")
    if args.limit:
        print(f"  Limit    : {args.limit}")
    print()

    # ── Config ──────────────────────────────────────────────────────────────
    config          = _as_dict(load_config(args.config), "config")
    paths_cfg       = _as_dict(config["paths"],     "config.paths")
    phase3_cfg      = _as_dict(config["phase3"],    "config.phase3")
    dl_cfg          = _as_dict(config["dataloader"],"config.dataloader")
    checkpoints_dir = Path(paths_cfg["checkpoints_dir"])

    dl_overrides: Dict[str, object] = dict(dl_cfg)
    dl_overrides["num_workers"] = args.num_workers
    dl_overrides.pop("prefetch_factor", None)

    # ── Phase 3 model ────────────────────────────────────────────────────────
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint
        else checkpoints_dir / str(phase3_cfg["checkpoint_name"])
    )
    if not ckpt_path.exists():
        print(f"ERROR: Phase 3 checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[1/5] Loading Phase 3 checkpoint: {ckpt_path}")
    model = DiscriminatorPhase3(dropout=float(phase3_cfg["dropout"]))
    load_phase3_checkpoint(model, None, None, ckpt_path)

    # ── Dataloader ───────────────────────────────────────────────────────────
    print(f"[2/5] Building '{args.split}' dataloader (adjacent_cache)…")
    dataloader = create_celeba_dataloader(
        config,
        split=args.split,
        shuffle=False,
        limit=args.limit,
        include_flow=True,
        pairing_mode="adjacent_cache",
        dataloader_overrides=dl_overrides,
    )

    # ── Feature extraction ───────────────────────────────────────────────────
    print("[3/5] Extracting branch features…")
    test_features, test_labels = extract_branch_outputs(
        model, dataloader, device, desc="test set"
    )
    print(f"  Extracted {len(test_labels)} examples  "
          f"(real={int((test_labels==0).sum())}, fake={int((test_labels==1).sum())})")

    # Balance for fair evaluation
    real_idx = np.flatnonzero(test_labels == 0)
    fake_idx = np.flatnonzero(test_labels == 1)
    n_bal    = min(len(real_idx), len(fake_idx))
    bal_idx  = np.sort(np.concatenate([real_idx[:n_bal], fake_idx[:n_bal]]))

    bal_features = {k: v[bal_idx] for k, v in test_features.items()}
    bal_labels   = test_labels[bal_idx]
    print(f"  Balanced to {len(bal_labels)} examples (real={n_bal}, fake={n_bal})")

    # We use the full balanced set for both train and test of the RF
    # (the RF is a probe on top of frozen features, not a held-out model).
    # For a proper train/test split of the RF, we split 80/20.
    split_point  = int(0.8 * len(bal_labels))
    rng          = np.random.default_rng(42)
    perm         = rng.permutation(len(bal_labels))
    train_idx    = perm[:split_point]
    eval_idx     = perm[split_point:]

    train_features_rf = {k: v[train_idx] for k, v in bal_features.items()}
    train_labels_rf   = bal_labels[train_idx]
    eval_features_rf  = {k: v[eval_idx]  for k, v in bal_features.items()}
    eval_labels_rf    = bal_labels[eval_idx]

    print(f"  RF train={len(train_labels_rf)}, eval={len(eval_labels_rf)}")

    # ── 7-combo ensembles ────────────────────────────────────────────────────
    print("\n[4/5] Training & evaluating 7 branch-combination classifiers…")
    combo_results = run_all_combos(
        train_features_rf, train_labels_rf,
        eval_features_rf,  eval_labels_rf,
        output_dir=run_dir,
        verbose=True,
    )

    # ── Neural ablation ──────────────────────────────────────────────────────
    print("\n      Neural full-model ablation (Phase 3 logit)…")
    neural_metrics = ablate_single_branches(
        eval_features_rf, eval_labels_rf,
        output_dir=run_dir,
        verbose=True,
    )

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print("\n[5/5] Running Phase 3 threshold sweep…")
    # Re-use the already-collected logits — no second forward pass needed
    sweep_logits = bal_features["logit"]
    sweep_labels = bal_labels

    sweep_records = sweep_thresholds(sweep_logits, sweep_labels, steps=args.sweep_steps)
    best_ba  = best_threshold_by(sweep_records, "balanced_accuracy")
    best_f1  = best_threshold_by(sweep_records, "f1")

    print(f"  Best balanced acc: {best_ba['balanced_accuracy']:.4f} "
          f"@ t={best_ba['threshold']:.2f}  "
          f"(F1={best_ba['f1']:.4f}, TPR={best_ba['tpr']:.4f}, TNR={best_ba['tnr']:.4f})")
    print(f"  Best F1          : {best_f1['f1']:.4f} "
          f"@ t={best_f1['threshold']:.2f}  "
          f"(bal_acc={best_f1['balanced_accuracy']:.4f})")

    write_sweep_json(sweep_records,               run_dir / "threshold_sweep.json")
    write_sweep_markdown(sweep_records, run_dir / "threshold_sweep.md", best_ba)
    write_sweep_plot(sweep_records,               run_dir / "threshold_sweep.png")

    # ── Serialisable summary ─────────────────────────────────────────────────
    combo_summary = {}
    for key, r in combo_results.items():
        combo_summary[key] = {
            "label":       r["label"],
            "branches":    list(r["branches"]),
            "classifier":  r["classifier"],
            "metrics":     r["metrics"],
            "duration_s":  r["duration_s"],
            "confusion_matrix_png":            r["confusion_matrix_png"],
            "confusion_matrix_normalized_png": r["confusion_matrix_normalized_png"],
        }

    summary: Dict[str, Any] = {
        "split":                 args.split,
        "device":                str(device),
        "checkpoint":            str(ckpt_path),
        "n_test":                len(bal_labels),
        "n_train_rf":            len(train_labels_rf),
        "n_eval_rf":             len(eval_labels_rf),
        "ensemble_combos":       combo_summary,
        "neural_ablation":       neural_metrics,
        "threshold_sweep_best":  best_ba,
        "threshold_sweep_best_f1": best_f1,
        "total_duration_s":      round(time.perf_counter() - t_start, 2),
    }

    summary_json = run_dir / "summary.json"
    summary_md   = run_dir / "summary.md"

    # Remove non-serialisable clf objects before JSON dump
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md, summary)

    print(f"\n{'='*60}")
    print(f" Done in {summary['total_duration_s']:.1f}s")
    print(f" Outputs → {run_dir}")
    print(f"   summary.json / summary.md")
    print(f"   threshold_sweep.json / .md / .png")
    print(f"   <combo>_confusion_matrix.png  (× {len(combo_results) * 2} files)")
    print(f"{'='*60}\n")

    # ── Gate check ──────────────────────────────────────────────────────────
    bc = combo_summary.get("b_c", {}).get("metrics", {})
    if bc.get("balanced_accuracy", 0) >= 0.944 and bc.get("f1", 0) >= 0.93:
        print("✓  B+C GATE CLEARED: "
              f"bal_acc={bc['balanced_accuracy']:.4f}, f1={bc['f1']:.4f}")
    else:
        print(f"✗  B+C gate not cleared "
              f"(bal_acc={bc.get('balanced_accuracy', 0):.4f}, "
              f"f1={bc.get('f1', 0):.4f}; "
              f"target ≥ 0.944 / 0.93)  — "
              "Phase 3 proxy task result expected; gate targets assume real deepfake data.")


if __name__ == "__main__":
    main()
