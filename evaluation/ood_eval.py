"""Week 4 OOD evaluation on the four forensics GAN datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.celeba_loader import load_config
from data.augmentations import build_transforms
from data.forensics_loader import (
    ForensicsPairingMode,
    create_forensics_dataloader,
    discover_forensics_datasets,
    normalize_split,
    resolve_forensics_root,
)
from evaluation.ensemble import (
    COMBO_CONFIGS,
    BranchFeatures,
    build_combo_features,
    train_logistic_ensemble,
    train_rf_ensemble,
)
from evaluation.eval import (
    compute_auc_roc,
    compute_balanced_accuracy,
    compute_f1,
    plot_confusion_matrix,
)
from evaluation.feature_cache import load_feature_cache
from models.branch_a import BranchABaseline
from models.discriminator import (
    DiscriminatorPhase3,
    DiscriminatorPhase4,
    load_phase3_checkpoint,
    load_phase3_into_phase4,
)


DEFAULT_THRESHOLD = 0.61
DEFAULT_CELEBA_FEATURES = Path("runs/celeba_features/phase3_train_adjacent_cache.npz")


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def evaluate_ood_neural(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, Any]:
    probabilities = stable_sigmoid(logits)
    default_predictions = (probabilities >= 0.5).astype(np.int64)
    tuned_predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "default_threshold": 0.5,
        "default_metrics": _metrics_from_predictions(labels, default_predictions, probabilities),
        "threshold": float(threshold),
        "metrics": _metrics_from_predictions(labels, tuned_predictions, probabilities),
    }


def evaluate_ood_ensemble(
    train_features: BranchFeatures,
    train_labels: np.ndarray,
    test_features: BranchFeatures,
    test_labels: np.ndarray,
    *,
    output_dir: Path,
    combo_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(combo_keys) if combo_keys else {combo.key for combo in COMBO_CONFIGS}
    results: Dict[str, Dict[str, Any]] = {}
    for combo in COMBO_CONFIGS:
        if combo.key not in requested:
            continue
        x_train = build_combo_features(train_features, combo.branches)
        x_test = build_combo_features(test_features, combo.branches)
        clf = (
            train_rf_ensemble(x_train, train_labels)
            if combo.classifier == "rf"
            else train_logistic_ensemble(x_train, train_labels)
        )
        predictions = clf.predict(x_test)
        scores = _classifier_scores(clf, x_test)
        cm_path = output_dir / f"{combo.key}_confusion_matrix.png"
        plot_confusion_matrix(test_labels, predictions, cm_path)
        results[combo.key] = {
            "label": combo.label,
            "branches": list(combo.branches),
            "classifier": combo.classifier,
            "metrics": _metrics_from_predictions(test_labels, predictions, scores),
            "confusion_matrix_png": str(cm_path),
            "predictions": predictions,
            "scores": scores,
        }
    return results


def run_forensics_eval(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    checkpoint: Path,
    forensics_root: Path,
    dataset: Optional[str] = None,
    split: str = "test",
    pairing: ForensicsPairingMode = "adjacent_same_class",
    mode: str = "all",
    threshold: Optional[float] = None,
    device: torch.device,
    celeba_features: Path = DEFAULT_CELEBA_FEATURES,
    phase4_checkpoint: Optional[Path] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    limit: Optional[int] = None,
    max_batches: Optional[int] = None,
    run_in_domain: bool = False,
    branch_a_checkpoint: Optional[Path] = None,
    branch_a_diagnostic: bool = False,
) -> Dict[str, Any]:
    """Run forensics OOD evaluation and return the summary written to disk."""
    start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    split = normalize_split(split)
    resolved_threshold = threshold if threshold is not None else _load_default_threshold()
    datasets = _select_datasets(forensics_root, dataset)

    model = _load_phase3_model(config, checkpoint, device)
    phase4_model = (
        _load_phase4_model(config, phase4_checkpoint, device)
        if phase4_checkpoint is not None and phase4_checkpoint.exists()
        else None
    )
    transfer_features: Optional[BranchFeatures] = None
    transfer_labels: Optional[np.ndarray] = None
    transfer_metadata: dict[str, Any] = {}
    if mode in {"all", "ensemble"} and celeba_features.exists():
        transfer_features, transfer_labels, transfer_metadata = load_feature_cache(celeba_features)

    dataset_records = []
    pooled_parts: list[tuple[BranchFeatures, np.ndarray, np.ndarray, list[str]]] = []

    for dataset_root in datasets:
        dataset_key = _dataset_key(dataset_root)
        dataset_dir = run_dir / "per_dataset" / dataset_key
        loader = create_forensics_dataloader(
            dataset_root,
            split=split,
            pairing_mode=pairing,
            batch_size=batch_size,
            num_workers=num_workers,
            limit=limit,
            shuffle=False,
        )
        features, labels, paths = _extract_features_with_paths(
            model,
            loader,
            device,
            desc=f"{dataset_root.parent.name} {split}",
            max_batches=max_batches,
        )
        logits = features["logit"]
        record: Dict[str, Any] = {
            "dataset": dataset_root.parent.name if dataset_root.name == dataset_root.parent.name else dataset_root.name,
            "dataset_root": str(dataset_root),
            "split": split,
            "pairing_mode": pairing,
            "checkpoint": str(checkpoint),
            "n_images": int(labels.shape[0]),
            "class_counts": _class_counts(labels),
        }

        if mode in {"all", "neural"}:
            neural = evaluate_ood_neural(logits, labels, threshold=resolved_threshold)
            record["neural"] = neural
            neural_csv = dataset_dir / "phase3_per_image_scores.csv"
            _write_neural_csv(neural_csv, paths, labels, logits, resolved_threshold)
            record["per_image_scores_csv"] = str(neural_csv)

        if phase4_model is not None and mode in {"all", "neural"}:
            phase4_logits, phase4_labels, _ = _collect_logits_with_paths(
                phase4_model,
                loader,
                device,
                desc=f"phase4 {dataset_key}",
                max_batches=max_batches,
            )
            record["phase4_neural"] = evaluate_ood_neural(
                phase4_logits,
                phase4_labels,
                threshold=resolved_threshold,
            )

        if mode in {"all", "ensemble"}:
            if transfer_features is not None and transfer_labels is not None:
                transfer_results = evaluate_ood_ensemble(
                    transfer_features,
                    transfer_labels,
                    features,
                    labels,
                    output_dir=dataset_dir,
                )
                record["ensemble_combos"] = _strip_runtime_arrays(transfer_results)
                ensemble_csv = dataset_dir / "ensemble_per_image_scores.csv"
                _write_ensemble_csv(ensemble_csv, paths, labels, transfer_results)
                record["ensemble_per_image_scores_csv"] = str(ensemble_csv)
            else:
                record["ensemble_error"] = f"CelebA feature cache missing: {celeba_features}"

            if run_in_domain:
                train_loader = create_forensics_dataloader(
                    dataset_root,
                    split="train",
                    pairing_mode=pairing,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    limit=limit,
                    shuffle=False,
                )
                train_features, train_labels, _ = _extract_features_with_paths(
                    model,
                    train_loader,
                    device,
                    desc=f"{dataset_key} train features",
                    max_batches=max_batches,
                )
                in_domain = evaluate_ood_ensemble(
                    train_features,
                    train_labels,
                    features,
                    labels,
                    output_dir=dataset_dir / "in_domain",
                )
                record["in_domain_ensemble_combos"] = _strip_runtime_arrays(in_domain)

        if branch_a_diagnostic and branch_a_checkpoint is not None:
            record["branch_a_pair_diagnostic"] = evaluate_branch_a_pairs(
                branch_a_checkpoint,
                dataset_root,
                split=split,
                device=device,
                limit=limit,
            )

        dataset_records.append(record)
        pooled_parts.append((features, labels, logits, paths))

    pooled_summary = _build_pooled_summary(
        pooled_parts,
        threshold=resolved_threshold,
        transfer_features=transfer_features,
        transfer_labels=transfer_labels,
        run_dir=run_dir,
        mode=mode,
    )
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "forensics_root": str(forensics_root),
        "split": split,
        "pairing_mode": pairing,
        "device": str(device),
        "threshold": resolved_threshold,
        "checkpoint": str(checkpoint),
        "phase4_checkpoint": str(phase4_checkpoint) if phase4_checkpoint else None,
        "celeba_feature_cache": str(celeba_features),
        "celeba_feature_cache_metadata": transfer_metadata,
        "datasets": dataset_records,
        "pooled": pooled_summary,
        "duration_s": round(time.perf_counter() - start, 3),
        "handoff_note": (
            "Dev 2 should consume summary.json/per_dataset CSV artifacts for now; "
            "a stable evaluate_ood(...) import wrapper is intentionally deferred."
        ),
    }
    _write_summary(run_dir, summary)
    return summary


def write_forensics_summary(run_dir: Path) -> None:
    summary_path = Path(run_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _write_summary_markdown(Path(run_dir) / "summary.md", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3/4 on forensics OOD datasets")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/phase3_a_b_c.pt")
    parser.add_argument("--phase4-checkpoint", default=None)
    parser.add_argument("--forensics-root", default="data/forensics")
    parser.add_argument("--dataset", default=None, help='Optional single dataset, e.g. "Data Set 2"')
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--run-dir", default="runs/forensics_eval")
    parser.add_argument("--pairing", default="adjacent_same_class", choices=("adjacent_same_class", "degenerate"))
    parser.add_argument("--mode", default="all", choices=("neural", "ensemble", "all"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--celeba-features", default=str(DEFAULT_CELEBA_FEATURES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-domain", action="store_true", help="Also train diagnostic RFs on forensics train split")
    parser.add_argument("--branch-a-diagnostic", action="store_true", help="Run pair-labelled Branch A diagnostic")
    parser.add_argument("--branch-a-checkpoint", default="checkpoints/phase1_branch_a_best.pt")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    summary = run_forensics_eval(
        config,
        run_dir=Path(args.run_dir),
        checkpoint=Path(args.checkpoint),
        phase4_checkpoint=Path(args.phase4_checkpoint) if args.phase4_checkpoint else None,
        forensics_root=Path(args.forensics_root),
        dataset=args.dataset,
        split=args.split,
        pairing=args.pairing,
        mode=args.mode,
        threshold=args.threshold,
        device=_resolve_device(args.device),
        celeba_features=Path(args.celeba_features),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        max_batches=args.max_batches,
        run_in_domain=args.in_domain,
        branch_a_checkpoint=Path(args.branch_a_checkpoint),
        branch_a_diagnostic=args.branch_a_diagnostic,
    )
    print(f"Wrote {summary['run_dir']}/summary.json")


def evaluate_branch_a_pairs(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    *,
    split: str = "test",
    device: torch.device,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the pair-labelled Branch A diagnostic on a forensics dataset.

    This is not the same task as GAN single-image real/fake classification:
    positive pairs are cross real/fake and negative pairs are same-domain.
    """
    root = resolve_forensics_root(dataset_root)
    real_images = _class_image_paths(root, split, "real")
    fake_images = _class_image_paths(root, split, "fake")
    positive_count = min(len(real_images), len(fake_images))
    positive_pairs = list(zip(real_images[:positive_count], fake_images[:positive_count]))
    negative_pairs = _same_domain_negative_pairs(real_images, fake_images, positive_count)
    if limit is not None:
        positive_pairs = positive_pairs[:limit]
        negative_pairs = negative_pairs[:limit]
    pair_count = min(len(positive_pairs), len(negative_pairs))
    if pair_count == 0:
        raise ValueError("Branch A diagnostic requires real/fake images and same-domain pairs")
    positive_pairs = positive_pairs[:pair_count]
    negative_pairs = negative_pairs[:pair_count]

    model = _load_branch_a_checkpoint(Path(checkpoint_path), device)
    transform = build_transforms(image_size=64, train=False)
    logits: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for (frame_a_path, frame_b_path), label in (
            [(pair, 1) for pair in positive_pairs] + [(pair, 0) for pair in negative_pairs]
        ):
            frame_a = _load_image_tensor(frame_a_path, transform, device)
            frame_b = _load_image_tensor(frame_b_path, transform, device)
            logits.append(float(model(frame_a, frame_b).item()))
            labels.append(label)
    logits_array = np.asarray(logits, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities = stable_sigmoid(logits_array)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "warning": (
            "Diagnostic only: positive pairs are cross real/fake images and "
            "negative pairs are same-domain pairs, not GAN single-image labels."
        ),
        "n_pairs": int(labels_array.shape[0]),
        "metrics": _metrics_from_predictions(labels_array, predictions, probabilities),
        "positive_rate": float(predictions.mean()),
        "mean_probability": float(probabilities.mean()),
        "positive_pair_mean_probability": float(probabilities[labels_array == 1].mean()),
        "negative_pair_mean_probability": float(probabilities[labels_array == 0].mean()),
    }


def _extract_features_with_paths(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str,
    max_batches: Optional[int],
) -> tuple[BranchFeatures, np.ndarray, list[str]]:
    model.eval().to(device)
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    path_list: list[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            frame_a = batch["frame_a"].to(device)
            frame_b = batch["frame_b"].to(device)
            flow = batch["flow"].to(device)
            output = model.forward_with_branch_features(frame_a, frame_b, flow)
            a_list.append(output["a"].cpu().numpy())
            b_list.append(output["b"].cpu().numpy())
            c_list.append(output["c"].cpu().numpy())
            logit_list.append(output["logit"].cpu().numpy())
            label_list.append(batch["label"].cpu().numpy().astype(np.int64))
            path_list.extend(str(path) for path in batch["path_a"])
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return (
        {
            "a": np.concatenate(a_list, axis=0),
            "b": np.concatenate(b_list, axis=0),
            "c": np.concatenate(c_list, axis=0),
            "logit": np.concatenate(logit_list, axis=0),
        },
        np.concatenate(label_list, axis=0),
        path_list,
    )


def _collect_logits_with_paths(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str,
    max_batches: Optional[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features, labels, paths = _extract_features_with_paths(
        model,
        dataloader,
        device,
        desc=desc,
        max_batches=max_batches,
    )
    return features["logit"], labels, paths


def _load_branch_a_checkpoint(checkpoint_path: Path, device: torch.device) -> BranchABaseline:
    model = BranchABaseline().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Branch A checkpoint must contain a model_state_dict mapping")
    model.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()})
    return model.eval()


def _load_image_tensor(path: Path, transform: Any, device: torch.device) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0).to(device)


def _class_image_paths(dataset_root: Path, split: str, class_name: str) -> list[Path]:
    class_dir = dataset_root / normalize_split(split) / class_name
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _same_domain_negative_pairs(
    real_images: Sequence[Path],
    fake_images: Sequence[Path],
    target_count: int,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for images in (real_images, fake_images):
        for index in range(0, len(images) - 1, 2):
            pairs.append((images[index], images[index + 1]))
            if len(pairs) >= target_count:
                return pairs
    return pairs


def _build_pooled_summary(
    pooled_parts: Iterable[tuple[BranchFeatures, np.ndarray, np.ndarray, list[str]]],
    *,
    threshold: float,
    transfer_features: Optional[BranchFeatures],
    transfer_labels: Optional[np.ndarray],
    run_dir: Path,
    mode: str,
) -> Dict[str, Any]:
    parts = list(pooled_parts)
    if not parts:
        return {}
    features: BranchFeatures = {
        key: np.concatenate([item[0][key] for item in parts], axis=0)
        for key in ("a", "b", "c", "logit")
    }
    labels = np.concatenate([item[1] for item in parts], axis=0)
    pooled: Dict[str, Any] = {"n_images": int(labels.shape[0]), "class_counts": _class_counts(labels)}
    if mode in {"all", "neural"}:
        pooled["neural"] = evaluate_ood_neural(features["logit"], labels, threshold=threshold)
    if mode in {"all", "ensemble"} and transfer_features is not None and transfer_labels is not None:
        ensemble = evaluate_ood_ensemble(
            transfer_features,
            transfer_labels,
            features,
            labels,
            output_dir=run_dir / "pooled",
        )
        pooled["ensemble_combos"] = _strip_runtime_arrays(ensemble)
    return pooled


def _write_neural_csv(path: Path, paths: Sequence[str], labels: np.ndarray, logits: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = stable_sigmoid(logits)
    predictions = (probabilities >= threshold).astype(np.int64)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "logit", "prob", "pred"])
        writer.writeheader()
        for image_path, label, logit, probability, prediction in zip(paths, labels, logits, probabilities, predictions):
            writer.writerow({
                "path": image_path,
                "label": int(label),
                "logit": float(logit),
                "prob": float(probability),
                "pred": int(prediction),
            })


def _write_ensemble_csv(
    path: Path,
    paths: Sequence[str],
    labels: np.ndarray,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label"]
    for key in results:
        fieldnames.extend([f"{key}_score", f"{key}_pred"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, image_path in enumerate(paths):
            row: dict[str, Any] = {"path": image_path, "label": int(labels[row_index])}
            for key, result in results.items():
                row[f"{key}_score"] = float(result["scores"][row_index])
                row[f"{key}_pred"] = int(result["predictions"][row_index])
            writer.writerow(row)


def _write_summary(run_dir: Path, summary: Mapping[str, Any]) -> None:
    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_markdown(run_dir / "summary.md", summary)


def _write_summary_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Week 4 Forensics OOD Evaluation",
        "",
        f"Split: `{summary['split']}` | Pairing: `{summary['pairing_mode']}` | Threshold: `{summary['threshold']}`",
        "",
        "## Neural Phase 3",
        "",
        "| Dataset | N | Bal Acc @0.5 | Bal Acc @threshold | F1 | AUC-ROC |",
        "| ------- | -: | -----------: | -----------------: | --: | ------: |",
    ]
    for record in summary.get("datasets", []):
        neural = record.get("neural", {})
        default_metrics = neural.get("default_metrics", {})
        metrics = neural.get("metrics", {})
        lines.append(
            f"| {record.get('dataset')} | {record.get('n_images')} "
            f"| {_fmt(default_metrics.get('balanced_accuracy'))} "
            f"| {_fmt(metrics.get('balanced_accuracy'))} "
            f"| {_fmt(metrics.get('f1'))} "
            f"| {_fmt(metrics.get('auc_roc'))} |"
        )
    pooled = summary.get("pooled", {})
    if pooled.get("neural"):
        neural = pooled["neural"]
        lines.append(
            f"| pooled | {pooled.get('n_images')} "
            f"| {_fmt(neural['default_metrics'].get('balanced_accuracy'))} "
            f"| {_fmt(neural['metrics'].get('balanced_accuracy'))} "
            f"| {_fmt(neural['metrics'].get('f1'))} "
            f"| {_fmt(neural['metrics'].get('auc_roc'))} |"
        )
    lines += [
        "",
        "## Transfer Ensemble",
        "",
        "| Dataset | A | B | C | A+B | A+C | B+C | A+B+C |",
        "| ------- | --: | --: | --: | --: | --: | --: | ----: |",
    ]
    for record in summary.get("datasets", []):
        combos = record.get("ensemble_combos", {})
        lines.append(_combo_row(str(record.get("dataset")), combos))
    if pooled.get("ensemble_combos"):
        lines.append(_combo_row("pooled", pooled["ensemble_combos"]))
    path.write_text("\n".join(lines), encoding="utf-8")


def _combo_row(name: str, combos: Mapping[str, Any]) -> str:
    keys = ("a", "b", "c", "a_b", "a_c", "b_c", "a_b_c")
    values = [_fmt(combos.get(key, {}).get("metrics", {}).get("balanced_accuracy")) for key in keys]
    return f"| {name} | " + " | ".join(values) + " |"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "balanced_accuracy": compute_balanced_accuracy(labels, predictions),
        "f1": compute_f1(labels, predictions),
        "auc_roc": compute_auc_roc(labels, scores),
    }


def _classifier_scores(clf: Any, x_values: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(x_values)
        return probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]
    return clf.decision_function(x_values)


def _strip_runtime_arrays(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stripped: Dict[str, Dict[str, Any]] = {}
    for key, value in results.items():
        stripped[key] = {
            item_key: item_value
            for item_key, item_value in value.items()
            if item_key not in {"predictions", "scores"}
        }
    return stripped


def _class_counts(labels: np.ndarray) -> Dict[str, int]:
    return {"real": int((labels == 0).sum()), "fake": int((labels == 1).sum())}


def _select_datasets(forensics_root: Path, dataset: Optional[str]) -> list[Path]:
    if dataset:
        candidate = forensics_root / dataset
        if candidate.exists():
            return [resolve_forensics_root(candidate)]
        return [resolve_forensics_root(Path(dataset))]
    return discover_forensics_datasets(forensics_root)


def _dataset_key(dataset_root: Path) -> str:
    name = dataset_root.name if dataset_root.name != dataset_root.parent.name else dataset_root.parent.name
    return name.lower().replace(" ", "_")


def _load_default_threshold() -> float:
    sweep_path = Path("runs/ensemble_ablation/threshold_sweep.json")
    if not sweep_path.exists():
        return DEFAULT_THRESHOLD
    try:
        records = json.loads(sweep_path.read_text(encoding="utf-8"))
        best = max(records, key=lambda row: row["balanced_accuracy"])
        return float(best["threshold"])
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_THRESHOLD


def _load_phase3_model(config: Mapping[str, Any], checkpoint: Path, device: torch.device) -> DiscriminatorPhase3:
    phase3_cfg = dict(config.get("phase3", {}))
    model = DiscriminatorPhase3(dropout=float(phase3_cfg.get("dropout", 0.3)))
    load_phase3_checkpoint(model, None, None, checkpoint)
    return model.eval().to(device)


def _load_phase4_model(config: Mapping[str, Any], checkpoint: Path, device: torch.device) -> DiscriminatorPhase4:
    phase4_cfg = dict(config.get("phase4", {}))
    model = DiscriminatorPhase4(dropout=float(phase4_cfg.get("dropout", 0.3)))
    load_phase3_into_phase4(model, checkpoint)
    return model.eval().to(device)


def _resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"mps", "cuda"}:
        print(f"WARNING: {name} unavailable, falling back to CPU")
    return torch.device("cpu")


if __name__ == "__main__":
    main(sys.argv[1:])
