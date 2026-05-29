from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch
import numpy as np
from data.augmentations import build_transforms
from models.branch_a import BranchABaseline
from models import DiscriminatorPhase3, load_phase3_checkpoint
from evaluation import compute_binary_classification_metrics


def _load_image_tensor(path: Path, transform, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB")).unsqueeze(0).to(device)


def _load_branch_a_checkpoint(checkpoint_path: Path, device: torch.device) -> BranchABaseline:
    model = BranchABaseline().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Branch A checkpoint must contain a model_state_dict mapping")
    remapped_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(remapped_state_dict)
    model.eval()
    return model


def _image_paths(data_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    real_dir = data_root / split / "real"
    fake_dir = data_root / split / "fake"
    real_images = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
    fake_images = sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
    return real_images, fake_images


def _same_domain_negative_pairs(real_images: list[Path], fake_images: list[Path], target_count: int) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for images in (real_images, fake_images):
        for index in range(0, len(images) - 1, 2):
            pairs.append((images[index], images[index + 1]))
            if len(pairs) >= target_count:
                return pairs
    return pairs


def evaluate_forensics(checkpoint_path, data_root, split="test", device="cpu"):
    transform = build_transforms(image_size=64, train=False)
    device = torch.device(device)
    real_images, fake_images = _image_paths(Path(data_root), split)
    
    print(f"Real: {len(real_images)}, Fake: {len(fake_images)}")
    
    # load model
    model = DiscriminatorPhase3()
    load_phase3_checkpoint(model, None, None, Path(checkpoint_path))
    model.eval().to(device)
    
    logits, labels = [], []
    
    for path, label in [(p, 0) for p in real_images] + [(p, 1) for p in fake_images]:
        frame = _load_image_tensor(path, transform, device)
        # Phase 3 needs frame pair + flow — use same frame twice, zero flow
        flow = torch.zeros(1, 2, 64, 64).to(device)
        with torch.no_grad():
            logit = model(frame, frame, flow)
        logits.append(logit.item())
        labels.append(label)
    
    logits = np.array(logits)
    labels = np.array(labels)
    metrics = compute_binary_classification_metrics(logits=logits, labels=labels, average_loss=0.0)
    print(metrics)
    return metrics


def evaluate_branch_a_pairs(checkpoint_path, data_root, split="test", device="cpu"):
    transform = build_transforms(image_size=64, train=False)
    device = torch.device(device)
    real_images, fake_images = _image_paths(Path(data_root), split)
    positive_count = min(len(real_images), len(fake_images))
    positive_pairs = list(zip(real_images[:positive_count], fake_images[:positive_count]))
    negative_pairs = _same_domain_negative_pairs(real_images, fake_images, positive_count)

    if not positive_pairs:
        raise ValueError("Branch A pair eval requires at least one real image and one fake image")
    if not negative_pairs:
        raise ValueError("Branch A pair eval requires at least two same-domain images for negative pairs")
    pair_count = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:pair_count]
    negative_pairs = negative_pairs[:pair_count]

    print(
        "Branch A pair eval is diagnostic only: this checkpoint was trained on CelebA pair labels, "
        "not GAN artifact labels."
    )
    print(f"Real: {len(real_images)}, Fake: {len(fake_images)}")
    print(f"Positive real+fake pairs: {len(positive_pairs)}, Negative same-domain pairs: {len(negative_pairs)}")

    model = _load_branch_a_checkpoint(Path(checkpoint_path), device)
    logits, labels = [], []
    eval_pairs = [(pair, 1) for pair in positive_pairs] + [(pair, 0) for pair in negative_pairs]
    for (frame_a_path, frame_b_path), label in eval_pairs:
        frame_a = _load_image_tensor(frame_a_path, transform, device)
        frame_b = _load_image_tensor(frame_b_path, transform, device)
        with torch.no_grad():
            logit = model(frame_a, frame_b)
        logits.append(logit.item())
        labels.append(label)

    logits_array = np.array(logits)
    labels_array = np.array(labels)
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = compute_binary_classification_metrics(logits=logits_array, labels=labels_array, average_loss=0.0)
    metrics.update(
        {
            "positive_rate": float(predictions.mean()),
            "mean_probability": float(probabilities.mean()),
            "positive_pair_mean_probability": float(probabilities[labels_array == 1].mean()),
            "negative_pair_mean_probability": float(probabilities[labels_array == 0].mean()),
        }
    )
    print(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on a forensics split")
    parser.add_argument(
        "--mode",
        default="phase3-single",
        choices=("phase3-single", "branch-a-pairs"),
        help="Evaluation mode to run",
    )
    parser.add_argument("--checkpoint", default="checkpoints/phase3_a_b_c.pt", help="Path to a Phase 3 checkpoint")
    parser.add_argument(
        "--branch-a-checkpoint",
        default="checkpoints/phase1_branch_a_best.pt",
        help="Path to a Branch A checkpoint for --mode branch-a-pairs",
    )
    parser.add_argument("--data-root", required=True, help="Dataset root containing split/real and split/fake folders")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"), help="Torch device to run on")
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but MPS is not available")
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = build_parser().parse_args()
    device = _resolve_device(args.device)
    if args.mode == "branch-a-pairs":
        evaluate_branch_a_pairs(
            checkpoint_path=args.branch_a_checkpoint,
            data_root=args.data_root,
            split=args.split,
            device=device,
        )
    else:
        evaluate_forensics(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            split=args.split,
            device=device,
        )


if __name__ == "__main__":
    main()
