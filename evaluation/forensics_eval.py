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
from models import DiscriminatorPhase3, load_phase3_checkpoint
from evaluation import compute_binary_classification_metrics


def evaluate_forensics(checkpoint_path, data_root, split="test", device="cpu"):
    transform = build_transforms(image_size=64, train=False)
    
    real_dir = Path(data_root) / split / "real"
    fake_dir = Path(data_root) / split / "fake"
    
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    fake_images = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
    
    print(f"Real: {len(real_images)}, Fake: {len(fake_images)}")
    
    # load model
    model = DiscriminatorPhase3()
    load_phase3_checkpoint(model, None, None, Path(checkpoint_path))
    model.eval().to(device)
    
    logits, labels = [], []
    
    for path, label in [(p, 0) for p in real_images] + [(p, 1) for p in fake_images]:
        with Image.open(path) as img:
            frame = transform(img.convert("RGB")).unsqueeze(0).to(device)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 3 checkpoint on a forensics split")
    parser.add_argument("--checkpoint", required=True, help="Path to a Phase 3 checkpoint")
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
    evaluate_forensics(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        device=device,
    )


if __name__ == "__main__":
    main()
