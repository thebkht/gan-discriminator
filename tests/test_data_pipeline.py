from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import torch
from PIL import Image

from data.augmentations import build_transforms
from data.celeba_loader import CelebAFramePairDataset, create_celeba_dataloader
from data.precompute_flow import precompute_flow
from training.tracker import Tracker

try:
    import tensorboard  # noqa: F401
except ModuleNotFoundError:
    tensorboard = None


class DataPipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_root = self.root / "celeba" / "img_align_celeba"
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.identity_file = self.root / "celeba" / "identity_CelebA.txt"
        self._write_sample_images(count=8)
        self.identity_file.write_text(
            "\n".join(
                [
                    "000001.jpg 1",
                    "000002.jpg 1",
                    "000003.jpg 2",
                    "000004.jpg 2",
                    "000005.jpg 3",
                    "000006.jpg 3",
                    "000007.jpg 4",
                    "000008.jpg 4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_sample_images(self, count: int) -> None:
        for idx in range(1, count + 1):
            image = Image.new("RGB", (178, 218), color=(idx * 10 % 255, idx * 20 % 255, idx * 30 % 255))
            image.save(self.image_root / f"{idx:06d}.jpg")

    def _base_config(self) -> dict:
        return {
            "paths": {
                "image_dir": str(self.image_root),
                "identity_file": str(self.identity_file),
            },
            "dataset": {
                "image_size": 64,
                "fake_ratio": 0.5,
                "gaussian_noise_std": 0.05,
            },
            "dataloader": {
                "batch_size": 64,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
            },
        }

    def test_shape_and_no_nan(self) -> None:
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.5,
            train=False,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["frame_a"].shape), (3, 64, 64))
        self.assertEqual(tuple(sample["frame_b"].shape), (3, 64, 64))
        self.assertFalse(torch.isnan(sample["frame_a"]).any())
        self.assertFalse(torch.isnan(sample["frame_b"]).any())

    def test_label_balance_matches_fake_ratio(self) -> None:
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.5,
            train=False,
        )
        labels = [int(dataset[idx]["label"]) for idx in range(len(dataset))]
        self.assertEqual(sum(labels), len(labels) // 2)

    def test_identity_pairs_use_same_identity_when_file_present(self) -> None:
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.0,
            train=False,
        )
        sample = dataset[0]
        metadata = sample["metadata"]
        self.assertEqual(metadata["pair_type"], "real")
        self.assertEqual(metadata["pair_strategy"], "same_identity")
        self.assertEqual(metadata["identity"], 1)
        self.assertTrue(metadata["pair_path"].endswith("000002.jpg"))

    def test_adjacent_fallback_when_identity_file_missing(self) -> None:
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.root / "celeba" / "missing_identity.txt",
            image_size=64,
            fake_ratio=0.0,
            train=False,
        )
        sample = dataset[2]
        metadata = sample["metadata"]
        self.assertEqual(metadata["pair_strategy"], "adjacent_fallback")
        self.assertTrue(metadata["pair_path"].endswith("000004.jpg"))
        self.assertIsNone(metadata["identity"])

    def test_identity_singleton_falls_back_to_adjacent_pair(self) -> None:
        self.identity_file.write_text(
            "\n".join(
                [
                    "000001.jpg 1",
                    "000002.jpg 2",
                    "000003.jpg 2",
                    "000004.jpg 3",
                    "000005.jpg 3",
                    "000006.jpg 4",
                    "000007.jpg 4",
                    "000008.jpg 5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.0,
            train=False,
        )
        sample = dataset[0]
        metadata = sample["metadata"]
        self.assertEqual(metadata["pair_type"], "real")
        self.assertEqual(metadata["pair_strategy"], "identity_singleton_adjacent")
        self.assertEqual(metadata["identity"], 1)
        self.assertTrue(metadata["pair_path"].endswith("000002.jpg"))

    def test_augmentation_normalizes_to_expected_range(self) -> None:
        transform = build_transforms(image_size=64, train=True)
        tensor = transform(Image.open(self.image_root / "000001.jpg").convert("RGB"))
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertFalse(torch.isnan(tensor).any())
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertGreaterEqual(float(tensor.min()), -1.0)

    def test_dataloader_throughput_smoke(self) -> None:
        self._write_sample_images(count=256)
        config = self._base_config()
        config["paths"]["identity_file"] = str(self.root / "celeba" / "missing_identity.txt")
        loader = create_celeba_dataloader(config, split="train", limit=256)
        start = time.perf_counter()
        batch = next(iter(loader))
        elapsed = time.perf_counter() - start
        self.assertEqual(batch["frame_a"].shape[0], 64)
        self.assertLess(elapsed, 5.0)

    def test_flow_precompute_smoke(self) -> None:
        out_dir = self.root / "flow_cache"
        count = precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)
        self.assertEqual(count, len(list(self.image_root.glob("*.jpg"))))
        sample_flow = torch.load(out_dir / "000001_flow.pt")
        self.assertEqual(tuple(sample_flow.shape), (2, 64, 64))

    @unittest.skipIf(tensorboard is None, "tensorboard is not installed in the current environment")
    def test_tracker_emits_tensorboard_logs(self) -> None:
        run_dir = self.root / "runs" / "smoke"
        with Tracker(run_dir) as tracker:
            tracker.log_scalar("loss/train", 0.5, step=1)
            tracker.log_image("preview", torch.zeros(3, 64, 64), step=1)
            tracker.log_hparams({"batch_size": 64}, {"accuracy": 0.9})
            tracker.flush()
        event_files = list(run_dir.rglob("events.out.tfevents.*"))
        self.assertTrue(event_files)
