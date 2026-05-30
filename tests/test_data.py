from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Mapping, cast

import torch
from PIL import Image

from data.augmentations import build_transforms
from data.celeba_loader import (
    CelebAFramePairDataset,
    FramePairSample,
    _load_flow_tensor,
    create_celeba_dataloader,
    verify_flow_cache,
)
from data.face_align import align_face, align_face_or_fallback
from data.forensics_loader import (
    ForensicsFramePairDataset,
    create_forensics_dataloader,
    discover_forensics_datasets,
    normalize_split,
    resolve_forensics_root,
)
from data.precompute_flow import precompute_flow
from training.tracker import Tracker

try:
    import tensorboard  # noqa: F401
except ModuleNotFoundError:
    tensorboard = None


class DataPipelineTestCase(unittest.TestCase):
    @staticmethod
    def _metadata(sample: FramePairSample) -> Mapping[str, object]:
        return cast(Mapping[str, object], sample["metadata"])

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
        sample = cast(FramePairSample, dataset[0])
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
        labels = [int(cast(FramePairSample, dataset[idx])["label"]) for idx in range(len(dataset))]
        self.assertEqual(sum(labels), len(labels) // 2)

    def test_identity_pairs_use_same_identity_when_file_present(self) -> None:
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.0,
            train=False,
        )
        sample = cast(FramePairSample, dataset[0])
        metadata = self._metadata(sample)
        self.assertEqual(metadata["pair_type"], "real")
        self.assertEqual(metadata["pair_strategy"], "same_identity")
        self.assertEqual(metadata["identity"], 1)
        self.assertTrue(str(metadata["pair_path"]).endswith("000002.jpg"))

    def test_adjacent_fallback_when_identity_file_missing(self) -> None:
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.root / "celeba" / "missing_identity.txt",
            image_size=64,
            fake_ratio=0.0,
            train=False,
        )
        sample = cast(FramePairSample, dataset[2])
        metadata = self._metadata(sample)
        self.assertEqual(metadata["pair_strategy"], "adjacent_fallback")
        self.assertTrue(str(metadata["pair_path"]).endswith("000004.jpg"))
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
        sample = cast(FramePairSample, dataset[0])
        metadata = self._metadata(sample)
        self.assertEqual(metadata["pair_type"], "real")
        self.assertEqual(metadata["pair_strategy"], "identity_singleton_adjacent")
        self.assertEqual(metadata["identity"], 1)
        self.assertTrue(str(metadata["pair_path"]).endswith("000002.jpg"))

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
        batch = cast(Mapping[str, torch.Tensor], next(iter(loader)))
        elapsed = time.perf_counter() - start
        self.assertEqual(batch["frame_a"].shape[0], 64)
        self.assertLess(elapsed, 5.0)

    def test_flow_precompute_smoke(self) -> None:
        out_dir = self.root / "flow_cache"
        count = precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)
        self.assertEqual(count, len(list(self.image_root.glob("*.jpg"))))
        sample_flow = torch.load(out_dir / "000001_flow.pt")
        self.assertEqual(tuple(sample_flow.shape), (2, 64, 64))

    def test_getitem_includes_flow_tensor(self) -> None:
        out_dir = self.root / "flow_cache"
        precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.0,
            train=False,
            flow_cache_dir=out_dir,
        )

        sample = cast(FramePairSample, dataset[0])

        self.assertIn("flow", sample)
        flow = cast(torch.Tensor, sample.get("flow"))
        self.assertEqual(tuple(flow.shape), (2, 64, 64))
        self.assertEqual(flow.dtype, torch.float32)
        self.assertFalse(torch.isnan(flow).any())

    def test_adjacent_cache_pairing_matches_precompute_partner(self) -> None:
        out_dir = self.root / "flow_cache"
        precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)
        dataset = CelebAFramePairDataset(
            image_dir=self.image_root,
            identity_file=self.identity_file,
            image_size=64,
            fake_ratio=0.5,
            train=False,
            flow_cache_dir=out_dir,
            pairing_mode="adjacent_cache",
        )

        sample = cast(FramePairSample, dataset[2])
        metadata = self._metadata(sample)

        self.assertTrue(str(metadata["pair_path"]).endswith("000004.jpg"))
        self.assertEqual(metadata["pair_strategy"], "adjacent_cache_identity_match")
        self.assertEqual(int(cast(torch.Tensor, sample["label"]).item()), 0)

    def test_collate_includes_flow_batch(self) -> None:
        out_dir = self.root / "flow_cache"
        precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)
        config = self._base_config()
        config["paths"]["flow_cache_dir"] = str(out_dir)
        config["phase3"] = {"include_flow": True, "pairing_mode": "adjacent_cache"}

        loader = create_celeba_dataloader(config, split="train", limit=8)
        batch = cast(Mapping[str, torch.Tensor], next(iter(loader)))

        self.assertIn("flow", batch)
        self.assertEqual(tuple(batch["flow"].shape), (6, 2, 64, 64))

    def test_flow_cache_load_real_format(self) -> None:
        out_dir = self.root / "flow_cache"
        precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)

        flow = _load_flow_tensor(out_dir / "000001_flow.pt")

        self.assertEqual(tuple(flow.shape), (2, 64, 64))

    def test_verify_flow_cache_counts_missing_and_extra(self) -> None:
        out_dir = self.root / "flow_cache"
        precompute_flow(self.image_root, out_dir, method="farneback", image_size=64)
        (out_dir / "000003_flow.pt").unlink()
        torch.save(torch.zeros(2, 64, 64), out_dir / "extra_flow.pt")

        summary = verify_flow_cache(self.image_root, out_dir)

        self.assertEqual(summary["expected_count"], 8)
        self.assertEqual(summary["count"], 8)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["extra_count"], 1)
        self.assertIn("000003", cast(list[str], summary["missing"]))
        self.assertIn("extra", cast(list[str], summary["extra"]))

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


class ForensicsLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.top = self.root / "forensics"
        self.dataset_outer = self.top / "Data Set 1"
        self.dataset_root = self.dataset_outer / "Data Set 1"
        for split in ("train", "validation", "test"):
            for class_name in ("real", "fake"):
                class_dir = self.dataset_root / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(3):
                    image = Image.new(
                        "RGB",
                        (80, 72),
                        color=(
                            (idx + 1) * 30,
                            20 if class_name == "real" else 180,
                            40 if split == "test" else 120,
                        ),
                    )
                    image.save(class_dir / f"{class_name}_{idx:03d}.jpg")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolves_nested_data_set_path(self) -> None:
        self.assertEqual(resolve_forensics_root(self.dataset_outer), self.dataset_root)
        self.assertEqual(resolve_forensics_root(self.dataset_root), self.dataset_root)
        self.assertEqual(discover_forensics_datasets(self.top), [self.dataset_root])

    def test_normalize_validation_split_alias(self) -> None:
        self.assertEqual(normalize_split("val"), "validation")
        dataset = ForensicsFramePairDataset(self.dataset_outer, split="val", pairing_mode="degenerate")
        self.assertEqual(dataset.split, "validation")

    def test_sample_shapes_label_and_finite_flow(self) -> None:
        dataset = ForensicsFramePairDataset(self.dataset_outer, split="test")
        sample = dataset[0]

        self.assertEqual(tuple(sample["frame_a"].shape), (3, 64, 64))
        self.assertEqual(tuple(sample["frame_b"].shape), (3, 64, 64))
        self.assertEqual(tuple(sample["flow"].shape), (2, 64, 64))
        self.assertIn(int(sample["label"].item()), {0, 1})
        self.assertFalse(torch.isnan(sample["flow"]).any())
        self.assertNotEqual(sample["path_a"], sample["path_b"])

    def test_forensics_dataloader_batches_contract(self) -> None:
        loader = create_forensics_dataloader(
            self.dataset_outer,
            split="validation",
            pairing_mode="degenerate",
            batch_size=4,
            num_workers=0,
        )
        batch = next(iter(loader))

        self.assertEqual(tuple(batch["frame_a"].shape), (4, 3, 64, 64))
        self.assertEqual(tuple(batch["frame_b"].shape), (4, 3, 64, 64))
        self.assertEqual(tuple(batch["flow"].shape), (4, 2, 64, 64))
        self.assertEqual(len(batch["path_a"]), 4)

    def test_forensics_limit_preserves_both_classes(self) -> None:
        dataset = ForensicsFramePairDataset(
            self.dataset_outer,
            split="test",
            pairing_mode="degenerate",
            limit=4,
        )

        labels = [int(dataset[index]["label"].item()) for index in range(len(dataset))]

        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)

    def test_forensics_loader_reads_aligned_cache(self) -> None:
        aligned_root = self.root / "forensics_aligned"
        aligned_class_dir = aligned_root / "Data Set 1" / "test" / "real"
        aligned_class_dir.mkdir(parents=True, exist_ok=True)
        fake_dir = aligned_root / "Data Set 1" / "test" / "fake"
        fake_dir.mkdir(parents=True, exist_ok=True)
        for class_name in ("real", "fake"):
            for idx in range(3):
                Image.new("RGB", (64, 64), color=(255, 0, 0)).save(
                    aligned_root / "Data Set 1" / "test" / class_name / f"{class_name}_{idx:03d}.jpg"
                )

        dataset = ForensicsFramePairDataset(
            self.dataset_outer,
            split="test",
            pairing_mode="degenerate",
            aligned_root=aligned_root,
        )
        sample = dataset[0]

        self.assertEqual(tuple(sample["frame_a"].shape), (3, 64, 64))
        self.assertIn("forensics_aligned", sample["path_a"])
        self.assertEqual(sample["path_a"], sample["path_b"])

    def test_face_align_uses_mock_mtcnn_and_fallback(self) -> None:
        image = Image.new("RGB", (80, 72), color=(20, 40, 60))

        class Detector:
            def __init__(self, result):
                self.result = result

            def __call__(self, _image):
                return self.result

        aligned = align_face(image, mtcnn=Detector(Image.new("RGB", (64, 64), color=(1, 2, 3))))
        fallback = align_face_or_fallback(image, image_size=64, mtcnn=Detector(None))

        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.size, (64, 64))
        self.assertEqual(fallback.size, (64, 64))
