from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models import (
    BranchAEncoder,
    BranchB_Spatiotemporal,
    DiscriminatorPhase2,
    load_phase2_checkpoint,
    load_pretrained_branch_a,
)
from models.branch_b import SUMMARY_FEATURE_NAMES, _scalar_stats


# GOLDEN_BRANCH_B_SUMMARY is a snapshot of the committed 8-D summary at seed 0 — not a
# target specification. Regenerate only if the temporal summary formula intentionally changes.
GOLDEN_BRANCH_B_SUMMARY = torch.tensor(
    [
        [
            1.0871810048e10,
            1.0229133312e11,
            5.8339314893e11,
            -3.7027964518e11,
            4.6016162634e-01,
            4.6552490967e12,
            3.5693359375e-01,
            7.3092571136e10,
        ],
        [
            3.8384798720e09,
            3.9767601152e10,
            2.3013149901e11,
            -1.4252330189e11,
            4.4111368060e-01,
            1.8080401981e12,
            3.6767578125e-01,
            2.8111591424e10,
        ],
        [
            -3.0510755840e09,
            4.0941551616e10,
            1.8501748326e11,
            -1.9699046810e11,
            4.1170740128e-01,
            1.8579410125e12,
            2.8027343750e-01,
            2.9813532672e10,
        ],
        [
            -1.1166570496e10,
            1.0507102618e11,
            3.6747968512e11,
            -5.5201719910e11,
            4.3321883678e-01,
            4.7817482568e12,
            2.6855468750e-01,
            7.6299599872e10,
        ],
    ],
    dtype=torch.float32,
)
PHASE1_CKPT = Path("checkpoints/phase1_branch_a_best.pt")


class BranchBModelTestCase(unittest.TestCase):
    def _build_fixed_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame_a = torch.linspace(-1.0, 1.0, steps=4 * 3 * 64 * 64, dtype=torch.float32).reshape(
            4, 3, 64, 64
        )
        frame_b = torch.linspace(1.0, -1.0, steps=4 * 3 * 64 * 64, dtype=torch.float32).reshape(
            4, 3, 64, 64
        )
        return frame_a, frame_b

    def test_branch_b_output_shape(self) -> None:
        model = BranchB_Spatiotemporal(BranchAEncoder())
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)

        output = model(frame_a, frame_b)

        self.assertEqual(tuple(output.shape), (4, 32))
        self.assertFalse(torch.isnan(output).any())

    def test_branch_b_summary_shape(self) -> None:
        model = BranchB_Spatiotemporal(BranchAEncoder())
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)

        summary = model._summary_features(frame_a, frame_b)

        self.assertEqual(tuple(summary.shape), (4, 8))
        self.assertEqual(len(SUMMARY_FEATURE_NAMES), 8)

    def test_branch_b_summary_numerical_regression(self) -> None:
        torch.manual_seed(0)
        model = BranchB_Spatiotemporal(BranchAEncoder())
        model.eval()
        frame_a, frame_b = self._build_fixed_pair()

        with torch.no_grad():
            output = model._summary_features(frame_a, frame_b)

        self.assertTrue(torch.allclose(output, GOLDEN_BRANCH_B_SUMMARY, rtol=1e-5, atol=1e-4))

    def test_scalar_stats_dim(self) -> None:
        x = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64)

        stats = _scalar_stats(x, ("mean", "std", "max", "min"))

        self.assertEqual(tuple(stats.shape), (4, 4))
        self.assertTrue(torch.equal(stats[:, 2], x.max(dim=1).values))
        self.assertTrue(torch.equal(stats[:, 3], x.min(dim=1).values))


class DiscriminatorPhase2TestCase(unittest.TestCase):
    def test_discriminator_phase2_output_shape(self) -> None:
        model = DiscriminatorPhase2()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)

        logits = model(frame_a, frame_b)

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertFalse(torch.isnan(logits).any())

    def test_branch_a_partial_freeze_after_step(self) -> None:
        torch.manual_seed(0)
        model = DiscriminatorPhase2(backbone_train_last_n=2)
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.branch_b.expander.parameters())},
                {"params": list(model.branch_a.features[3].parameters())},
                {"params": list(model.branch_a.features[4].parameters())},
                {"params": list(model.fusion.parameters())},
            ],
            lr=2e-4,
            betas=(0.5, 0.999),
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        before = {key: value.clone() for key, value in model.branch_a.state_dict().items()}

        optimizer.zero_grad(set_to_none=True)
        logits = model(frame_a, frame_b)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        after = model.branch_a.state_dict()
        for key in before:
            is_frozen_block = any(key.startswith(f"features.{index}.") for index in range(3))
            if is_frozen_block:
                self.assertTrue(torch.equal(before[key], after[key]), msg=f"frozen branch_a changed at {key}")

    def test_branch_a_train_mode_splits_frozen_and_tail_blocks(self) -> None:
        model = DiscriminatorPhase2(backbone_train_last_n=2)

        model.train(True)

        for index, block in enumerate(model.branch_a.features):
            if index < 3:
                self.assertFalse(block.training, msg=f"expected block {index} in eval mode")
            else:
                self.assertTrue(block.training, msg=f"expected block {index} in train mode")

    @unittest.skipIf(not PHASE1_CKPT.exists(), "phase1 checkpoint not available")
    def test_load_phase1_encoder(self) -> None:
        model = DiscriminatorPhase2()

        load_pretrained_branch_a(model, PHASE1_CKPT)

        for parameter in model.branch_a.parameters():
            self.assertFalse(parameter.requires_grad)
        fresh_branch = BranchAEncoder()
        loaded_parameter = next(model.branch_a.parameters()).detach()
        fresh_parameter = next(fresh_branch.parameters()).detach()
        self.assertFalse(torch.allclose(loaded_parameter, fresh_parameter))
        logits = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2,))

    def test_load_phase2_checkpoint_rejects_legacy_branch_b_keys(self) -> None:
        model = DiscriminatorPhase2()
        legacy_path = Path("tests/_legacy_phase2_ckpt.pt")
        torch.save(
            {"model_state_dict": {"branch_b.embed.projection.weight": torch.randn(1, 1)}},
            legacy_path,
        )
        self.addCleanup(lambda: legacy_path.unlink(missing_ok=True))

        with self.assertRaisesRegex(RuntimeError, "legacy EmbedCNN keys"):
            load_phase2_checkpoint(model, legacy_path)
