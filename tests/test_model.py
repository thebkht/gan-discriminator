from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models import (
    BranchAEncoder,
    BranchB_Spatiotemporal,
    BranchC_Physics,
    DiscriminatorPhase2,
    DiscriminatorPhase3,
    load_phase2_checkpoint,
    load_phase2_into_phase3,
    load_phase3_checkpoint,
    load_pretrained_branch_a,
)
from models.branch_b import SUMMARY_FEATURE_NAMES, _scalar_stats
from training.losses import HingeLoss


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

GOLDEN_BRANCH_C_FLOW = torch.tensor(
    [
        [
            7.9348e-03,
            2.0742e-08,
            7.9349e-03,
            7.9348e-03,
            1.1921e-07,
            -7.6907e-03,
            2.0742e-08,
            -7.6905e-03,
            -7.6907e-03,
            1.1921e-07,
            2.1509e00,
            2.0131e-01,
            2.5000e00,
            1.8029e00,
            6.9708e-01,
            2.1509e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            1.0000e00,
        ],
        [
            7.9348e-03,
            1.4477e-08,
            7.9349e-03,
            7.9348e-03,
            7.4506e-08,
            -7.6907e-03,
            7.7920e-09,
            -7.6906e-03,
            -7.6907e-03,
            2.9802e-08,
            7.9638e-01,
            1.8026e-01,
            1.1180e00,
            5.0008e-01,
            6.1791e-01,
            7.9638e-01,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            1.0000e00,
        ],
        [
            7.9348e-03,
            7.7920e-09,
            7.9348e-03,
            7.9348e-03,
            2.9802e-08,
            -7.6907e-03,
            1.4477e-08,
            -7.6906e-03,
            -7.6907e-03,
            7.4506e-08,
            7.9638e-01,
            1.8026e-01,
            1.1180e00,
            5.0008e-01,
            6.1791e-01,
            7.9638e-01,
            0.0000e00,
            1.0000e00,
            0.0000e00,
            0.0000e00,
        ],
        [
            7.9348e-03,
            2.0742e-08,
            7.9349e-03,
            7.9348e-03,
            1.1921e-07,
            -7.6907e-03,
            2.0742e-08,
            -7.6905e-03,
            -7.6907e-03,
            1.1921e-07,
            2.1509e00,
            2.0131e-01,
            2.5000e00,
            1.8029e00,
            6.9708e-01,
            2.1509e00,
            0.0000e00,
            1.0000e00,
            0.0000e00,
            0.0000e00,
        ],
    ],
    dtype=torch.float32,
)

GOLDEN_BRANCH_C_HSV = torch.tensor(
    [
        [5.8333e-01, 0.0000e00, 8.1097e-01, 2.0833e-01, 8.3333e-02, 2.1104e-08, 1.7402e-01, 9.5834e-01],
        [5.8333e-01, 0.0000e00, 3.6465e-01, 4.5833e-01, 8.3333e-02, 2.1084e-08, 2.3557e-01, 7.0834e-01],
        [5.8333e-01, 2.5767e-08, 2.3557e-01, 7.0834e-01, 8.3333e-02, 9.4223e-09, 3.6465e-01, 4.5833e-01],
        [5.8333e-01, 2.5810e-08, 1.7402e-01, 9.5834e-01, 8.3333e-02, 1.1176e-08, 8.1097e-01, 2.0833e-01],
    ],
    dtype=torch.float32,
)


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


class BranchCModelTestCase(unittest.TestCase):
    def _fixed_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_a = torch.linspace(-1.0, 1.0, steps=4 * 3 * 64 * 64, dtype=torch.float32).reshape(4, 3, 64, 64)
        frame_b = torch.linspace(1.0, -1.0, steps=4 * 3 * 64 * 64, dtype=torch.float32).reshape(4, 3, 64, 64)
        flow = torch.linspace(-2.0, 2.0, steps=4 * 2 * 64 * 64, dtype=torch.float32).reshape(4, 2, 64, 64)
        return frame_a, frame_b, flow

    def test_branch_c_output_shape(self) -> None:
        model = BranchC_Physics()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)
        flow = torch.randn(4, 2, 64, 64)

        output = model(frame_a, frame_b, flow)

        self.assertEqual(tuple(output.shape), (4, 28))
        self.assertFalse(torch.isnan(output).any())

    def test_branch_c_flow_features_golden(self) -> None:
        model = BranchC_Physics().eval()
        frame_a, frame_b, flow = self._fixed_inputs()

        with torch.no_grad():
            output = model(frame_a, frame_b, flow)

        self.assertTrue(torch.allclose(output[:, :20], GOLDEN_BRANCH_C_FLOW, rtol=1e-4, atol=1e-5))

    def test_branch_c_hsv_features_golden(self) -> None:
        model = BranchC_Physics().eval()
        frame_a, frame_b, flow = self._fixed_inputs()

        with torch.no_grad():
            output = model(frame_a, frame_b, flow)

        self.assertTrue(torch.allclose(output[:, 20:], GOLDEN_BRANCH_C_HSV, rtol=1e-4, atol=1e-5))


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


class DiscriminatorPhase3TestCase(unittest.TestCase):
    def _phase2_checkpoint(self) -> tuple[DiscriminatorPhase2, Path]:
        phase2_model = DiscriminatorPhase2()
        checkpoint_path = Path("tests/_phase2_tmp_ckpt.pt")
        torch.save(
            {
                "phase": 2,
                "epoch": 3,
                "model_state_dict": phase2_model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "best_validation_metrics": {"balanced_accuracy": 0.9, "f1": 0.9, "loss": 0.1},
            },
            checkpoint_path,
        )
        self.addCleanup(lambda: checkpoint_path.unlink(missing_ok=True))
        return phase2_model, checkpoint_path

    def test_discriminator_phase3_output_shape(self) -> None:
        model = DiscriminatorPhase3()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)
        flow = torch.randn(4, 2, 64, 64)

        logits = model(frame_a, frame_b, flow)

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertFalse(torch.isnan(logits).any())

    def test_load_phase2_into_phase3_skips_fusion_keys(self) -> None:
        phase2_model, checkpoint_path = self._phase2_checkpoint()
        phase3_model = DiscriminatorPhase3()
        fusion_before = {key: value.clone() for key, value in phase3_model.fusion.state_dict().items()}

        load_phase2_into_phase3(phase3_model, checkpoint_path)

        for key, value in phase2_model.branch_a.state_dict().items():
            self.assertTrue(torch.equal(value, phase3_model.branch_a.state_dict()[key]), msg=f"branch_a mismatch at {key}")
        for key, value in phase2_model.branch_b.state_dict().items():
            self.assertTrue(torch.equal(value, phase3_model.branch_b.state_dict()[key]), msg=f"branch_b mismatch at {key}")
        for key, value in fusion_before.items():
            self.assertTrue(torch.equal(value, phase3_model.fusion.state_dict()[key]), msg=f"fusion changed at {key}")

    def test_branch_a_b_frozen_after_phase3_step(self) -> None:
        _, checkpoint_path = self._phase2_checkpoint()
        model = DiscriminatorPhase3()
        load_phase2_into_phase3(model, checkpoint_path)
        optimizer = torch.optim.Adam(model.fusion.parameters(), lr=2e-4, betas=(0.5, 0.999))
        criterion = torch.nn.BCEWithLogitsLoss()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)
        flow = torch.randn(4, 2, 64, 64)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        before_a = {key: value.clone() for key, value in model.branch_a.state_dict().items()}
        before_b = {key: value.clone() for key, value in model.branch_b.state_dict().items()}

        optimizer.zero_grad(set_to_none=True)
        logits = model(frame_a, frame_b, flow)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        for key, value in before_a.items():
            self.assertTrue(torch.equal(value, model.branch_a.state_dict()[key]), msg=f"branch_a changed at {key}")
        for key, value in before_b.items():
            self.assertTrue(torch.equal(value, model.branch_b.state_dict()[key]), msg=f"branch_b changed at {key}")

    def test_load_phase3_checkpoint_returns_next_epoch(self) -> None:
        _, checkpoint_path = self._phase2_checkpoint()
        model = DiscriminatorPhase3()
        load_phase2_into_phase3(model, checkpoint_path)
        optimizer = torch.optim.Adam(model.fusion.parameters(), lr=2e-4, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        phase3_checkpoint_path = Path("tests/_phase3_tmp_ckpt.pt")
        torch.save(
            {
                "phase": 3,
                "epoch": 4,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_metrics": {"balanced_accuracy": 0.85, "f1": 0.8, "loss": 0.2},
            },
            phase3_checkpoint_path,
        )
        self.addCleanup(lambda: phase3_checkpoint_path.unlink(missing_ok=True))

        next_epoch = load_phase3_checkpoint(model, optimizer, scheduler, phase3_checkpoint_path)

        self.assertEqual(next_epoch, 5)


class LossTestCase(unittest.TestCase):
    def test_hinge_loss_matches_committed_label_convention(self) -> None:
        logits = torch.tensor([2.0, -2.0, 0.0, 0.0], dtype=torch.float32)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

        loss = HingeLoss()(logits, labels)

        self.assertAlmostEqual(float(loss.item()), 0.5, places=6)
