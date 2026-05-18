from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models import BranchAEncoder, BranchB_Spatiotemporal, DiscriminatorPhase2, load_pretrained_branch_a
from models.branch_b import _scalar_stats


# GOLDEN_BRANCH_B is a snapshot of the committed implementation at seed 0 — not a
# target specification. Regenerate only if the formula intentionally changes.
GOLDEN_BRANCH_B = torch.tensor(
    [
        [-208144.46875, 2990590.0, 5215930.0, -0.00867898017168045, 0.12469834089279175, 0.2174881100654602, 2374705.25, 9062243.0],
        [-81864.625, 1095498.625, 1760118.125, -0.009315049275755882, 0.1246524453163147, 0.20027685165405273, 902030.9375, 3383793.25],
        [131928.734375, 995424.0625, 2911462.25, 0.016423286870121956, 0.12391640245914459, 0.3624364137649536, 784809.125, 2911462.25],
        [299666.25, 2843934.5, 7981622.5, 0.013098770752549171, 0.12431178987026215, 0.34888628125190735, 2168975.5, 7981622.5],
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
        model = BranchB_Spatiotemporal()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)

        output = model(frame_a, frame_b)

        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertFalse(torch.isnan(output).any())

    def test_branch_b_numerical_regression(self) -> None:
        torch.manual_seed(0)
        model = BranchB_Spatiotemporal()
        model.eval()
        frame_a, frame_b = self._build_fixed_pair()

        with torch.no_grad():
            output = model(frame_a, frame_b)

        self.assertTrue(torch.allclose(output, GOLDEN_BRANCH_B, rtol=1e-5, atol=1e-4))

    def test_scalar_stats_dim(self) -> None:
        x = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64)

        stats = _scalar_stats(x, ("mean", "std", "max"))

        self.assertEqual(tuple(stats.shape), (4, 3))
        self.assertTrue(torch.equal(stats[:, 2], x.max(dim=1).values))


class DiscriminatorPhase2TestCase(unittest.TestCase):
    def test_discriminator_phase2_output_shape(self) -> None:
        model = DiscriminatorPhase2()
        frame_a = torch.randn(4, 3, 64, 64)
        frame_b = torch.randn(4, 3, 64, 64)

        logits = model(frame_a, frame_b)

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertFalse(torch.isnan(logits).any())

    def test_branch_a_frozen_after_step(self) -> None:
        torch.manual_seed(0)
        model = DiscriminatorPhase2()
        optimizer = torch.optim.Adam(
            list(model.branch_b.parameters()) + list(model.fusion.parameters()),
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
            self.assertTrue(torch.equal(before[key], after[key]), msg=f"branch_a changed at {key}")

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

