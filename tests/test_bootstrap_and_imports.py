from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from data import augmentations, celeba_loader, precompute_flow
from training import tracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_celeba.sh"


class BootstrapAndImportTestCase(unittest.TestCase):
    def test_missing_kaggle_cli_fails_clearly(self) -> None:
        kaggle_path = shutil.which("kaggle")
        self.assertIsNotNone(kaggle_path)
        kaggle_dir = str(Path(kaggle_path).resolve().parent)
        filtered_path = os.pathsep.join(
            segment for segment in os.environ.get("PATH", "").split(os.pathsep) if segment and segment != kaggle_dir
        )
        env = os.environ.copy()
        env["PATH"] = filtered_path
        result = subprocess.run(
            ["bash", str(DOWNLOAD_SCRIPT)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kaggle CLI is not installed", result.stderr)

    def test_missing_kaggle_credentials_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            env = os.environ.copy()
            env["HOME"] = temp_home
            result = subprocess.run(
                ["bash", str(DOWNLOAD_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Kaggle credentials not found", result.stderr)

    def test_imports_resolve_across_modules(self) -> None:
        self.assertTrue(hasattr(augmentations, "build_transforms"))
        self.assertTrue(hasattr(celeba_loader, "CelebAFramePairDataset"))
        self.assertTrue(hasattr(celeba_loader, "create_celeba_dataloader"))
        self.assertTrue(hasattr(precompute_flow, "precompute_flow"))
        self.assertTrue(hasattr(tracker, "Tracker"))
