"""Experiment tracking abstraction with a TensorBoard default backend."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch


class Tracker:
    """Thin tracker wrapper so training code does not depend on a concrete backend."""

    def __init__(self, log_dir: str | Path, backend: str = "tensorboard") -> None:
        self.backend = backend.lower()
        if self.backend != "tensorboard":
            raise ValueError(f"Unsupported tracker backend: {backend}")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TensorBoard is not installed. Run `pip install -r requirements.txt` to enable tracking."
            ) from exc
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, value, step)

    def log_image(self, name: str, image: torch.Tensor, step: int) -> None:
        tensor = image.detach().cpu()
        if tensor.ndim == 3:
            self._writer.add_image(name, tensor, step)
            return
        if tensor.ndim == 4:
            self._writer.add_images(name, tensor, step)
            return
        raise ValueError("Expected image tensor with 3 or 4 dimensions")

    def log_hparams(
        self,
        hparams: Mapping[str, object],
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._writer.add_hparams(dict(hparams), dict(metrics or {}))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
