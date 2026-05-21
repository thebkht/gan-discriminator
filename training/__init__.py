"""Training exports for the Week 1 and Week 2 training entrypoints."""

from training.tracker import Tracker

__all__ = ["Tracker", "train_branch_a", "train_phase2"]


def __getattr__(name: str):
    if name == "train_branch_a":
        from training.trainer import train_branch_a

        return train_branch_a
    if name == "train_phase2":
        from training.phase2_trainer import train_phase2

        return train_phase2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
