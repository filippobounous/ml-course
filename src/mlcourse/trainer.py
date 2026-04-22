"""Reusable training harness — W6 populates this, W7-W12 extend it.

This file is intentionally a stub in the Phase A scaffold. The full implementation
(Hydra config integration, W&B logging, mixed precision on MPS, gradient
accumulation, checkpointing, evaluation hooks) is delivered in Week 6.

The placeholder below lets downstream module code import the symbol and raise a
clear error if it is invoked before that module has been built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainerConfig:
    """Minimal config shape; extended with Hydra in Week 6."""

    max_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    log_every_n_steps: int = 50
    extras: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Placeholder trainer. See `modules/06_pytorch_trainer/` for the full build."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()

    def fit(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "mlcourse.Trainer is a Phase-A stub. It is implemented in Week 6 "
            "(see modules/06_pytorch_trainer/)."
        )
