"""Reusable training harness used from Week 6 onwards.

Torch is an optional dependency — this module is importable without torch
(for tooling like mypy/ruff), but `Trainer.fit` will raise ImportError if
torch is missing. Install with `pip install -e ".[dl]"`.

Features:
  * Device auto-detection (CUDA → MPS → CPU).
  * Gradient accumulation, gradient clipping.
  * Mixed-precision autocast (CUDA, MPS, or CPU) when `mixed_precision=True`.
  * Deterministic seeding, torch RNG included.
  * Checkpoint save / resume with full RNG state round-trip.
  * Optional W&B logging gated by the `MLCOURSE_WANDB` env var.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlcourse.utils.repro import detect_device, seed_everything

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    lr: float = 1e-3
    device: str = "auto"  # "auto" → detect_device(), else {"cpu","mps","cuda"}
    seed: int = 0
    deterministic: bool = True
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    mixed_precision: bool = False
    log_every_n_steps: int = 50
    checkpoint_dir: str | None = None
    # Free-form space for downstream weeks to stash their own config.
    extras: dict[str, Any] = field(default_factory=dict)


LossFn = Callable[["torch.Tensor", "torch.Tensor"], "torch.Tensor"]


class Trainer:
    """Compact PyTorch training loop used throughout the course."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self._resolve_device()
        # Populated during `fit` so evaluation / logging can reuse them.
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._wandb_run = None

    # -- Public API ------------------------------------------------------------
    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        loss_fn: LossFn | None,
        optimizer: Optimizer,
    ) -> dict[str, list[float]]:
        """Train `model` on `train_loader`.

        If `loss_fn` is None the model is expected to produce the scalar loss
        itself from the batch (useful for objectives that don't fit the
        `(x, y) → loss` shape, e.g. DDPM / self-supervised / contrastive).
        """
        self._require_torch()
        self._seed()
        self._init_wandb()

        self.model = model.to(self.config.device)
        self.optimizer = optimizer

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                tl = self._run_epoch(train_loader, loss_fn, train=True)
                self.history["train_loss"].append(tl)
                if val_loader is not None:
                    vl = self._run_epoch(val_loader, loss_fn, train=False)
                    self.history["val_loss"].append(vl)
                    self._log({"epoch": epoch, "train_loss": tl, "val_loss": vl})
                else:
                    self._log({"epoch": epoch, "train_loss": tl})
            return self.history
        finally:
            if self._wandb_run is not None:
                self._wandb_run.finish()  # type: ignore[no-untyped-call]

    def save_checkpoint(self, path: str | Path) -> Path:
        self._require_torch()
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call fit() before saving a checkpoint.")
        import torch

        payload = {
            "config": self.config.__dict__,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "rng": {
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        return out

    def load_checkpoint(self, path: str | Path, *, model: nn.Module, optimizer: Optimizer) -> None:
        self._require_torch()
        import torch

        payload = torch.load(Path(path), map_location=self.config.device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.random.set_rng_state(payload["rng"]["torch"])
        if payload["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        self.model = model
        self.optimizer = optimizer
        self.history = payload.get("history", {"train_loss": [], "val_loss": []})

    # -- Internals -------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, loss_fn: LossFn | None, *, train: bool) -> float:
        """Run one epoch.

        Two modes:
        1. **Standard supervised**: `loss_fn` is a callable `(logits, y) -> loss`.
           Batches are unpacked into `(x, y)` and the model is called as `model(x)`.
        2. **Custom loss**: `loss_fn is None`. The model itself is responsible for
           computing the scalar loss — its forward must return either a scalar
           tensor or a `(logits, loss)` pair. Used by W10 DDPM (custom
           noise-prediction loss with a schedule) and any model whose loss
           doesn't fit the `(x, y)` shape.
        """
        import torch

        assert self.model is not None and self.optimizer is not None
        self.model.train(mode=train)

        total_loss = 0.0
        total_count = 0
        # torch.no_grad is only for eval.
        no_grad = contextlib.nullcontext() if train else torch.no_grad()

        with no_grad:
            for step, batch in enumerate(loader, start=1):
                if loss_fn is None:
                    # Custom-loss path: whole batch to device, model returns loss.
                    batch = self._batch_to_device(batch)
                    with self._autocast():
                        out = self.model(*batch) if isinstance(batch, tuple) else self.model(batch)
                        loss = out[1] if isinstance(out, tuple) else out
                        loss = loss / self.config.grad_accum_steps
                    count = self._batch_size(batch)
                else:
                    x, y = self._unpack(batch)
                    x = x.to(self.config.device)
                    y = y.to(self.config.device)
                    with self._autocast():
                        logits = self.model(x)
                        loss = loss_fn(logits, y) / self.config.grad_accum_steps
                    count = y.shape[0] if hasattr(y, "shape") else 1

                if train:
                    loss.backward()
                    if step % self.config.grad_accum_steps == 0:
                        if self.config.grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.grad_clip_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.detach()) * self.config.grad_accum_steps * count
                total_count += count

        return total_loss / max(total_count, 1)

    @staticmethod
    def _unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    def _batch_to_device(self, batch: Any) -> Any:
        """Move tensors / nested tuples of tensors to `self.config.device`."""
        import torch as _t

        device = self.config.device
        if isinstance(batch, _t.Tensor):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return tuple(b.to(device) if isinstance(b, _t.Tensor) else b for b in batch)
        return batch

    @staticmethod
    def _batch_size(batch: Any) -> int:
        import torch as _t

        if isinstance(batch, _t.Tensor):
            return int(batch.shape[0]) if batch.ndim else 1
        if isinstance(batch, (list, tuple)):
            for b in batch:
                if isinstance(b, _t.Tensor) and b.ndim:
                    return int(b.shape[0])
        return 1

    def _autocast(self):
        import torch

        if not self.config.mixed_precision:
            return contextlib.nullcontext()
        device_type = (
            "mps"
            if self.config.device == "mps"
            else ("cuda" if self.config.device == "cuda" else "cpu")
        )
        dtype = torch.float16 if device_type in {"mps", "cuda"} else torch.bfloat16
        return torch.autocast(device_type=device_type, dtype=dtype)

    def _resolve_device(self) -> None:
        if self.config.device == "auto":
            self.config.device = detect_device()

    def _seed(self) -> None:
        seed_everything(self.config.seed, deterministic_torch=self.config.deterministic)

    def _init_wandb(self) -> None:
        if os.environ.get("MLCOURSE_WANDB") != "1":
            return
        try:  # pragma: no cover - optional dependency
            import wandb

            self._wandb_run = wandb.init(
                project=os.environ.get("MLCOURSE_WANDB_PROJECT", "mlcourse"),
                config=self.config.__dict__,
                reinit=True,
            )
        except ImportError:
            self._wandb_run = None

    def _log(self, metrics: dict[str, Any]) -> None:
        if self._wandb_run is not None:  # pragma: no cover - side effect
            self._wandb_run.log(metrics)

    @staticmethod
    def _require_torch() -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover - environment guard
            raise ImportError(
                "PyTorch is required for Trainer.fit(). Install with `pip install -e '.[dl]'`."
            ) from e
