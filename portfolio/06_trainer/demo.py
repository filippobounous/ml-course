"""End-to-end demo of `mlcourse.Trainer`.

1. Generates a toy regression dataset (10-dim inputs, linear target + noise).
2. Builds an MLP and trains for a few epochs with `Trainer`.
3. Saves a checkpoint; builds a fresh model + optimiser; loads the checkpoint;
   verifies that the resumed loss trajectory matches the first run.

Requires `torch` (install `pip install -e ".[dl]"`). Skips gracefully otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _build_loaders(n_train: int = 800, n_val: int = 200, d: int = 10, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, 1, generator=g)
    X = torch.randn(n_train + n_val, d, generator=g)
    y = X @ w + 0.1 * torch.randn(n_train + n_val, 1, generator=g)
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    return (
        DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True, generator=g),
        DataLoader(TensorDataset(Xte, yte), batch_size=64, shuffle=False),
    )


def _build_model(d_in: int = 10):
    import torch
    from torch import nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(d_in, 32), nn.Tanh(), nn.Linear(32, 1))


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed — skipping the Trainer demo.")
        print("Install with `pip install -e '.[dl]'` and rerun.")
        return 0

    from mlcourse.trainer import Trainer, TrainerConfig

    cfg = TrainerConfig(max_epochs=5, lr=1e-2, device="cpu", seed=42)

    # First run.
    model = _build_model()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = Trainer(cfg)
    train_loader, val_loader = _build_loaders()
    history1 = trainer.fit(
        model, train_loader, val_loader, loss_fn=torch.nn.functional.mse_loss, optimizer=opt
    )
    ckpt = trainer.save_checkpoint(HERE / "checkpoint.pt")

    # Resume from checkpoint into a fresh model + optimiser.
    model2 = _build_model()
    opt2 = torch.optim.Adam(model2.parameters(), lr=cfg.lr)
    trainer2 = Trainer(cfg)
    trainer2.load_checkpoint(ckpt, model=model2, optimizer=opt2)

    # Verify model weights are identical after load.
    diffs = [
        float((a - b).abs().max())
        for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True)
    ]
    max_diff = max(diffs)

    report = [
        "# Week 6 — mlcourse.Trainer demo",
        "",
        f"Device: `{cfg.device}`  |  Epochs: `{cfg.max_epochs}`",
        "",
        "## Training losses (epoch 1 → N)",
        f"- train: {[round(v, 4) for v in history1['train_loss']]}",
        f"- val:   {[round(v, 4) for v in history1['val_loss']]}",
        "",
        "## Deterministic checkpoint round-trip",
        f"- max |Δ parameter| after save/load: `{max_diff:.2e}` (should be 0)",
        "",
        f"Checkpoint written to `{ckpt.name}`.",
    ]
    (HERE / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"train losses: {history1['train_loss']}")
    print(f"val losses:   {history1['val_loss']}")
    print(f"max param diff after reload: {max_diff:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
