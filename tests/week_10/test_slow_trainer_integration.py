"""Slow integration test for W10: `mlcourse.Trainer` + custom DDPM loss path.

Trains a tiny UNet-DDPM on random 8×8 images for 1 epoch. Asserts training
loss decreases and that the new `loss_fn=None` path in `Trainer.fit`
actually works end-to-end.

Skipped by default; run with `pytest --run-slow -q`.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_ddpm_trainer_custom_loss_path():
    torch = pytest.importorskip("torch")
    # Import DDPM pieces from the portfolio — mirrors how train.py wires them.
    import importlib.util
    import sys
    from pathlib import Path

    from torch.utils.data import DataLoader, TensorDataset

    from mlcourse.trainer import Trainer, TrainerConfig

    ddpm_path = Path(__file__).resolve().parents[2] / "portfolio" / "10_ddpm" / "ddpm.py"
    spec = importlib.util.spec_from_file_location("w10_ddpm_slow", ddpm_path)
    assert spec and spec.loader
    ddpm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ddpm
    spec.loader.exec_module(ddpm)

    torch.manual_seed(0)
    # 64 "images" of 1×8×8, random normal. Not pretty — just enough signal.
    x = torch.randn(64, 1, 8, 8)
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    schedule = ddpm.DiffusionSchedule.linear(50)
    unet = ddpm.SmallUNet(in_ch=1, base=8)

    class DDPMLossModule(torch.nn.Module):
        def __init__(self, unet, schedule):
            super().__init__()
            self.unet = unet
            self.schedule = schedule

        def forward(self, images, _labels):
            return ddpm.ddpm_loss(self.unet, images, self.schedule)

    model = DDPMLossModule(unet, schedule)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=1e-3)
    trainer = Trainer(TrainerConfig(max_epochs=3, device="cpu", seed=0))
    trainer.fit(model, loader, loss_fn=None, optimizer=optimizer)

    history = trainer.history["train_loss"]
    assert len(history) == 3
    # DDPM loss is variance + model error; with our tiny UNet + random images
    # we should see a clear drop within three epochs.
    assert history[-1] < history[0], (
        f"ddpm train loss should drop: {history[0]:.4f} → {history[-1]:.4f}"
    )
