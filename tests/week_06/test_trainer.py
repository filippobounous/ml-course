"""Auto-graded checks for the Week 6 Trainer harness.

All torch-dependent tests are gated on `pytest.importorskip("torch")` so the
rest of the course's test suite stays green even when torch is not installed
(e.g. on a plain pip install without `[dl]`).
"""

from __future__ import annotations

import pytest

from mlcourse.trainer import Trainer, TrainerConfig
from mlcourse.utils import detect_device


def test_config_defaults():
    cfg = TrainerConfig()
    assert cfg.max_epochs == 10
    assert cfg.lr == 1e-3
    assert cfg.device == "auto"
    assert cfg.deterministic is True
    assert cfg.extras == {}


def test_config_overrides():
    cfg = TrainerConfig(max_epochs=3, lr=1e-2, device="cpu", extras={"model": "toy"})
    assert cfg.max_epochs == 3
    assert cfg.lr == 1e-2
    assert cfg.device == "cpu"
    assert cfg.extras == {"model": "toy"}


def test_trainer_resolves_auto_device():
    t = Trainer(TrainerConfig(device="auto"))
    assert t.config.device in {"cpu", "mps", "cuda"}


def test_detect_device_is_valid():
    assert detect_device() in {"cpu", "mps", "cuda"}


# -- Torch-required tests ------------------------------------------------------


@pytest.fixture(scope="module")
def torch_env():
    torch = pytest.importorskip("torch")
    return torch


@pytest.fixture
def tiny_supervised(torch_env):
    torch = torch_env
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 4, generator=g)
    w = torch.randn(4, 1, generator=g)
    y = X @ w + 0.1 * torch.randn(64, 1, generator=g)
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True, generator=g)
    return loader


def _tiny_model(torch):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def test_fit_runs_end_to_end(torch_env, tiny_supervised):
    torch = torch_env
    model = _tiny_model(torch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(TrainerConfig(max_epochs=2, lr=1e-2, device="cpu", seed=0))
    history = trainer.fit(
        model, tiny_supervised, loss_fn=torch.nn.functional.mse_loss, optimizer=opt
    )
    assert len(history["train_loss"]) == 2
    assert history["train_loss"][1] <= history["train_loss"][0] + 1e-6


def test_checkpoint_round_trip(tmp_path, torch_env, tiny_supervised):
    torch = torch_env
    model = _tiny_model(torch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(TrainerConfig(max_epochs=1, lr=1e-2, device="cpu", seed=0))
    trainer.fit(model, tiny_supervised, loss_fn=torch.nn.functional.mse_loss, optimizer=opt)
    ckpt = trainer.save_checkpoint(tmp_path / "ckpt.pt")

    # Fresh model + optimiser.
    model2 = _tiny_model(torch)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    trainer2 = Trainer(TrainerConfig(device="cpu"))
    trainer2.load_checkpoint(ckpt, model=model2, optimizer=opt2)

    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.allclose(a, b, atol=0), "parameters must match bit-for-bit after load"


def test_mixed_precision_context_is_safe(torch_env):
    torch = torch_env
    # On CPU, autocast bf16 is a no-op but must not raise.
    cfg = TrainerConfig(device="cpu", mixed_precision=True)
    trainer = Trainer(cfg)
    with trainer._autocast():
        x = torch.randn(2, 3)
        y = x @ x.T  # exercise the context
    assert y.shape == (2, 2)
