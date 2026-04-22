"""Week 10 — schedules, DDIM determinism, InfoNCE, torch-gated UNet checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "10_diffusion_multimodal"
    / "problems"
    / "solutions.py"
)
DDPM_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "10_ddpm" / "ddpm.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load(SOLUTIONS_PATH, "w10_solutions")


# -- Schedules -----------------------------------------------------------------


def test_linear_schedule_shapes(sols):
    T = 100
    sch = sols.linear_schedule(T, beta_start=1e-4, beta_end=2e-2)
    assert sch.betas.shape == (T,)
    assert sch.alphas.shape == (T,)
    assert sch.alpha_bars.shape == (T,)
    # Monotonically decreasing cumulative alphas.
    assert np.all(np.diff(sch.alpha_bars) < 0)


def test_linear_schedule_endpoints(sols):
    sch = sols.linear_schedule(1000)
    # α̅_0 ≈ 1 - β_0 ≈ 1.
    assert sch.alpha_bars[0] > 0.999
    # α̅_T should have driven variance close to 1 on the forward chain.
    assert sch.alpha_bars[-1] < 0.01


def test_cosine_schedule_strictly_positive(sols):
    sch = sols.cosine_schedule(1000)
    assert np.all(sch.alphas > 0)
    assert np.all(sch.betas > 0)


def test_q_sample_variance_matches_closed_form(sols):
    rng = np.random.default_rng(0)
    sch = sols.linear_schedule(1000)
    x0 = rng.standard_normal((1000, 16))
    noise = rng.standard_normal((1000, 16))
    xt = sols.q_sample(x0, t=500, schedule=sch, noise=noise)
    ab = sch.alpha_bars[500]
    # E[||xt||^2 / D] ≈ ab * E[||x0||^2]/D + (1 - ab)
    mean_sq = float((xt**2).mean())
    expected = ab * 1.0 + (1 - ab) * 1.0  # both x0 and noise are unit-variance
    assert abs(mean_sq - expected) < 0.1


# -- DDIM ----------------------------------------------------------------------


def test_ddim_is_deterministic_at_eta_zero(sols):
    sch = sols.linear_schedule(200)

    def score(x, _t):
        return 0.5 * x

    a = sols.ddim_sample(score, (1, 8), sch, n_steps=20, seed=0, eta=0.0)
    b = sols.ddim_sample(score, (1, 8), sch, n_steps=20, seed=0, eta=0.0)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_ddim_varies_at_eta_one(sols):
    sch = sols.linear_schedule(200)

    def score(x, _t):
        return 0.1 * x

    a = sols.ddim_sample(score, (1, 8), sch, n_steps=20, seed=0, eta=1.0)
    b = sols.ddim_sample(score, (1, 8), sch, n_steps=20, seed=1, eta=1.0)
    assert not np.allclose(a, b)


# -- InfoNCE -------------------------------------------------------------------


def test_infonce_identity_embeddings(sols):
    eye = np.eye(5)
    loss = sols.clip_infonce_loss(eye, eye, temperature=0.07)
    # Perfect alignment on identity vectors — loss should be strictly below
    # uniform entropy log(5) ≈ 1.609.
    assert 0.0 < loss < 1.5


def test_infonce_symmetric(sols):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 4))
    b = rng.standard_normal((6, 4))
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    l1 = sols.clip_infonce_loss(a_norm, b_norm)
    l2 = sols.clip_infonce_loss(b_norm, a_norm)
    assert l1 == pytest.approx(l2, abs=1e-10)


# -- Torch-gated UNet checks ---------------------------------------------------


@pytest.fixture(scope="module")
def ddpm_module():
    pytest.importorskip("torch")
    return _load(DDPM_PATH, "w10_ddpm")


def test_small_unet_forward_shape(ddpm_module):
    import torch

    model = ddpm_module.SmallUNet(in_ch=1, base=16)
    x = torch.randn(2, 1, 28, 28)
    t = torch.zeros(2, dtype=torch.long)
    y = model(x, t)
    assert y.shape == (2, 1, 28, 28)


def test_ddpm_loss_is_scalar(ddpm_module):
    import torch

    model = ddpm_module.SmallUNet(in_ch=1, base=16)
    schedule = ddpm_module.DiffusionSchedule.linear(100)
    x0 = torch.randn(4, 1, 28, 28)
    loss = ddpm_module.ddpm_loss(model, x0, schedule)
    assert loss.ndim == 0
    assert float(loss) >= 0.0


def test_timestep_embedding_shape(ddpm_module):
    import torch

    t = torch.arange(4)
    emb = ddpm_module.timestep_embedding(t, dim=16)
    assert emb.shape == (4, 16)
