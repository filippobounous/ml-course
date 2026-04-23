"""Week 12 — Cole-Hopf IC, walk-forward splits, TCA Sharpe, PINN shapes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "12_applied_capstone"
    / "problems"
    / "solutions.py"
)
PINN_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "12_capstone" / "pinn_burgers.py"
STATARB_PATH = (
    Path(__file__).resolve().parents[2] / "portfolio" / "12_capstone" / "statarb_walkforward.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load(SOLUTIONS_PATH, "w12_solutions")


# -- Cole-Hopf -----------------------------------------------------------------


def test_cole_hopf_recovers_ic_at_t0(sols):
    x = np.linspace(-1, 1, 51)
    t = np.array([0.0])
    U = sols.burgers_cole_hopf(x, t)
    np.testing.assert_allclose(U[0], -np.sin(np.pi * x), atol=1e-12)


def test_cole_hopf_zero_boundary(sols):
    # Analytical BC: u(±1, t) = 0 for all t > 0.
    x = np.array([-1.0, 1.0])
    t = np.array([0.1, 0.5, 1.0])
    U = sols.burgers_cole_hopf(x, t)
    np.testing.assert_allclose(U, np.zeros_like(U), atol=1e-6)


# -- Walk-forward splits -------------------------------------------------------


def test_walk_forward_splits_partition(sols):
    splits = sols.walk_forward_splits(n=100, train_size=30, val_size=10)
    # Each val window is contiguous and ordered.
    val_starts = [s.val_idx.start for s in splits]
    assert val_starts == sorted(val_starts)
    # Consecutive val windows do not overlap.
    from itertools import pairwise

    for a, b in pairwise(splits):
        assert a.val_idx.stop <= b.val_idx.start


def test_walk_forward_splits_respect_embargo(sols):
    splits = sols.walk_forward_splits(n=200, train_size=50, val_size=20, embargo=5)
    # After the first fold, each train's upper bound is >= previous val end + embargo.
    for k, sp in enumerate(splits):
        if k == 0:
            continue
        prev_val_end = splits[k - 1].val_idx.stop
        assert sp.train_idx.stop >= prev_val_end + 5


# -- TCA Sharpe ----------------------------------------------------------------


def test_annualised_sharpe_zero_turnover_unchanged(sols):
    rng = np.random.default_rng(0)
    returns = rng.normal(scale=0.01, size=252) + 0.0005
    zero_cost = sols.annualised_sharpe(returns, cost_bps=5.0, turnover=np.zeros(252))
    baseline = sols.annualised_sharpe(returns)
    assert zero_cost == pytest.approx(baseline, rel=1e-12)


def test_annualised_sharpe_monotone_in_cost(sols):
    rng = np.random.default_rng(0)
    returns = rng.normal(scale=0.01, size=252) + 0.001
    turnover = np.full(252, 0.1)
    no_cost = sols.annualised_sharpe(returns, cost_bps=0.0, turnover=turnover)
    low = sols.annualised_sharpe(returns, cost_bps=1.0, turnover=turnover)
    high = sols.annualised_sharpe(returns, cost_bps=20.0, turnover=turnover)
    assert no_cost > low > high


def test_sharpe_leakage_demo(sols):
    biased, honest = sols.sharpe_leakage_demo(n=1000, n_features=400, seed=0)
    # Biased selection on pure noise produces a large (spurious) Sharpe;
    # honest walk-forward is close to zero.
    assert abs(biased) > 2.0 * abs(honest)


# -- Torch-gated PINN checks ---------------------------------------------------


@pytest.fixture(scope="module")
def pinn_mod():
    pytest.importorskip("torch")
    return _load(PINN_PATH, "w12_pinn")


def test_pinn_forward_shape(pinn_mod):
    import torch

    model = pinn_mod.PINN(hidden=32, depth=4)
    x = torch.randn(10, 1)
    t = torch.rand(10, 1)
    u = model(x, t)
    assert u.shape == (10, 1)


def test_pde_residual_shape(pinn_mod):
    import torch

    model = pinn_mod.PINN(hidden=16, depth=3)
    x = torch.randn(8, 1)
    t = torch.rand(8, 1)
    r = pinn_mod.pde_residual(model, x, t, nu=0.01)
    assert r.shape == (8, 1)


def test_gradnorm_reweighter_brings_grad_norms_toward_parity(pinn_mod):
    """Two losses with dramatically different gradient magnitudes should have
    their weights adjusted so the *weighted* gradient norms become comparable.
    """
    import torch

    torch.manual_seed(0)
    # One small tanh MLP, two losses: loss_a is scaled 1e-3, loss_b scaled 1.0
    # — raw gradient norms differ by 1000×.
    net = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    x = torch.randn(8, 1)
    out = net(x)
    loss_a = 1e-3 * (out**2).mean()
    loss_b = 1.0 * ((out - 1.0) ** 2).mean()

    reweighter = pinn_mod.GradNormReweighter(["a", "b"], alpha=0.0)  # no EMA
    weights = reweighter.step({"a": loss_a, "b": loss_b}, list(net.parameters()))

    # weighted grads should now be roughly equal in L2 norm.
    def grad_norm(loss, params):
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        return float(sum((g**2).sum() for g in grads) ** 0.5)

    norm_a = weights["a"] * grad_norm(loss_a, list(net.parameters()))
    norm_b = weights["b"] * grad_norm(loss_b, list(net.parameters()))
    # After reweighting with alpha=0, they should match to within a factor of 2.
    assert 0.5 < norm_a / norm_b < 2.0
    # And the raw ratio should be much worse (bigger than 10× gap).
    raw_ratio = grad_norm(loss_a, list(net.parameters())) / max(
        grad_norm(loss_b, list(net.parameters())), 1e-12
    )
    assert raw_ratio < 0.1 or raw_ratio > 10.0


def test_gradnorm_reweighter_preserves_keys(pinn_mod):
    import torch

    net = torch.nn.Linear(2, 1)
    x = torch.randn(4, 2)
    losses = {"res": (net(x) ** 2).mean(), "ic": ((net(x) - 1) ** 2).mean()}
    reweighter = pinn_mod.GradNormReweighter(list(losses), alpha=0.5)
    weights = reweighter.step(losses, list(net.parameters()))
    assert set(weights.keys()) == {"res", "ic"}
    for w in weights.values():
        assert w > 0.0


def test_pinn_config_exposes_gradnorm_option(pinn_mod):
    cfg = pinn_mod.PINNConfig(loss_weighting="gradnorm")
    assert cfg.loss_weighting == "gradnorm"
    assert cfg.reweight_every > 0


# -- Walk-forward stat-arb (uses the torch-free Week-4 engine) -----------------


def test_walk_forward_statarb_nonzero_pnl():
    wf = _load(STATARB_PATH, "w12_statarb")
    w4 = wf._load_week4()
    returns = w4.simulate_returns(n_periods=1000, n_assets=20, n_factors=2, seed=0)
    report = wf.walk_forward_statarb(returns, train_size=300, val_size=100, embargo=5, cost_bps=5.0)
    # At least two folds, non-empty metrics.
    assert len(report.windows) >= 2
    # Gross Sharpe should be >= net Sharpe (costs cannot help).
    assert report.gross_sharpe >= report.net_sharpe - 1e-9
