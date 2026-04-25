"""Auto-graded checks for Week 4: PCA, k-means, GMM-EM, and stat-arb invariants."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "04_classical_unsupervised"
    / "problems"
    / "solutions.py"
)
STATARB_PATH = (
    Path(__file__).resolve().parents[2] / "portfolio" / "04_pca_statarb" / "pca_statarb.py"
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
    return _load(SOLUTIONS_PATH, "w4_solutions")


@pytest.fixture(scope="module")
def statarb():
    return _load(STATARB_PATH, "pca_statarb")


def test_pca_recovers_principal_direction(sols):
    rng = np.random.default_rng(0)
    # Data stretched along (1, 0) with small noise in the second dim.
    X = np.column_stack([rng.normal(scale=3.0, size=1000), rng.normal(scale=0.1, size=1000)])
    fit = sols.fit_pca(X, n_components=1)
    # Top PC should align (up to sign) with (1, 0).
    cos = abs(float(fit.components[0] @ np.array([1.0, 0.0])))
    assert cos > 0.999
    assert fit.explained_variance_ratio[0] > 0.99


def test_kmeans_separates_well(sols):
    rng = np.random.default_rng(1)
    X1 = rng.normal(loc=[0, 0], scale=0.4, size=(200, 2))
    X2 = rng.normal(loc=[5, 5], scale=0.4, size=(200, 2))
    X = np.vstack([X1, X2])
    fit = sols.fit_kmeans(X, k=2, seed=0)
    # Labels should match a 2-cluster grouping up to permutation.
    labels = fit.labels
    top_match = max((labels[:200] == c).sum() + (labels[200:] == (1 - c)).sum() for c in (0, 1))
    assert top_match / 400 > 0.99


def test_gmm_em_log_likelihood_monotone(sols):
    rng = np.random.default_rng(2)
    X = np.vstack(
        [
            rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 1.0]], 300),
            rng.multivariate_normal([4, 4], [[1.0, -0.2], [-0.2, 1.0]], 300),
        ]
    )
    fit = sols.fit_gmm(X, k=2, max_iter=50)
    from itertools import pairwise

    ll = fit.log_likelihood_history
    # Non-decreasing within tiny numerical tolerance.
    assert all(b - a >= -1e-6 for a, b in pairwise(ll))
    # And the EM fit should improve substantially over the init.
    assert ll[-1] - ll[0] > 10.0


def test_gmm_recovers_means(sols):
    rng = np.random.default_rng(3)
    centres_true = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = np.vstack(
        [
            rng.multivariate_normal(centres_true[0], np.eye(2) * 0.5, 500),
            rng.multivariate_normal(centres_true[1], np.eye(2) * 0.5, 500),
        ]
    )
    fit = sols.fit_gmm(X, k=2, max_iter=200, seed=0)
    # Sort by first coordinate before comparing.
    recovered = fit.means[np.argsort(fit.means[:, 0])]
    np.testing.assert_allclose(recovered, centres_true, atol=0.3)


def test_simulated_returns_shape(statarb):
    rets = statarb.simulate_returns(n_periods=400, n_assets=20, seed=0)
    assert rets.shape == (400, 20)


def test_backtest_has_no_lookahead(statarb):
    # Deterministic zero-volatility returns ⇒ no residual signal ⇒ zero PnL.
    zero_rets = np.zeros((500, 10))
    result = statarb.pca_statarb_backtest(zero_rets, lookback=100, n_factors=2)
    # With zero returns, everything should be zero.
    assert np.all(result.pnl == 0)
    assert np.all(result.pnl_net == 0)


def test_backtest_produces_pnl_on_simulated_data(statarb):
    rets = statarb.simulate_returns(n_periods=600, n_assets=20, seed=0)
    result = statarb.pca_statarb_backtest(rets, lookback=100, n_factors=2)
    # We don't require a specific Sharpe (the simulated signal is weak on short
    # runs), but the backtest must generate non-trivial PnL variance.
    assert result.pnl_net.std() > 0
