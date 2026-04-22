"""Auto-graded checks for the Week 1 problem set implementations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "01_math_foundations"
    / "problems"
    / "solutions.py"
)


def _load_solutions():
    import sys

    spec = importlib.util.spec_from_file_location("w1_solutions", MODULE_PATH)
    assert spec and spec.loader, f"Could not load {MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass with `from __future__ import annotations`
    # can resolve forward references via typing.get_type_hints.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load_solutions()


def test_pseudoinverse_matches_numpy(sols):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 8))
    np.testing.assert_allclose(sols.pseudoinverse_via_svd(A), np.linalg.pinv(A), atol=1e-10)


def test_pseudoinverse_rank_deficient(sols):
    rng = np.random.default_rng(1)
    U = rng.standard_normal((30, 5))
    # Rank-5 matrix embedded in a 10-column space.
    A = U @ rng.standard_normal((5, 10))
    ours = sols.pseudoinverse_via_svd(A, rcond=1e-10)
    ref = np.linalg.pinv(A, rcond=1e-10)
    # Same action on the column space even if the pseudoinverses aren't equal elementwise.
    np.testing.assert_allclose(A @ ours @ A, A @ ref @ A, atol=1e-8)


def test_ridge_pseudoinverse_approaches_pinv_as_lambda_goes_to_zero(sols):
    rng = np.random.default_rng(2)
    A = rng.standard_normal((50, 6))
    diff_large = np.linalg.norm(sols.pseudoinverse_via_ridge(A, lam=1e-1) - np.linalg.pinv(A))
    diff_small = np.linalg.norm(sols.pseudoinverse_via_ridge(A, lam=1e-10) - np.linalg.pinv(A))
    assert diff_small < diff_large


def test_gaussian_mle_recovers_moments(sols):
    rng = np.random.default_rng(3)
    mu_true = np.array([1.0, -2.0, 0.5])
    cov_true = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, -0.1], [0.0, -0.1, 0.8]])
    X = rng.multivariate_normal(mu_true, cov_true, size=20_000)
    fit = sols.gaussian_mle(X)
    np.testing.assert_allclose(fit.mean, mu_true, atol=0.05)
    np.testing.assert_allclose(fit.cov_unbiased, cov_true, atol=0.1)
    # Biased estimator uses N; unbiased uses N-1 → cov_biased < cov_unbiased elementwise in magnitude.
    factor = (X.shape[0] - 1) / X.shape[0]
    np.testing.assert_allclose(fit.cov_biased, fit.cov_unbiased * factor, atol=1e-12)


def test_langevin_equilibrates_to_gibbs(sols):
    rng = np.random.default_rng(4)
    # Longer trajectory for an adequate histogram.
    traj = sols.simulate_langevin(
        sols.double_well_grad,
        x0=np.array(0.0),
        dt=5e-3,
        n_steps=200_000,
        rng=rng,
    )
    burn = 10_000
    samples = traj[burn:]
    # The Gibbs distribution ∝ exp(-U) is symmetric, so the mean should be ~0.
    assert abs(samples.mean()) < 0.2
    # And it should concentrate near the wells at x = ±1.
    assert np.mean(np.abs(samples) > 0.3) > 0.5
