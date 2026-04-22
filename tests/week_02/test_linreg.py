"""Auto-graded checks for the Week 2 NumPy linear-models library."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "02_numpy_linreg" / "linreg.py"


def _load():
    spec = importlib.util.spec_from_file_location("linreg", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lib():
    return _load()


@pytest.fixture
def linear_problem():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    beta = np.array([1.0, -2.0, 0.5, 0.0, 0.0])
    y = X @ beta + 0.1 * rng.standard_normal(200) + 0.3
    return X, y, beta, 0.3


def test_ols_closed_form_recovers_beta(lib, linear_problem):
    X, y, beta, b = linear_problem
    model = lib.fit_ols_closed_form(X, y)
    np.testing.assert_allclose(model.coef, beta, atol=0.05)
    assert abs(model.intercept - b) < 0.05


def test_ols_closed_form_matches_lstsq(lib, linear_problem):
    X, y, *_ = linear_problem
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    ref, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    model = lib.fit_ols_closed_form(X, y)
    np.testing.assert_allclose(model.coef, ref, atol=1e-10)


def test_ridge_matches_sklearn(lib, linear_problem):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge

    X, y, *_ = linear_problem
    ours = lib.fit_ridge(X, y, alpha=1.5)
    ref = Ridge(alpha=1.5).fit(X, y)
    np.testing.assert_allclose(ours.coef, ref.coef_, atol=1e-8)
    assert abs(ours.intercept - ref.intercept_) < 1e-8


def test_sgd_decreases_loss(lib, linear_problem):
    X, y, *_ = linear_problem
    model = lib.fit_sgd(X, y, lr=5e-2, n_epochs=30, batch_size=32, seed=0)
    assert model.history[-1] < 0.2 * model.history[0]


def test_lasso_support_recovery(lib):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 20))
    beta = np.zeros(20)
    beta[[0, 3, 7]] = [2.0, -1.5, 1.0]
    y = X @ beta + 0.05 * rng.standard_normal(300)
    model = lib.fit_lasso(X, y, alpha=0.05, max_iter=5000, tol=1e-8)
    support = set(int(i) for i in np.flatnonzero(np.abs(model.coef) > 1e-2))
    assert support == {0, 3, 7}


def test_kfold_partitions_correctly(lib):
    splits = lib.kfold_indices(50, k=5, seed=0)
    assert len(splits) == 5
    for train, val in splits:
        assert set(train).isdisjoint(val)
        assert len(train) + len(val) == 50
    all_val = np.concatenate([v for _, v in splits])
    assert sorted(all_val.tolist()) == list(range(50))


def test_cv_mse_convex_in_log_alpha(lib):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 8))
    beta = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta + 0.3 * rng.standard_normal(200)
    alphas = np.logspace(-3, 3, 7)

    def mse_at(a: float) -> float:
        mean_mse, _ = lib.cross_val_mse(lambda Xf, yf: lib.fit_ridge(Xf, yf, alpha=a), X, y, k=5)
        return mean_mse

    mses = np.array([mse_at(a) for a in alphas])
    assert mses.argmin() not in {0, len(alphas) - 1}


def test_fit_sgd_weight_decay_reduces_norm(lib, linear_problem):
    X, y, *_ = linear_problem
    m_no = lib.fit_sgd(X, y, lr=5e-2, n_epochs=40, alpha=0.0, seed=0)
    m_yes = lib.fit_sgd(X, y, lr=5e-2, n_epochs=40, alpha=0.5, seed=0)
    assert np.linalg.norm(m_yes.coef) < np.linalg.norm(m_no.coef)
