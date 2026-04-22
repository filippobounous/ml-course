"""Auto-graded checks for the Week 3 problem set (IRLS + info-gain helpers)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "03_classical_supervised"
    / "problems"
    / "solutions.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("w3_solutions", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load()


def _make_logistic_problem(n=500, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.array([1.2, -0.8, 0.5, 0.0, 0.0])
    logits = X @ beta + 0.3
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(np.int64)
    return X, y, beta


def test_irls_converges(sols):
    X, y, _ = _make_logistic_problem()
    fit = sols.fit_logistic_irls(X, y, alpha=0.01, max_iter=50, tol=1e-8)
    assert fit.n_iter < 25, f"IRLS should converge fast; got {fit.n_iter} iterations"
    # NLL should be monotone non-increasing.
    assert all(a >= b - 1e-9 for a, b in zip(fit.history, fit.history[1:], strict=False))


def test_irls_matches_sklearn(sols):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X, y, _ = _make_logistic_problem(n=1000, p=5)
    ours = sols.fit_logistic_irls(X, y, alpha=1.0)
    # sklearn's `C` is inverse-regularisation in (1 / (2α)) conventions; match it.
    sk = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500).fit(X, y)
    # Close enough; both are optimising the same convex objective up to conventions.
    np.testing.assert_allclose(ours.coef, sk.coef_[0], atol=0.05)
    assert abs(ours.intercept - sk.intercept_[0]) < 0.1


def test_predict_proba_shape(sols):
    X, y, _ = _make_logistic_problem()
    fit = sols.fit_logistic_irls(X, y)
    proba = fit.predict_proba(X)
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_gini_and_entropy_bounds(sols):
    uniform = np.array([0.5, 0.5])
    skewed = np.array([1.0, 0.0])
    assert sols.entropy(uniform) == pytest.approx(1.0)
    assert sols.entropy(skewed) == pytest.approx(0.0, abs=1e-6)
    assert sols.gini(uniform) == pytest.approx(0.5)
    assert sols.gini(skewed) == pytest.approx(0.0)


def test_information_gain_on_perfect_split(sols):
    parent = np.array([0.5, 0.5])
    # Perfect split: one child all class 0, the other all class 1.
    gain = sols.information_gain(
        parent,
        [(0.5, np.array([1.0, 0.0])), (0.5, np.array([0.0, 1.0]))],
    )
    assert gain == pytest.approx(1.0)


def test_information_gain_on_useless_split(sols):
    parent = np.array([0.5, 0.5])
    gain = sols.information_gain(
        parent,
        [(0.5, np.array([0.5, 0.5])), (0.5, np.array([0.5, 0.5]))],
    )
    assert gain == pytest.approx(0.0, abs=1e-9)
