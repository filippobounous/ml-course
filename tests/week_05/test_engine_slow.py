"""Slow integration test for the Week-5 micrograd engine.

Trains the scalar MLP on two-moons for a handful of epochs and asserts test
accuracy crosses a hard threshold. This is the **verification** that our
gradient-correctness unit tests imply end-to-end learning behaviour.

Skipped by default; run with `pytest --run-slow -q`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlcourse.autograd import MLP, Adam, Value


def _two_moons(n: int = 200, noise: float = 0.18, seed: int = 0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, np.pi, n1)
    theta2 = rng.uniform(0, np.pi, n2)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    X2 = np.column_stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)])
    X = np.vstack([X1, X2]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([np.zeros(n1, dtype=np.float64), np.ones(n2, dtype=np.float64)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _bce(logit: Value, target: float) -> Value:
    p = logit.sigmoid()
    eps = 1e-7
    if target > 0.5:
        return -(p + eps).log()
    return -((1 - p) + eps).log()


@pytest.mark.slow
def test_micrograd_mlp_reaches_high_accuracy_on_two_moons():
    X_train, y_train = _two_moons(n=200, noise=0.18, seed=0)
    X_test, y_test = _two_moons(n=200, noise=0.18, seed=1)

    mlp = MLP(n_in=2, n_outs=[8, 8, 1], hidden_activation="tanh", seed=0)
    opt = Adam(mlp.parameters(), lr=3e-2)

    rng = np.random.default_rng(0)
    for _ in range(40):  # epochs
        for i in rng.permutation(len(X_train)):
            logit = mlp([float(X_train[i, 0]), float(X_train[i, 1])])
            assert isinstance(logit, Value)
            loss = _bce(logit, float(y_train[i]))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate.
    correct = 0
    for i in range(len(X_test)):
        logit = mlp([float(X_test[i, 0]), float(X_test[i, 1])])
        assert isinstance(logit, Value)
        pred = 1.0 if logit.data > 0.0 else 0.0
        correct += int(pred == y_test[i])
    acc = correct / len(X_test)

    # Threshold set below the 97% observed by demo.py with a wider margin so
    # this test is robust to minor numerical drift on different machines.
    assert acc >= 0.88, f"two-moons test accuracy {acc:.3f} below 0.88 threshold"
