"""NumPy-only linear-models mini-library.

Implements:
  * OLS (closed-form and SGD)
  * Ridge regression (closed-form)
  * Lasso regression (coordinate descent)
  * K-fold cross-validation

Inspired by the scikit-learn API (fit / predict) but zero external dependencies
beyond NumPy. Every solver returns a `LinearModel` with consistent attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


@dataclass
class LinearModel:
    """A fitted linear model y ≈ X β + b."""

    coef: ArrayF
    intercept: float
    history: list[float]  # optional training loss per epoch / iteration

    def predict(self, X: ArrayF) -> ArrayF:
        return np.asarray(X, dtype=np.float64) @ self.coef + self.intercept


# -----------------------------------------------------------------------------
# Helpers


def _center(X: ArrayF, y: ArrayF) -> tuple[ArrayF, ArrayF, ArrayF, float]:
    """Return centered X, y and the corresponding means."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    return X - x_mean, y - y_mean, x_mean, y_mean


def _assemble(coef: ArrayF, x_mean: ArrayF, y_mean: float, history: list[float]) -> LinearModel:
    intercept = y_mean - float(x_mean @ coef)
    return LinearModel(coef=coef, intercept=intercept, history=history)


# -----------------------------------------------------------------------------
# OLS (closed form)


def fit_ols_closed_form(X: ArrayF, y: ArrayF) -> LinearModel:
    """Closed-form OLS via the pseudoinverse; robust to rank deficiency."""
    Xc, yc, x_mean, y_mean = _center(X, y)
    coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return _assemble(coef, x_mean, y_mean, history=[])


# -----------------------------------------------------------------------------
# Ridge (closed form)


def fit_ridge(X: ArrayF, y: ArrayF, *, alpha: float = 1.0) -> LinearModel:
    """Closed-form ridge: β = (XᵀX + αI)⁻¹ Xᵀy on centered data."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    Xc, yc, x_mean, y_mean = _center(X, y)
    p = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(p)
    coef = np.linalg.solve(A, Xc.T @ yc)
    return _assemble(coef, x_mean, y_mean, history=[])


# -----------------------------------------------------------------------------
# SGD (OLS + optional L2)


def fit_sgd(
    X: ArrayF,
    y: ArrayF,
    *,
    lr: float = 1e-2,
    alpha: float = 0.0,
    n_epochs: int = 50,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
) -> LinearModel:
    """Mini-batch SGD for OLS (optionally with L2 weight decay)."""
    if lr <= 0 or n_epochs <= 0 or batch_size <= 0:
        raise ValueError("lr, n_epochs, and batch_size must be positive.")
    Xc, yc, x_mean, y_mean = _center(X, y)
    rng = np.random.default_rng(seed)
    N, p = Xc.shape
    coef = np.zeros(p, dtype=np.float64)
    history: list[float] = []
    for _ in range(n_epochs):
        idx = rng.permutation(N) if shuffle else np.arange(N)
        for start in range(0, N, batch_size):
            batch = idx[start : start + batch_size]
            Xb, yb = Xc[batch], yc[batch]
            residual = Xb @ coef - yb
            grad = (Xb.T @ residual) / len(batch) + alpha * coef
            coef = coef - lr * grad
        residuals_full = Xc @ coef - yc
        history.append(float(np.mean(residuals_full**2)))
    return _assemble(coef, x_mean, y_mean, history=history)


# -----------------------------------------------------------------------------
# Lasso (coordinate descent with soft-thresholding)


def _soft_threshold(z: float, gamma: float) -> float:
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def fit_lasso(
    X: ArrayF,
    y: ArrayF,
    *,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> LinearModel:
    """Coordinate-descent lasso with soft-thresholding.

    Implements the standard sklearn-style objective:
        (1 / (2N)) ||y - Xβ||² + α ||β||₁
    on centered features and target.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    Xc, yc, x_mean, y_mean = _center(X, y)
    N, p = Xc.shape
    col_sq = (Xc**2).sum(axis=0) / N
    coef = np.zeros(p, dtype=np.float64)
    residual = yc.copy()
    history: list[float] = []

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            old = coef[j]
            # Add back the contribution of coordinate j to the residual.
            residual = residual + Xc[:, j] * old
            if col_sq[j] == 0.0:
                coef[j] = 0.0
            else:
                rho = float(Xc[:, j] @ residual / N)
                coef[j] = _soft_threshold(rho, alpha) / col_sq[j]
            residual = residual - Xc[:, j] * coef[j]
            max_delta = max(max_delta, abs(coef[j] - old))
        history.append(float(np.mean(residual**2) + alpha * np.sum(np.abs(coef))))
        if max_delta < tol:
            break
    return _assemble(coef, x_mean, y_mean, history=history)


# -----------------------------------------------------------------------------
# K-fold cross-validation


def kfold_indices(n: int, k: int, *, seed: int = 0) -> list[tuple[ArrayF, ArrayF]]:
    """Return a list of (train_idx, val_idx) splits."""
    if k < 2 or k > n:
        raise ValueError("k must satisfy 2 <= k <= n.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    splits: list[tuple[ArrayF, ArrayF]] = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, val))
    return splits


def cross_val_mse(
    fit_fn: Callable[[ArrayF, ArrayF], LinearModel],
    X: ArrayF,
    y: ArrayF,
    *,
    k: int = 5,
    seed: int = 0,
) -> tuple[float, ArrayF]:
    """K-fold CV MSE given a fit function X, y -> LinearModel.

    Returns (mean_mse, per-fold-mse).
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    scores: list[float] = []
    for train, val in kfold_indices(Xa.shape[0], k, seed=seed):
        model = fit_fn(Xa[train], ya[train])
        preds = model.predict(Xa[val])
        scores.append(float(np.mean((preds - ya[val]) ** 2)))
    arr = np.asarray(scores, dtype=np.float64)
    return float(arr.mean()), arr
