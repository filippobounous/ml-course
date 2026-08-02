"""Week 3 — reference solutions.

Provides an IRLS (Newton) logistic-regression solver and helpers for
information-gain / gini computations. The tabular benchmark lives under
`portfolio/03_tabular_benchmark/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# IRLS logistic regression


@dataclass
class LogisticIRLS:
    coef: ArrayF
    intercept: float
    n_iter: int
    history: list[float]

    def predict_proba(self, X: ArrayF) -> ArrayF:
        z = np.asarray(X, dtype=np.float64) @ self.coef + self.intercept
        return _sigmoid(z)

    def predict(self, X: ArrayF, threshold: float = 0.5) -> ArrayF:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


def _sigmoid(z: ArrayF) -> ArrayF:
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def fit_logistic_irls(
    X: ArrayF,
    y: ArrayF,
    *,
    alpha: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> LogisticIRLS:
    """Logistic regression via IRLS (Newton) on the augmented matrix.

    Minimises
        -Σ [y_i log σ(xᵀβ + b) + (1-y_i) log(1-σ(xᵀβ + b))] + (α/2) ||β||²
    with an unpenalised intercept. Quadratic convergence once near the optimum.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch.")
    N, p = X.shape

    # Augment with intercept column.
    X_aug = np.hstack([np.ones((N, 1)), X])
    beta = np.zeros(p + 1)
    history: list[float] = []

    reg = alpha * np.eye(p + 1)
    reg[0, 0] = 0.0  # do not penalise the intercept

    t = 0
    for t in range(1, max_iter + 1):  # noqa: B007 — `t` is returned as n_iter
        eta = X_aug @ beta
        p_hat = _sigmoid(eta)
        # Clip to keep W non-singular.
        p_hat = np.clip(p_hat, 1e-12, 1 - 1e-12)
        W = p_hat * (1.0 - p_hat)
        # Closed-form Newton step (avoid forming the diagonal matrix).
        g = X_aug.T @ (p_hat - y) + reg @ beta
        H = (X_aug.T * W) @ X_aug + reg
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta - step
        nll = float(
            -np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
            + 0.5 * alpha * beta[1:] @ beta[1:]
        )
        history.append(nll)
        if np.linalg.norm(step) < tol:
            break

    return LogisticIRLS(coef=beta[1:], intercept=float(beta[0]), n_iter=t, history=history)


# -----------------------------------------------------------------------------
# Information gain helpers (for decision-tree intuition)


def gini(p: ArrayF) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(1.0 - np.sum(p**2))


def entropy(p: ArrayF, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def information_gain(parent_probs: ArrayF, child_sets: list[tuple[float, ArrayF]]) -> float:
    """IG = H(parent) − Σ w_i H(child_i).

    `child_sets` is a list of (weight, class-probability-vector) tuples; the
    weights should sum to 1.
    """
    total = sum(w for w, _ in child_sets)
    if not np.isclose(total, 1.0):
        raise ValueError(f"child weights must sum to 1, got {total:.4f}")
    return entropy(parent_probs) - sum(w * entropy(p) for w, p in child_sets)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, p = 400, 6
    X = rng.standard_normal((N, p))
    beta_true = np.array([1.0, -1.5, 0.5, 0.0, 0.0, 0.0])
    logits = X @ beta_true + 0.2
    y = (rng.uniform(size=N) < _sigmoid(logits)).astype(np.int64)
    fit = fit_logistic_irls(X, y, alpha=0.01)
    print("β̂:", fit.coef.round(3))
    print("b̂:", round(fit.intercept, 3))
    print("iters:", fit.n_iter)
