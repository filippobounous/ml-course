"""Week 1 — reference solutions for the implementation parts of the problem set.

Covers:
  * Moore-Penrose pseudoinverse via SVD vs ridge-regularised normal equations.
  * MLE of a multivariate Gaussian (biased vs unbiased covariance).
  * Euler-Maruyama simulation of the 1-D double-well Langevin SDE.

The numerical tests under `tests/week_01/` check correctness against either
`numpy.linalg.pinv`/`numpy.cov` or against the analytical Gibbs distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Problem 5 — Moore-Penrose pseudoinverse


def pseudoinverse_via_svd(A: NDArray[np.float64], rcond: float = 1e-12) -> NDArray[np.float64]:
    """A^+ via the thin SVD, truncating singular values below `rcond * sigma_max`.

    This is the textbook definition: A = U Σ V^T ⇒ A^+ = V Σ^+ U^T.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((A.shape[1], A.shape[0]))
    cutoff = rcond * s.max()
    s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


def pseudoinverse_via_ridge(A: NDArray[np.float64], lam: float = 1e-8) -> NDArray[np.float64]:
    """Ridge-regularised pseudoinverse: (A^T A + λI)^-1 A^T.

    Equivalent to replacing σ with σ / (σ² + λ) in the SVD — numerically stable
    for near-rank-deficient A at the cost of a small bias.
    """
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T)


# -----------------------------------------------------------------------------
# Problem 6 — Gaussian maximum likelihood


@dataclass(frozen=True)
class GaussianMLE:
    mean: NDArray[np.float64]
    cov_biased: NDArray[np.float64]
    cov_unbiased: NDArray[np.float64]


def gaussian_mle(X: NDArray[np.float64]) -> GaussianMLE:
    """MLE of a multivariate Gaussian from rows of `X`.

    Returns both the biased (MLE, divide by N) and the unbiased (divide by N-1)
    covariance estimates.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must have shape (N, D) with N >= 2.")
    N = X.shape[0]
    mean = X.mean(axis=0)
    centred = X - mean
    cov_biased = (centred.T @ centred) / N
    cov_unbiased = (centred.T @ centred) / (N - 1)
    return GaussianMLE(mean=mean, cov_biased=cov_biased, cov_unbiased=cov_unbiased)


# -----------------------------------------------------------------------------
# Problem 7 — Double-well Langevin SDE


def double_well_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """∇U for U(x) = x^4/4 - x^2/2, so ∇U(x) = x^3 - x."""
    return x**3 - x


def simulate_langevin(
    grad_U,
    x0: NDArray[np.float64],
    dt: float,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Euler-Maruyama simulation of dX = -∇U(X) dt + sqrt(2) dW.

    Returns an array of shape (n_steps + 1, ...) whose first slice is `x0`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x0, dtype=np.float64)
    traj = np.empty((n_steps + 1, *x.shape), dtype=np.float64)
    traj[0] = x
    sqrt_2dt = np.sqrt(2.0 * dt)
    for t in range(n_steps):
        noise = rng.standard_normal(x.shape)
        x = x - grad_U(x) * dt + sqrt_2dt * noise
        traj[t + 1] = x
    return traj


def double_well_potential(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.25 * x**4 - 0.5 * x**2


if __name__ == "__main__":
    # Quick self-check (also runs in `tests/week_01/`).
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 10))
    A_svd = pseudoinverse_via_svd(A)
    A_ref = np.linalg.pinv(A)
    print("pseudoinverse max error:", np.max(np.abs(A_svd - A_ref)))

    X = rng.multivariate_normal(np.zeros(3), np.eye(3), size=5000)
    fit = gaussian_mle(X)
    print("mean error:", np.linalg.norm(fit.mean))
    print("cov diag:", np.diag(fit.cov_unbiased).round(3))

    traj = simulate_langevin(double_well_grad, np.array(0.5), dt=5e-3, n_steps=200_000)
    print("Langevin mean |x|:", np.mean(np.abs(traj[1000:])))
