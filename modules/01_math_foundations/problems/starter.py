"""Week 01 — problem-set starter. **This is the file you edit.**

`pytest tests/week_01` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_01
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
    raise NotImplementedError("pseudoinverse_via_svd")


def pseudoinverse_via_ridge(A: NDArray[np.float64], lam: float = 1e-08) -> NDArray[np.float64]:
    """Ridge-regularised pseudoinverse: (A^T A + λI)^-1 A^T.

    Equivalent to replacing σ with σ / (σ² + λ) in the SVD — numerically stable
    for near-rank-deficient A at the cost of a small bias.
    """
    raise NotImplementedError("pseudoinverse_via_ridge")


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
    raise NotImplementedError("gaussian_mle")


# -----------------------------------------------------------------------------
# Problem 7 — Double-well Langevin SDE


def double_well_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """∇U for U(x) = x^4/4 - x^2/2, so ∇U(x) = x^3 - x."""
    raise NotImplementedError("double_well_grad")


def simulate_langevin(
    grad_U, x0: NDArray[np.float64], dt: float, n_steps: int, rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Euler-Maruyama simulation of dX = -∇U(X) dt + sqrt(2) dW.

    Returns an array of shape (n_steps + 1, ...) whose first slice is `x0`.
    """
    raise NotImplementedError("simulate_langevin")


def double_well_potential(x: NDArray[np.float64]) -> NDArray[np.float64]:
    raise NotImplementedError("double_well_potential")
