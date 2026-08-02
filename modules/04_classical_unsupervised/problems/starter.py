"""Week 04 — problem-set starter. **This is the file you edit.**

`pytest tests/week_04` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_04
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# PCA (via SVD)


@dataclass
class PCAFit:
    mean: ArrayF
    components: ArrayF
    singular_values: ArrayF
    explained_variance_ratio: ArrayF

    def transform(self, X: ArrayF) -> ArrayF:
        raise NotImplementedError("transform")


def fit_pca(X: ArrayF, *, n_components: int) -> PCAFit:
    raise NotImplementedError("fit_pca")


# -----------------------------------------------------------------------------
# k-means (Lloyd) with k-means++ seeding


def kmeans_plus_plus_init(X: ArrayF, k: int, rng: np.random.Generator) -> ArrayF:
    raise NotImplementedError("kmeans_plus_plus_init")


@dataclass
class KMeansFit:
    centers: ArrayF
    labels: NDArray[np.int64]
    inertia: float
    n_iter: int


def fit_kmeans(
    X: ArrayF, *, k: int, max_iter: int = 100, tol: float = 1e-06, seed: int = 0
) -> KMeansFit:
    raise NotImplementedError("fit_kmeans")


# -----------------------------------------------------------------------------
# GMM-EM (diagonal or full covariance)


@dataclass
class GMMFit:
    weights: ArrayF
    means: ArrayF
    covariances: ArrayF
    log_likelihood_history: list[float]
    n_iter: int


def _gaussian_log_prob(X: ArrayF, mean: ArrayF, cov: ArrayF) -> ArrayF:
    """log N(x | mean, cov) computed via the Cholesky factor for stability."""
    raise NotImplementedError("_gaussian_log_prob")


def fit_gmm(
    X: ArrayF, *, k: int, max_iter: int = 200, tol: float = 1e-05, reg: float = 1e-06, seed: int = 0
) -> GMMFit:
    raise NotImplementedError("fit_gmm")
