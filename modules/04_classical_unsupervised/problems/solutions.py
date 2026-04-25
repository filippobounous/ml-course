"""Week 4 — reference solutions.

Provides:
  * PCA via the SVD of the centered data matrix.
  * GMM-EM for diagonal and full covariances.
  * k-means (Lloyd) with k-means++ initialisation.
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
    components: ArrayF  # shape (k, p)
    singular_values: ArrayF  # shape (k,)
    explained_variance_ratio: ArrayF  # shape (k,)

    def transform(self, X: ArrayF) -> ArrayF:
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.components.T


def fit_pca(X: ArrayF, *, n_components: int) -> PCAFit:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    total = float((s**2).sum())
    return PCAFit(
        mean=mean,
        components=Vt[:n_components],
        singular_values=s[:n_components],
        explained_variance_ratio=(s[:n_components] ** 2) / total
        if total > 0
        else s[:n_components] * 0,
    )


# -----------------------------------------------------------------------------
# k-means (Lloyd) with k-means++ seeding


def kmeans_plus_plus_init(X: ArrayF, k: int, rng: np.random.Generator) -> ArrayF:
    N = X.shape[0]
    first = rng.integers(0, N)
    centers = [X[first]]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        probs = closest_sq / closest_sq.sum()
        idx = rng.choice(N, p=probs)
        centers.append(X[idx])
        new_sq = np.sum((X - centers[-1]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return np.stack(centers)


@dataclass
class KMeansFit:
    centers: ArrayF
    labels: NDArray[np.int64]
    inertia: float
    n_iter: int


def fit_kmeans(
    X: ArrayF, *, k: int, max_iter: int = 100, tol: float = 1e-6, seed: int = 0
) -> KMeansFit:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    centers = kmeans_plus_plus_init(X, k, rng)
    prev_inertia = np.inf
    t = 0
    for t in range(1, max_iter + 1):  # noqa: B007 — `t` is returned as n_iter
        # Pairwise squared distances; broadcasts to (N, k).
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d2.argmin(axis=1).astype(np.int64)
        inertia = float(d2[np.arange(X.shape[0]), labels].sum())
        # Update centers; leave empty clusters fixed (a rare occurrence with k-means++).
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
        if abs(prev_inertia - inertia) < tol:
            centers = new_centers
            break
        prev_inertia = inertia
        centers = new_centers
    return KMeansFit(centers=centers, labels=labels, inertia=inertia, n_iter=t)


# -----------------------------------------------------------------------------
# GMM-EM (diagonal or full covariance)


@dataclass
class GMMFit:
    weights: ArrayF  # (K,)
    means: ArrayF  # (K, D)
    covariances: ArrayF  # (K, D, D)
    log_likelihood_history: list[float]
    n_iter: int


def _gaussian_log_prob(X: ArrayF, mean: ArrayF, cov: ArrayF) -> ArrayF:
    """log N(x | mean, cov) computed via the Cholesky factor for stability."""
    D = X.shape[1]
    diff = X - mean  # (N, D)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(cov + 1e-6 * np.eye(D))
    # Solve L y = diff^T for y, then ||y||^2 gives the Mahalanobis quadratic.
    y = np.linalg.solve(chol, diff.T)
    quad = (y**2).sum(axis=0)
    log_det = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (D * np.log(2.0 * np.pi) + log_det + quad)


def fit_gmm(
    X: ArrayF,
    *,
    k: int,
    max_iter: int = 200,
    tol: float = 1e-5,
    reg: float = 1e-6,
    seed: int = 0,
) -> GMMFit:
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    rng = np.random.default_rng(seed)

    # k-means++ init for means; empirical cov for all components; uniform weights.
    means = kmeans_plus_plus_init(X, k, rng)
    cov_init = np.cov(X, rowvar=False) + reg * np.eye(D)
    covariances = np.stack([cov_init.copy() for _ in range(k)])
    weights = np.full(k, 1.0 / k)
    history: list[float] = []

    for t in range(1, max_iter + 1):
        # E-step: log-responsibilities via log-sum-exp for numerical stability.
        log_probs = np.stack(
            [
                np.log(weights[j]) + _gaussian_log_prob(X, means[j], covariances[j])
                for j in range(k)
            ],
            axis=1,
        )  # (N, K)
        max_log = log_probs.max(axis=1, keepdims=True)
        log_norm = max_log + np.log(np.exp(log_probs - max_log).sum(axis=1, keepdims=True))
        log_resp = log_probs - log_norm
        resp = np.exp(log_resp)

        ll = float(log_norm.sum())
        history.append(ll)

        # M-step.
        Nk = resp.sum(axis=0) + 1e-12
        new_means = (resp.T @ X) / Nk[:, None]
        new_covs = np.empty_like(covariances)
        for j in range(k):
            diff = X - new_means[j]
            new_covs[j] = (resp[:, j : j + 1] * diff).T @ diff / Nk[j] + reg * np.eye(D)
        new_weights = Nk / N

        if t > 1 and abs(history[-1] - history[-2]) < tol:
            weights, means, covariances = new_weights, new_means, new_covs
            break
        weights, means, covariances = new_weights, new_means, new_covs

    return GMMFit(
        weights=weights,
        means=means,
        covariances=covariances,
        log_likelihood_history=history,
        n_iter=t,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X1 = rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 1.0]], 400)
    X2 = rng.multivariate_normal([4, 4], [[1.0, -0.2], [-0.2, 1.0]], 400)
    X = np.vstack([X1, X2])
    fit = fit_gmm(X, k=2, max_iter=100)
    print("weights:", fit.weights.round(3))
    print("means:", fit.means.round(3))
    print("ll history (first/last):", fit.log_likelihood_history[0], fit.log_likelihood_history[-1])
