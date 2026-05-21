"""Fréchet Inception Distance for the W10 DDPM ablation.

Replaces the pixel-statistics proxy used in the original `ablate.py` with the
canonical FID metric: extract 2048-D pool3 features from a pretrained
InceptionV3, fit Gaussians to the real and generated feature distributions,
return $\\|\\mu_r - \\mu_g\\|^2 + \\operatorname{tr}(\\Sigma_r + \\Sigma_g - 2(\\Sigma_r \\Sigma_g)^{1/2})$.

InceptionV3 expects $3 \\times 299 \\times 299$ ImageNet-normalised images.
FashionMNIST is $1 \\times 28 \\times 28$ in $[-1, 1]$. The adapter in
`extract_features` handles the channel-repeat + bilinear upsample +
ImageNet normalisation in a single pass per batch.

Numerical note: $\\operatorname{tr}\\!\\left((\\Sigma_r \\Sigma_g)^{1/2}\\right)$
is computed via `scipy.linalg.sqrtm`; on near-singular covariances the
matrix square root can pick up tiny imaginary components from floating-point
error. We take the real part and add a small diagonal regulariser if needed.

Usage:

    feature_model = load_inception(device)
    real_feats = extract_features(real_images, feature_model, device)
    fake_feats = extract_features(fake_images, feature_model, device)
    fid = frechet_distance(*statistics(real_feats), *statistics(fake_feats))

Reference: Heusel et al. 2017, "GANs Trained by a Two Time-Scale Update Rule".
"""

from __future__ import annotations

from collections.abc import Callable

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "FID requires numpy + torch. Install with `pip install -e '.[dl,diffusion]'`."
    ) from e


# ImageNet preprocessing constants (Inception was trained on these stats).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_inception(device: str = "cpu") -> torch.nn.Module:
    """Return a pretrained InceptionV3 with the final FC layer replaced by Identity.

    The output is the 2048-D pool3 feature vector — the canonical FID feature.
    """
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = torch.nn.Identity()  # type: ignore[assignment]
    model.eval()
    return model.to(device)


def _prepare_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Adapt a $[-1, 1]$-scaled image batch to InceptionV3's expected input.

    Handles:
      1. Channel repeat (greyscale → RGB) if `images` is 1-channel.
      2. Bilinear upsample to $3 \\times 299 \\times 299$.
      3. Rescale from $[-1, 1]$ to $[0, 1]$, then apply ImageNet stats.

    Args:
        images: `(N, C, H, W)` with $C \\in \\{1, 3\\}$ and pixel range $[-1, 1]$.
    """
    if images.dim() != 4:
        raise ValueError(f"expected a 4-D tensor (N, C, H, W); got shape {tuple(images.shape)}")
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError(f"expected 1 or 3 channels; got {images.shape[1]}")
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    # [-1, 1] -> [0, 1] -> ImageNet-normalised.
    images = (images + 1.0) / 2.0
    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_features(
    images: torch.Tensor,
    feature_model: Callable[[torch.Tensor], torch.Tensor],
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Run `feature_model` on `images` in mini-batches, return a `(N, D)` array.

    `feature_model` may be the InceptionV3 returned by `load_inception` *or*
    any module that takes a batch of InceptionV3-shaped inputs and returns
    a `(B, D)` feature tensor — handy for tests with a stub extractor.

    Args:
        images: `(N, C, H, W)` in $[-1, 1]$.
        feature_model: callable returning $(B, D)$ embeddings.
        device: where to run `feature_model`.
        batch_size: mini-batch size for the forward pass.
    """
    feats: list[np.ndarray] = []
    images = images.to(device)
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size]
            prepped = _prepare_for_inception(batch)
            out = feature_model(prepped)
            if isinstance(out, tuple):  # aux_logits=True → (logits, aux)
                out = out[0]
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical mean vector and covariance matrix of `features`."""
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """The FID between two Gaussians fit to feature distributions.

    Heusel 2017 eq. 6:
        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁ Σ₂)^{1/2}).
    """
    from scipy import linalg

    mu_diff = mu1 - mu2
    # Σ₁ Σ₂ can be near-singular; nudge with an ε I on the diagonal.
    cov_prod = sigma1.dot(sigma2)
    sqrt_cov = linalg.sqrtm(cov_prod)
    if not np.isfinite(sqrt_cov).all():
        offset = np.eye(sigma1.shape[0]) * eps
        sqrt_cov = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # The matrix sqrt may have small imaginary parts from floating-point noise.
    if np.iscomplexobj(sqrt_cov):
        if not np.allclose(np.diagonal(sqrt_cov).imag, 0, atol=1e-3):
            raise ValueError("sqrtm produced significant imaginary diagonal")
        sqrt_cov = sqrt_cov.real
    return float(
        mu_diff.dot(mu_diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_cov)
    )


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_model: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    """Convenience: extract features from both sets, fit Gaussians, return FID.

    If `feature_model` is None, loads InceptionV3 internally.
    """
    if feature_model is None:
        feature_model = load_inception(device)
    real_feats = extract_features(real_images, feature_model, device=device, batch_size=batch_size)
    fake_feats = extract_features(fake_images, feature_model, device=device, batch_size=batch_size)
    mu_r, sigma_r = statistics(real_feats)
    mu_f, sigma_f = statistics(fake_feats)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
