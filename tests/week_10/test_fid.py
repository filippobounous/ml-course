"""Tests for `portfolio/10_ddpm/fid.py`.

Fast tests cover the FID math via a stub feature extractor: this avoids the
~100 MB InceptionV3 download for the per-PR test suite. The slow-tier test at
the bottom exercises the real InceptionV3 pipeline on a handful of images
(opt in via `--run-slow`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FID_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "10_ddpm" / "fid.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_frechet_distance_zero_when_identical():
    """FID between a Gaussian and itself is 0 up to sqrtm round-off."""
    fid_mod = _load(FID_PATH, "fid_eq_self")
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(64, 128))
    mu, sigma = fid_mod.statistics(feats)
    d = fid_mod.frechet_distance(mu, sigma, mu, sigma)
    # scipy.linalg.sqrtm picks up a few ulps of error on a 128×128 covariance;
    # bound is conservative for a typical IEEE-754 trace.
    assert abs(d) < 1e-4, f"expected FID ≈ 0 between identical Gaussians, got {d}"


def test_frechet_distance_positive_for_shifted_means():
    """FID grows with squared mean offset when covariances match."""
    fid_mod = _load(FID_PATH, "fid_shifted_mean")
    mu1 = np.zeros(16)
    mu2 = np.ones(16) * 0.5
    sigma = np.eye(16)
    d = fid_mod.frechet_distance(mu1, sigma, mu2, sigma)
    # mu-diff norm² = 16 * 0.25 = 4; covariance terms cancel.
    assert abs(d - 4.0) < 1e-6, f"expected FID = 4 for mu-shift, got {d}"


def test_frechet_distance_symmetric():
    """FID(A, B) == FID(B, A)."""
    fid_mod = _load(FID_PATH, "fid_symmetric")
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, scale=1.0, size=(64, 32))
    b = rng.normal(loc=0.3, scale=1.5, size=(64, 32))
    mu_a, sigma_a = fid_mod.statistics(a)
    mu_b, sigma_b = fid_mod.statistics(b)
    d_ab = fid_mod.frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
    d_ba = fid_mod.frechet_distance(mu_b, sigma_b, mu_a, sigma_a)
    assert abs(d_ab - d_ba) < 1e-6, f"asymmetric: {d_ab} vs {d_ba}"


def test_frechet_distance_grows_with_noise_scale():
    """Increasing the std of one distribution monotonically grows FID."""
    fid_mod = _load(FID_PATH, "fid_monotone")
    rng = np.random.default_rng(2)
    a = rng.normal(size=(128, 8))
    mu_a, sigma_a = fid_mod.statistics(a)
    distances = []
    for scale in (1.0, 1.5, 2.0, 3.0):
        b = rng.normal(scale=scale, size=(128, 8))
        mu_b, sigma_b = fid_mod.statistics(b)
        distances.append(fid_mod.frechet_distance(mu_a, sigma_a, mu_b, sigma_b))
    # Differences may not be monotonic with N=128 noise, but a 3× scale must
    # produce a strictly larger distance than the 1× scale.
    assert distances[3] > distances[0], f"FID didn't grow with noise scale: {distances}"


def test_extract_features_handles_greyscale_input():
    """The Inception adapter accepts (N, 1, 28, 28) FashionMNIST-shaped batches."""
    torch = pytest.importorskip("torch")
    fid_mod = _load(FID_PATH, "fid_grey_in")
    # Stub feature extractor: 3×299×299 → 32-D mean. Replaces InceptionV3.
    embed_dim = 32

    class StubFeatures(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.shape[1:] == (3, 299, 299), f"unexpected shape {x.shape}"
            # Mean-pool to a fixed-dim vector deterministically.
            pooled = x.mean(dim=(2, 3))  # (N, 3)
            # Tile to embed_dim.
            return pooled.repeat(1, embed_dim // 3 + 1)[:, :embed_dim]

    images = torch.randn(8, 1, 28, 28)
    feats = fid_mod.extract_features(images, StubFeatures(), device="cpu", batch_size=4)
    assert feats.shape == (8, embed_dim)
    assert np.isfinite(feats).all()


def test_extract_features_rejects_unsupported_channel_count():
    """4-channel input is rejected with a clear error."""
    torch = pytest.importorskip("torch")
    fid_mod = _load(FID_PATH, "fid_bad_channels")

    class StubFeatures(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(2, 3))

    images = torch.randn(2, 4, 28, 28)
    with pytest.raises(ValueError, match="1 or 3 channels"):
        fid_mod.extract_features(images, StubFeatures(), device="cpu", batch_size=2)


@pytest.mark.slow
def test_inception_pipeline_end_to_end():
    """Slow tier: real InceptionV3 + tiny image batch.

    First run downloads ~100 MB of weights; subsequent runs hit the
    torchvision cache. The point is to verify the end-to-end path —
    not to compute a meaningful FID.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("scipy")
    fid_mod = _load(FID_PATH, "fid_real_inception")

    model = fid_mod.load_inception(device="cpu")
    real = torch.randn(8, 1, 28, 28)
    fake = torch.randn(8, 1, 28, 28) + 0.5  # slightly shifted
    real_feats = fid_mod.extract_features(real, model, device="cpu", batch_size=4)
    fake_feats = fid_mod.extract_features(fake, model, device="cpu", batch_size=4)
    assert real_feats.shape == (8, 2048), f"expected 2048-D pool3 features, got {real_feats.shape}"
    mu_r, sigma_r = fid_mod.statistics(real_feats)
    mu_f, sigma_f = fid_mod.statistics(fake_feats)
    fid = fid_mod.frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    assert np.isfinite(fid)
    assert fid > 0, f"expected positive FID between shifted distributions, got {fid}"
