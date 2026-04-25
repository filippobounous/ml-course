"""Week 7 — manual NumPy conv + torch-gated classifier checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "modules" / "07_cnns_vision" / "problems" / "solutions.py"
)
CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2] / "portfolio" / "07_vision_classifier" / "classifier.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load(SOLUTIONS_PATH, "w7_solutions")


def test_conv2d_shape(sols):
    x = np.zeros((3, 8, 8))
    w = np.zeros((5, 3, 3, 3))
    y = sols.conv2d_forward(x, w)
    assert y.shape == (5, 6, 6)


def test_conv2d_matches_numerical_reference(sols):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    y = sols.conv2d_forward(x, w, b)
    # Manually compute one output cell and compare.
    c_out, i, j = 1, 2, 2
    patch = x[:, i : i + 3, j : j + 3]
    expected = float((patch * w[c_out]).sum()) + b[c_out]
    assert y[c_out, i, j] == pytest.approx(expected, rel=1e-10)


def test_conv2d_backward_matches_numerical_grad(sols):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5, 5))
    w = rng.standard_normal((2, 2, 3, 3))
    b = rng.standard_normal(2)
    y = sols.conv2d_forward(x, w, b)
    # Scalar loss L = sum(y).
    dy = np.ones_like(y)
    _dx, dw, db = sols.conv2d_backward(dy, x, w)

    # Numerical gradient on one element of w.
    h = 1e-5
    o, ci, u, v = 1, 0, 1, 2
    w_plus = w.copy()
    w_plus[o, ci, u, v] += h
    w_minus = w.copy()
    w_minus[o, ci, u, v] -= h
    numerical = (
        sols.conv2d_forward(x, w_plus, b).sum() - sols.conv2d_forward(x, w_minus, b).sum()
    ) / (2 * h)
    assert dw[o, ci, u, v] == pytest.approx(numerical, rel=1e-4, abs=1e-4)
    np.testing.assert_allclose(db, dy.sum(axis=(1, 2)))


def test_receptive_field_two_stacked_3x3(sols):
    layers = [sols.LayerSpec(kernel=3, stride=1), sols.LayerSpec(kernel=3, stride=1)]
    assert sols.receptive_field(layers) == 5


def test_receptive_field_with_downsampling(sols):
    layers = [
        sols.LayerSpec(kernel=3, stride=2),
        sols.LayerSpec(kernel=3, stride=1),
        sols.LayerSpec(kernel=3, stride=1),
    ]
    # After stride-2: RF = 1 + 2*1 = 3, jump=2
    # Next: RF = 3 + 2*2 = 7, jump=2
    # Next: RF = 7 + 2*2 = 11
    assert sols.receptive_field(layers) == 11


def test_resnet18_param_count_matches_torchvision(sols):
    pytest.importorskip("torch")
    from torchvision.models import resnet18

    expected = sum(p.numel() for p in resnet18(weights=None, num_classes=1000).parameters())
    assert sols.resnet18_param_count(num_classes=1000) == expected


# -- torch-gated classifier checks --------------------------------------------


@pytest.fixture(scope="module")
def classifier_module():
    pytest.importorskip("torch")
    return _load(CLASSIFIER_PATH, "w7_classifier")


def test_resnet18_for_cifar_forward(classifier_module):
    import torch

    model = classifier_module.resnet18_for_cifar(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_gradcam_returns_spatial_map(classifier_module):
    import torch

    model = classifier_module.resnet18_for_cifar(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    with classifier_module.GradCAM(model, target_layer=model.layer4) as cam:
        heatmap = cam(x, target_class=0)
    assert heatmap.ndim == 2
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0 + 1e-6


def test_fgsm_bounds_perturbation(classifier_module):
    import torch

    model = classifier_module.resnet18_for_cifar(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    eps = 0.01
    x_adv = classifier_module.fgsm(model, x, y, eps)
    assert x_adv.shape == x.shape
    # FGSM moves each pixel by exactly eps in some direction.
    max_delta = float((x_adv - x).abs().max())
    assert abs(max_delta - eps) < 1e-6
