"""Week 7 — reference solutions.

Implements:
  * 2-D convolution in NumPy (forward + backward) — so the backprop derivation
    is concrete.
  * Receptive-field computation for stacked conv layers.
  * ResNet-18 parameter count (closed form to cross-check torchvision's model).

Torch is *not* required for this file — these NumPy/math utilities exercise the
derivations covered in the lecture notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Manual 2-D convolution


def conv2d_forward(x: ArrayF, w: ArrayF, b: ArrayF | None = None, *, stride: int = 1) -> ArrayF:
    """NumPy 2-D cross-correlation.

    x:  (C_in, H, W)
    w:  (C_out, C_in, k, k)
    b:  (C_out,) or None
    Returns y: (C_out, H_out, W_out) where H_out = (H - k) // stride + 1.
    """
    if x.ndim != 3 or w.ndim != 4:
        raise ValueError("x must be 3-D (C, H, W) and w 4-D (O, I, k, k).")
    C_in, H, W = x.shape
    C_out, C_in_w, k, k2 = w.shape
    if C_in != C_in_w or k != k2:
        raise ValueError("Channel / kernel shapes incompatible.")
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    y = np.zeros((C_out, H_out, W_out), dtype=np.float64)
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i * stride : i * stride + k, j * stride : j * stride + k]
                y[o, i, j] = float((patch * w[o]).sum())
        if b is not None:
            y[o] += b[o]
    return y


def conv2d_backward(
    dy: ArrayF, x: ArrayF, w: ArrayF, *, stride: int = 1
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Gradients for a stride-`stride` 2-D cross-correlation.

    dy:  (C_out, H_out, W_out) — upstream gradient.
    Returns (dx, dw, db).
    """
    C_out, _, k, _ = w.shape
    H_out, W_out = dy.shape[1], dy.shape[2]
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = dy.sum(axis=(1, 2))
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i * stride : i * stride + k, j * stride : j * stride + k]
                dw[o] += dy[o, i, j] * patch
                dx[:, i * stride : i * stride + k, j * stride : j * stride + k] += (
                    dy[o, i, j] * w[o]
                )
    return dx, dw, db


# -----------------------------------------------------------------------------
# Receptive field


@dataclass(frozen=True)
class LayerSpec:
    kernel: int
    stride: int
    dilation: int = 1


def receptive_field(layers: list[LayerSpec]) -> int:
    """Effective receptive field (in input pixels) after stacking `layers`.

    Formula: RF_0 = 1; RF_{l+1} = RF_l + (k_{l+1} − 1) · d_{l+1} · prod_{m ≤ l} s_m.
    """
    rf = 1
    jump = 1
    for layer in layers:
        rf += (layer.kernel - 1) * layer.dilation * jump
        jump *= layer.stride
    return rf


# -----------------------------------------------------------------------------
# ResNet-18 parameter count (closed form)


def resnet18_param_count(num_classes: int = 1000) -> int:
    """Closed-form parameter count of the torchvision ResNet-18.

    Structure:
      * stem: 7×7 conv (3 → 64, stride 2) + BN(64)  — no bias on conv
      * layer1: 2× BasicBlock(64 → 64)
      * layer2: 2× BasicBlock(64 → 128, first block downsamples with 1×1 conv)
      * layer3: 2× BasicBlock(128 → 256, first block downsamples)
      * layer4: 2× BasicBlock(256 → 512, first block downsamples)
      * fc: Linear(512 → num_classes)

    Each 3×3 conv has C_in · C_out · 9 params; each BN has 2·C params;
    each BasicBlock has two 3×3 convs + two BNs; the downsampling block has
    an extra 1×1 conv (C_in · C_out) and BN (2·C_out).
    """

    def conv_bn(c_in: int, c_out: int, k: int) -> int:
        return c_in * c_out * k * k + 2 * c_out

    def basic_block(c_in: int, c_out: int, *, downsample: bool) -> int:
        p = conv_bn(c_in, c_out, 3) + conv_bn(c_out, c_out, 3)
        if downsample:
            p += c_in * c_out * 1 * 1 + 2 * c_out  # 1×1 conv + BN
        return p

    total = conv_bn(3, 64, 7)  # stem
    total += basic_block(64, 64, downsample=False) * 2
    total += basic_block(64, 128, downsample=True) + basic_block(128, 128, downsample=False)
    total += basic_block(128, 256, downsample=True) + basic_block(256, 256, downsample=False)
    total += basic_block(256, 512, downsample=True) + basic_block(512, 512, downsample=False)
    total += 512 * num_classes + num_classes  # fc with bias
    return total


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 8, 8))
    w = rng.standard_normal((5, 3, 3, 3))
    b = rng.standard_normal(5)
    y = conv2d_forward(x, w, b)
    print("conv2d out shape:", y.shape)

    rf = receptive_field(
        [
            LayerSpec(kernel=3, stride=1),
            LayerSpec(kernel=3, stride=1),
            LayerSpec(kernel=3, stride=2),
            LayerSpec(kernel=3, stride=1),
        ]
    )
    print("receptive field after 4 convs:", rf)

    print("ResNet-18 param count:", resnet18_param_count())
