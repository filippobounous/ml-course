"""Week 07 — problem-set starter. **This is the file you edit.**

`pytest tests/week_07` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_07
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
    raise NotImplementedError("conv2d_forward")


def conv2d_backward(
    dy: ArrayF, x: ArrayF, w: ArrayF, *, stride: int = 1
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Gradients for a stride-`stride` 2-D cross-correlation.

    dy:  (C_out, H_out, W_out) — upstream gradient.
    Returns (dx, dw, db).
    """
    raise NotImplementedError("conv2d_backward")


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
    raise NotImplementedError("receptive_field")


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
    raise NotImplementedError("resnet18_param_count")
