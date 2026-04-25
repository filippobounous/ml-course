"""Minimal optimisers that consume the scalar Value parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .engine import Value


class SGD:
    """Vanilla SGD with optional momentum and weight decay."""

    def __init__(
        self,
        params: Iterable[Value],
        *,
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [0.0 for _ in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self._velocity[i] = self.momentum * self._velocity[i] + g
            p.data -= self.lr * self._velocity[i]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


class Adam:
    """Adam with standard bias correction."""

    def __init__(
        self,
        params: Iterable[Value],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [0.0 for _ in self.params]
        self._v = [0.0 for _ in self.params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            m_hat = self._m[i] / (1 - self.beta1**self._t)
            v_hat = self._v[i] / (1 - self.beta2**self._t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0
