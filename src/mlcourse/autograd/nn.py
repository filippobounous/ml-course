"""Minimal neural-network wrappers on top of the scalar autograd engine.

The API is deliberately close to micrograd / early PyTorch:

    mlp = MLP(n_in=2, n_outs=[8, 8, 1])
    out = mlp(x)              # scalar Value
    loss.backward()           # populates .grad on every leaf parameter
    for p in mlp.parameters():
        p.data -= lr * p.grad
"""

from __future__ import annotations

import random
from itertools import pairwise
from typing import Literal

from .engine import Value

Activation = Literal["tanh", "relu", "none"]


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(
        self,
        n_in: int,
        *,
        activation: Activation = "tanh",
        init: str = "glorot",
        rng: random.Random | None = None,
    ) -> None:
        rng = rng or random.Random()
        if init == "glorot":
            # Symmetric activations → Glorot/Xavier uniform in (−a, a).
            a = (6.0 / (n_in + 1)) ** 0.5
            self.w = [Value(rng.uniform(-a, a)) for _ in range(n_in)]
        elif init == "he":
            # He for ReLU: N(0, 2/n_in).
            sigma = (2.0 / n_in) ** 0.5
            self.w = [Value(rng.gauss(0.0, sigma)) for _ in range(n_in)]
        else:
            raise ValueError(f"Unknown init={init!r}")
        self.b = Value(0.0)
        self.activation: Activation = activation

    def __call__(self, x: list[Value]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(f"input length {len(x)} != n_in={len(self.w)}")
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        return act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer(Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        activation: Activation = "tanh",
        init: str = "glorot",
        rng: random.Random | None = None,
    ) -> None:
        rng = rng or random.Random()
        self.neurons = [
            Neuron(n_in, activation=activation, init=init, rng=rng) for _ in range(n_out)
        ]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        out: list[Value] = []
        for n in self.neurons:
            out.extend(n.parameters())
        return out


class MLP(Module):
    """An MLP with tanh hidden activations and a linear output head."""

    def __init__(
        self,
        n_in: int,
        n_outs: list[int],
        *,
        hidden_activation: Activation = "tanh",
        init: str = "glorot",
        seed: int = 0,
    ) -> None:
        if not n_outs:
            raise ValueError("n_outs must be non-empty.")
        rng = random.Random(seed)
        sizes = [n_in, *n_outs]
        self.layers: list[Layer] = []
        for i, (fan_in, fan_out) in enumerate(pairwise(sizes)):
            is_last = i == len(n_outs) - 1
            self.layers.append(
                Layer(
                    fan_in,
                    fan_out,
                    activation="none" if is_last else hidden_activation,
                    init=init,
                    rng=rng,
                )
            )

    def __call__(self, x: list[Value] | list[float]) -> list[Value] | Value:
        xs: list[Value] = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            xs = layer(xs)
        return xs[0] if len(xs) == 1 else xs

    def parameters(self) -> list[Value]:
        out: list[Value] = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out
