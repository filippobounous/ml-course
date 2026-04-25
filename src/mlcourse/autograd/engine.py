"""Scalar reverse-mode autograd, micrograd-style.

Intentionally readable and under ~200 lines. Builds a DAG of `Value` nodes
with per-operation closures that know how to push gradient to their parents.
Topological sort during `backward()` guarantees each node contributes to its
parents only after all of its own upstream contributions have arrived.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


class Value:
    """A scalar in the computation graph."""

    __slots__ = ("_backward", "_op", "_prev", "data", "grad")

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Value, ...] = _children
        self._op: str = _op

    # -- Construction / coercion ------------------------------------------------
    @staticmethod
    def _as_value(x: Value | float | int) -> Value:
        return x if isinstance(x, Value) else Value(x)

    # -- Arithmetic ------------------------------------------------------------
    def __add__(self, other: Value | float | int) -> Value:
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | float | int) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Value | float | int) -> Value:
        return self + (-self._as_value(other))

    def __rsub__(self, other: Value | float | int) -> Value:
        return self._as_value(other) - self

    def __mul__(self, other: Value | float | int) -> Value:
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float | int) -> Value:
        return self * other

    def __truediv__(self, other: Value | float | int) -> Value:
        return self * self._as_value(other) ** -1

    def __rtruediv__(self, other: Value | float | int) -> Value:
        return self._as_value(other) * self**-1

    def __pow__(self, power: float | int) -> Value:
        if not isinstance(power, (int, float)):
            raise TypeError("power must be a number; no Value**Value for now.")
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward() -> None:
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # -- Unary nonlinearities --------------------------------------------------
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        if self.data <= 0.0:
            raise ValueError("log requires positive input.")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        # 1/(1+exp(-x)); reuse the exp primitive to keep backward trivial.
        return 1.0 / (1.0 + (-self).exp())

    # -- Backprop --------------------------------------------------------------
    def backward(self) -> None:
        """Populate `.grad` for every ancestor via a topological backward pass."""
        topo: list[Value] = []
        seen: set[int] = set()

        def visit(v: Value) -> None:
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                visit(p)
            topo.append(v)

        visit(self)
        # Zero out any stale gradients from a previous backward call.
        for v in topo:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # -- Utilities -------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def zero_grad(params: Iterable[Value]) -> None:
    for p in params:
        p.grad = 0.0
