"""Gradient correctness tests for the scalar autograd engine.

Tests fall in three tiers:
  1. Analytical: for simple scalar expressions with known closed-form gradients.
  2. Numerical: central-difference gradient check against the autograd result.
  3. Torch-equivalence: compare to `torch.autograd` where torch is installed.
"""

from __future__ import annotations

import math

import pytest

from mlcourse.autograd import MLP, SGD, Adam, Value

# -- Tier 1: analytical ---------------------------------------------------------


def test_addition_gradient():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    out.backward()
    assert a.grad == 1.0 and b.grad == 1.0


def test_multiplication_gradient():
    a, b = Value(3.0), Value(-4.0)
    out = a * b
    out.backward()
    assert a.grad == -4.0 and b.grad == 3.0


def test_power_gradient():
    a = Value(2.0)
    out = a**3
    out.backward()
    # d/dx x^3 = 3x^2 = 12 at x = 2.
    assert a.grad == pytest.approx(12.0)


def test_chain_rule_in_dag():
    # f(x) = x*x*x where x appears three times — DAG, not tree.
    x = Value(2.0)
    out = x * x * x
    out.backward()
    assert out.data == pytest.approx(8.0)
    assert x.grad == pytest.approx(12.0)


def test_tanh_and_relu_at_zero():
    z = Value(0.0)
    t = z.tanh()
    t.backward()
    assert t.data == 0.0 and z.grad == 1.0

    z2 = Value(0.0)
    r = z2.relu()
    r.backward()
    # ReLU at 0 has subgradient 0 (our convention).
    assert r.data == 0.0 and z2.grad == 0.0


def test_sigmoid_derivative_identity():
    # σ'(x) = σ(x)(1 − σ(x)).
    for x in (-2.0, -0.5, 0.0, 1.0, 3.0):
        v = Value(x)
        out = v.sigmoid()
        out.backward()
        s = 1.0 / (1.0 + math.exp(-x))
        assert out.data == pytest.approx(s, abs=1e-12)
        assert v.grad == pytest.approx(s * (1 - s), abs=1e-12)


# -- Tier 2: numerical gradient check ------------------------------------------


def _numerical_grad(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("x_val", [0.25, 1.0, -1.5])
def test_numerical_grad_on_complex_expression(x_val):
    # f(x) = tanh(x^2 + 1) * exp(-x) + log(1 + x^2)
    def build(xv):
        x = Value(xv)
        return (x**2 + 1).tanh() * (-x).exp() + (1 + x**2).log()

    v = build(x_val)
    v.backward()

    # Walk the graph to find the original x; easier: re-derive via definition.
    # Here we just hook into the known leaf.
    def fn(xv):
        return math.tanh(xv * xv + 1) * math.exp(-xv) + math.log(1 + xv * xv)

    expected = _numerical_grad(fn, x_val)
    # The `x` leaf of `v` is the deepest Value; easiest path is to recompute.
    x = Value(x_val)
    ((x**2 + 1).tanh() * (-x).exp() + (1 + x**2).log()).backward()
    assert x.grad == pytest.approx(expected, abs=1e-3)


# -- Tier 3: torch equivalence -------------------------------------------------


def test_matches_torch_on_mlp_forward_and_backward():
    torch = pytest.importorskip("torch")

    # Our MLP with fixed weights.
    mlp = MLP(n_in=3, n_outs=[4, 1], seed=42)
    # Build equivalent torch model manually from our Value-typed weights.
    tw1 = torch.tensor([[w.data for w in n.w] for n in mlp.layers[0].neurons], requires_grad=True)
    tb1 = torch.tensor([n.b.data for n in mlp.layers[0].neurons], requires_grad=True)
    tw2 = torch.tensor([[w.data for w in n.w] for n in mlp.layers[1].neurons], requires_grad=True)
    tb2 = torch.tensor([n.b.data for n in mlp.layers[1].neurons], requires_grad=True)

    x_py = [0.5, -0.8, 1.2]
    x_t = torch.tensor(x_py, requires_grad=False)

    # Ours.
    out = mlp(x_py)
    assert isinstance(out, Value)
    loss = (out - 1.0) ** 2
    loss.backward()

    # Torch.
    h = torch.tanh(tw1 @ x_t + tb1)
    logit_t = (tw2 @ h + tb2).squeeze()
    loss_t = (logit_t - 1.0) ** 2
    loss_t.backward()

    assert out.data == pytest.approx(float(logit_t), abs=1e-6)
    # Compare gradients on the first-layer first neuron.
    for our_w, t_w in zip(mlp.layers[0].neurons[0].w, tw1.grad[0], strict=True):
        assert our_w.grad == pytest.approx(float(t_w), abs=1e-6)


# -- Training sanity checks -----------------------------------------------------


def test_sgd_reduces_loss_on_xor():
    # Tiny XOR problem — MLP with two hidden layers should solve it.
    data = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
    mlp = MLP(n_in=2, n_outs=[4, 4, 1], hidden_activation="tanh", seed=0)
    opt = SGD(mlp.parameters(), lr=0.1, momentum=0.9)
    first_loss = None
    for _ in range(400):
        total = Value(0.0)
        for (x0, x1), y in data:
            out = mlp([x0, x1])
            assert isinstance(out, Value)
            total = total + (out - y) ** 2
        if first_loss is None:
            first_loss = float(total.data)
        opt.zero_grad()
        total.backward()
        opt.step()
    assert float(total.data) < 0.1 * first_loss


def test_adam_reduces_loss_on_xor():
    data = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
    mlp = MLP(n_in=2, n_outs=[4, 4, 1], hidden_activation="tanh", seed=1)
    opt = Adam(mlp.parameters(), lr=0.05)
    first_loss = None
    for _ in range(300):
        total = Value(0.0)
        for (x0, x1), y in data:
            out = mlp([x0, x1])
            assert isinstance(out, Value)
            total = total + (out - y) ** 2
        if first_loss is None:
            first_loss = float(total.data)
        opt.zero_grad()
        total.backward()
        opt.step()
    assert float(total.data) < 0.1 * first_loss
