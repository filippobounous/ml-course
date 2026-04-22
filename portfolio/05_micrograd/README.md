# 05 — Micrograd-style autograd engine

A pedagogical scalar reverse-mode autograd in ~200 lines — built from first
principles, unit-tested against `torch.autograd`, and used to train a small
MLP end-to-end on two-moons.

## What's where

- **`src/mlcourse/autograd/engine.py`** — the `Value` class: forward / backward
  for `+ − × / ** exp log tanh relu sigmoid`, topological-sort backward pass.
- **`src/mlcourse/autograd/nn.py`** — `Neuron`, `Layer`, `MLP` wrappers with
  Glorot / He initialisation.
- **`src/mlcourse/autograd/optim.py`** — `SGD` (with momentum + weight decay)
  and `Adam` (with bias correction) over scalar Values.
- **`portfolio/05_micrograd/demo.py`** — trains the MLP on two-moons and
  plots the training curve and decision boundary.
- **`tests/week_05/`** — gradient-equivalence tests vs `torch.autograd` on a
  half-dozen small graphs; optimiser sanity checks.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python portfolio/05_micrograd/demo.py
```

Runs in under 30 seconds on a single CPU core.

## Why it's shippable

- Hand-derived chain rule for every primitive (documented in
  `modules/05_nn_from_scratch/notebooks/lecture_notes.md`).
- End-to-end training on a non-trivial 2-D dataset with decision-boundary
  figure.
- Passes gradient-equivalence tests vs `torch.autograd` to 1e-6.
- Clean API, typed, zero external deps beyond the Python standard library +
  NumPy / PyTorch (the latter only in the tests).

## What I learned

*To be filled after completing Week 5. Suggested bullets:*

- Why reverse-mode is the only sensible choice for scalar-loss gradients.
- Why `.grad` is *accumulated* (DAGs have shared subexpressions).
- How initialisation choices interact with activation functions.
