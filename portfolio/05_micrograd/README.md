# 05 — Micrograd-style autograd engine

> Populated in Week 5. See `modules/05_nn_from_scratch/`.

## Problem
A pedagogical scalar autograd engine in ≤300 lines that makes
backpropagation unambiguous — unit-tested against torch-autograd.

## Method
- `Value` class with `+ - * / ** exp tanh relu` and `.backward()`.
- `Neuron / Layer / MLP` abstractions.
- Unit tests vs numerical gradients and torch gradients.

## Results
*Training curves on two-moons; gradient-equivalence unit-test log.*

## Reproduce
```bash
make -C portfolio/05_micrograd reproduce
```

## Why this matters
Classic recruiter signal that you understand reverse-mode autodiff from
first principles.
