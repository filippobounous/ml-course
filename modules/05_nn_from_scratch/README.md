# Week 5 — Neural networks from scratch

## Learning objectives

1. Derive **backpropagation** as reverse-mode automatic differentiation.
2. Implement a working **micrograd-style autograd engine** in ≤300 lines.
3. Justify **initialisation** (Glorot, He), **normalisation** (BatchNorm, LayerNorm), and **optimiser** choices from first principles.
4. Train a small MLP end-to-end on a toy dataset using your own engine.

## Topics

- Feed-forward networks as compositions of affine + nonlinearities.
- Computational graphs, reverse-mode autodiff.
- Loss landscapes, vanishing / exploding gradients.
- Initialisation: Glorot/Xavier, Kaiming/He; variance preservation derivations.
- Normalisation: BatchNorm (training vs eval), LayerNorm, RMSNorm.
- Optimisers: SGD + momentum + Nesterov, Adam, AdamW; learning-rate schedules.

## Deliverables

- Portfolio artifact: `portfolio/05_micrograd/` — autograd engine in ≤300 lines, unit tests, blog-post README.

## Reading plan

See `readings.md`.
