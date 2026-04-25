# 02 — NumPy linear-models mini-library

A clean, tested, pure-NumPy implementation of:

- **OLS** (closed form via `lstsq`, and mini-batch SGD with optional L2)
- **Ridge regression** (closed form)
- **Lasso** (coordinate descent with soft-thresholding)
- **K-fold cross-validation** with leak-free re-fitting per fold

All solvers follow a common `LinearModel` dataclass with `.predict(X)`, a
training history, and an explicit intercept. Zero external deps beyond NumPy.

## Layout

```
portfolio/02_numpy_linreg/
├── linreg.py          ← the library (single file, ~180 lines)
├── demo.py            ← reproducibility entry point (figures + report)
└── README.md
```

## Quickstart

```python
from portfolio.linreg import fit_ridge, cross_val_mse

model = fit_ridge(X, y, alpha=1.0)
preds = model.predict(X_test)
mean_mse, per_fold = cross_val_mse(lambda X, y: fit_ridge(X, y, alpha=1.0), X, y, k=5)
```

## Reproduce

```bash
python portfolio/02_numpy_linreg/demo.py
```

This generates a regularisation-path figure (ridge + lasso) on a simulated
problem with $p = 40$ features of which 5 are informative, and prints a small
benchmark against `scikit-learn` to check we are numerically close.

## Tests

Run `pytest tests/week_02 -q` — eight checks:

- closed-form OLS matches the normal equations
- ridge matches sklearn to $10^{-8}$
- SGD converges in the sense that the final loss is below the initial loss
- lasso recovers the support of a sparse simulated signal
- K-fold indices form a correct partition
- ridge CV MSE is convex in $\log \alpha$ (validated on a synthetic problem)

## What I learned

*To be filled after completing Week 2.*
