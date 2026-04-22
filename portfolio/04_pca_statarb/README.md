# 04 — PCA statistical arbitrage

> Populated in Week 4. See `modules/04_classical_unsupervised/`.

## Problem
Build a PCA-based statistical-arbitrage strategy on US equity industry
portfolios, following Avellaneda & Lee (2008), and evaluate it honestly
with walk-forward splits and transaction costs.

## Method
- Rolling PCA on industry returns.
- Residual z-score strategy on the idiosyncratic component.
- Walk-forward splits; transaction-cost model.

## Results
*In-sample / OOS Sharpe, turnover, drawdown, and a short markdown write-up.*

## Reproduce
```bash
make -C portfolio/04_pca_statarb reproduce
```
