# 03 — Tabular benchmark: logistic / RF / XGBoost / LightGBM

> Populated in Week 3. See `modules/03_classical_supervised/`.

## Problem
A head-to-head benchmark of classical tabular-ML methods on UCI Adult (or
Covertype), reporting not just accuracy/AUC but also calibration and
training time.

## Method
- Logistic regression (IRLS, implemented from scratch).
- Random forest, XGBoost, LightGBM (stock implementations).
- Stratified K-fold + held-out test split.
- Calibration: reliability diagrams + Brier score.

## Results
*ROC / PR / calibration plots + timing table land here.*

## Reproduce
```bash
make -C portfolio/03_tabular_benchmark reproduce
```
