# Week 2 — Statistical learning & ERM

## Learning objectives

1. State the **empirical risk minimisation** framework and its limits (bias–variance, capacity, PAC).
2. Derive **MLE, MAP, and posterior** estimators for linear / logistic / Gaussian models.
3. Implement **linear regression** end-to-end (closed form + SGD, ridge, lasso, cross-validation) in **NumPy only**.
4. Read an **MDP** definition fluently (sets up Week 11).

## Topics

- Decision theory: 0-1 vs squared vs log loss, calibration.
- ERM: function classes, Rademacher complexity, VC dimension (intuition, not proofs).
- Linear regression: normal equations, ridge / lasso, regularisation paths.
- Model selection: K-fold CV, information criteria (AIC/BIC), nested CV pitfalls.
- MDP primer: states, actions, rewards, Bellman operators, policies.

## Deliverables

- Portfolio artifact: `portfolio/02_numpy_linreg/` — NumPy linear-models mini-library (closed-form + SGD; ridge; lasso via coordinate descent; K-fold CV; unit tests; README with figures).
- Problem set in `problems/`.

## Reading plan

See `readings.md`.
