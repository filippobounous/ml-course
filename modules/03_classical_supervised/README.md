# Week 3 — Classical supervised learning

## Learning objectives

1. Derive **logistic regression** as MLE under a Bernoulli likelihood; understand softmax/multiclass.
2. Set up **SVMs** in primal and dual form; understand kernels as inner products in RKHS.
3. Fit **trees, random forests, gradient-boosted trees** with `xgboost` and `lightgbm`.
4. Reason about **calibration**, **ROC/PR curves**, and **class imbalance** in tabular problems.

## Topics

- Logistic regression, cross-entropy, Newton's method (IRLS).
- SVM: max-margin classifier, hinge loss, KKT conditions, dual formulation, kernel trick.
- Decision trees: CART, information gain vs Gini.
- Ensembles: bagging, boosting, gradient boosting (functional gradient view).
- **XGBoost** (2nd-order Taylor + regularisation) and **LightGBM** (histogram / GOSS).

## Deliverables

- Portfolio artifact: `portfolio/03_tabular_benchmark/` — UCI benchmark (Adult or Covertype) with logistic / RF / XGBoost / LightGBM and a calibration + ROC report.

## Reading plan

See `readings.md`.
