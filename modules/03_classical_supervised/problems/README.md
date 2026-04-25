# Problem set — Week 3

## Theory

1. **Logistic convexity.** Show the logistic negative log-likelihood is convex in $\beta$. Derive the Newton (IRLS) update.
2. **SVM dual.** Derive the dual of the soft-margin SVM from the primal via Lagrangian. Give the KKT conditions explicitly and interpret support vectors.
3. **Information gain.** For a binary split with proportions $(p_1, p_2)$ on each side and class frequencies, write the information gain and the Gini gain. Show they agree up to a monotone transform on small examples.

## Implementation

4. Implement **logistic regression** via Newton's method in NumPy (no scikit-learn). Compare to `sklearn.linear_model.LogisticRegression` on the breast-cancer dataset.
5. Benchmark **XGBoost vs LightGBM** on UCI Adult — report test AUC, calibration (Brier), training time, and memory. Use a held-out split plus stratified K-fold.

## Applied

6. Portfolio artifact: `portfolio/03_tabular_benchmark/` — add a ROC / PR / calibration report for all four methods (logistic, RF, XGBoost, LightGBM).

## Grading

Tests in `tests/week_03/` check the IRLS implementation matches sklearn to tolerance and that the benchmark JSON contains all required metrics.
