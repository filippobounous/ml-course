# Week 3 — Classical supervised learning (lecture notes)

*Reading pair: ESL Ch.4, 9, 10, 15 · Murphy PML-1 Ch.17 · Chen & Guestrin 2016.*

---

## 1. Logistic regression

For binary classification $y \in \{0, 1\}$, model

$$p(y = 1 \mid x) = \sigma(\beta^\top x), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

The negative log-likelihood is

$$L(\beta) = -\sum_{i} [y_i \log \sigma(\beta^\top x_i) + (1 - y_i) \log(1 - \sigma(\beta^\top x_i))].$$

This is **convex in $\beta$** — check: $\nabla^2 L = X^\top W X$ with $W_{ii} = \sigma(\cdot)(1 - \sigma(\cdot)) \succeq 0$.

**IRLS / Newton.** The Newton update

$$\beta^{(t+1)} = \beta^{(t)} - (\nabla^2 L)^{-1} \nabla L = (X^\top W X)^{-1} X^\top W z$$

with $z = X \beta^{(t)} + W^{-1} (y - p)$ is iteratively reweighted least squares. Converges quadratically once near the optimum.

**Multinomial / softmax.** For $K$ classes,

$$p(y = k \mid x) = \frac{e^{\beta_k^\top x}}{\sum_{j} e^{\beta_j^\top x}},$$

and the loss is cross-entropy $-\sum_{i,k} \mathbf{1}[y_i = k] \log p(y = k \mid x_i)$. Softmax is shift-invariant so implementations subtract $\max_k \beta_k^\top x$ before exponentiating.

## 2. Support vector machines

### Max-margin primal

$$\min_{w, b, \xi} \tfrac12 \|w\|^2 + C \sum_i \xi_i \quad\text{s.t. } y_i (w^\top x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0.$$

### Dual

$$\max_\alpha \sum_i \alpha_i - \tfrac12 \sum_{ij} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle \quad\text{s.t. } 0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0.$$

Support vectors are those with $\alpha_i > 0$. Inner products $\langle x_i, x_j \rangle$ can be replaced by any positive-definite kernel $K(x_i, x_j)$ — the **kernel trick**. Common kernels: linear, polynomial, RBF $\exp(-\gamma \|x - x'\|^2)$.

### KKT and complementary slackness

At the optimum,

- $\alpha_i = 0 \Rightarrow y_i(w^\top x_i + b) \ge 1$ (easy point).
- $0 < \alpha_i < C \Rightarrow y_i(w^\top x_i + b) = 1$ (on the margin).
- $\alpha_i = C \Rightarrow$ violator or within the margin.

## 3. Decision trees and ensembles

### CART

Greedy recursive splits minimising an impurity measure: **Gini** $\sum_k p_k(1 - p_k)$ or **entropy** $-\sum_k p_k \log p_k$. Gini and entropy are monotone transforms of each other on binary problems.

### Random forests

$B$ independent trees on bootstrap samples + random feature subsets per split; average predictions. Variance drops roughly as $\rho \sigma^2 + (1 - \rho) \sigma^2 / B$ where $\rho$ is the correlation between tree predictions.

### Gradient boosting (the **functional gradient** view)

Fit an additive model $F(x) = \sum_m \nu f_m(x)$ by iteratively adding a tree that approximates the negative gradient of the loss in function space. For squared loss this is residual fitting; for log-loss it is fitting the probabilistic gradient.

**XGBoost** adds three ingredients: (1) a regularised objective with leaf-score L2 penalty and a tree-complexity term; (2) a second-order Taylor expansion so each split's gain uses both $g$ and $h$; (3) careful missing-value handling and column subsampling. **LightGBM** replaces exact splits with a **histogram** of binned features and adds GOSS (retain large-gradient samples, subsample small-gradient ones).

## 4. Calibration

A classifier is **calibrated** when among inputs with $\hat p = 0.7$ roughly 70% have $y = 1$. Neural nets and boosted trees often are not — fix with:
- **Platt scaling** (sigmoid fit on a held-out set).
- **Isotonic regression** (monotone non-parametric fit, more flexible but needs more data).

Key metric: **Brier score** $\frac{1}{N} \sum_i (\hat p_i - y_i)^2$; decomposes into calibration + resolution + uncertainty.

## 5. Class imbalance

- Don't resample blindly; it changes your calibration.
- Prefer **class-weighted losses** and careful threshold selection.
- Evaluate with **AUC-PR** (precision–recall), not accuracy. Plot both ROC and PR.

## What to do with these notes

Work the problem set in `../problems/README.md`. Implement IRLS logistic regression (reference in `../problems/solutions.py`). Build the portfolio artifact in `../../../portfolio/03_tabular_benchmark/`: logistic / random forest / XGBoost / LightGBM on Covertype with full ROC/PR and calibration reporting.
