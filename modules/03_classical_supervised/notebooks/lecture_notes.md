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

Work the problem set in `../problems/README.md`. Implement IRLS logistic regression (reference in `../problems/solutions.py`). Build the portfolio artifact in `../../../portfolio/03_tabular_benchmark/`: logistic / random forest / XGBoost / LightGBM on UCI Adult with full ROC/PR and calibration reporting.

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three numerical exercises (one IRLS Newton step on a 3-point problem, SVM dual closed-form on three points, Gini vs entropy vs XGBoost second-order gain on a tiny split).

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1 Logistic + IRLS | 4 | Prove NLL convexity; derive IRLS by hand; implement Newton in NumPy. |
| §2 SVMs | 4 | Primal → dual derivation; KKT slackness; kernel trick on a Gram-matrix toy. |
| §3 Trees + ensembles | 4 | Gini / entropy splits on toy data; XGBoost second-order gain formula. |
| §4 Calibration | 2 | Brier decomposition; isotonic vs Platt scaling. |
| §5 Class imbalance | 1 | AUC-PR vs ROC on an imbalanced toy set. |
| Problem set + benchmark | 4 | IRLS implementation + XGBoost vs LightGBM benchmark on UCI Adult. |
| Office hours / review | 1 | Check proofs against `problems/solutions_theory.md`. |

## Self-assessment rubric

Before moving to Week 4, you should be able to answer "yes" to all of:

1. Can I derive the IRLS Newton update for logistic regression from $\nabla^2 L = X^\top W X$, and verify NLL convexity?
2. Can I derive the soft-margin SVM dual from the primal Lagrangian and interpret the KKT complementary slackness in terms of margin / on-margin / inside-margin support vectors?
3. Can I compute information gain and Gini gain for a binary split and explain why they typically pick the same split?
4. Can I derive XGBoost's second-order split-gain formula from the functional-gradient view of boosting?
5. Can I produce a calibration curve, decompose the Brier score into calibration + resolution + uncertainty, and explain when to use isotonic regression vs Platt scaling?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **SVM dual ↔ Lagrangian mechanics with inequality constraints.** The KKT conditions are exactly the constrained-extremum conditions: stationarity (Euler–Lagrange), primal feasibility (the constraint), dual feasibility ($\alpha_i \ge 0$ — Lagrange multipliers non-negative for inequality constraints), complementary slackness ($\alpha_i \cdot \text{constraint}_i = 0$ — multipliers only "fire" on active constraints). The slackness is the discrete-time analogue of a contact constraint going active/inactive in rigid-body dynamics.
- **Kernel trick ↔ working in a different basis on a Hilbert space.** A positive-definite kernel $K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$ implicitly maps to a (possibly infinite-dimensional) feature space $\mathcal{H}$. RBF kernels lift to $\mathcal{H} = $ a Sobolev-like RKHS. Same construction physicists use for Green's-function methods: never write the basis explicitly; just compute pairwise inner products.
- **Calibration ↔ thermal equilibrium of a binary observable.** A calibrated classifier emits probabilities that match empirical frequencies in the long run — exactly the condition that the binary observable be in thermal equilibrium with the data generator. Miscalibration is a free-energy gap.
- **Gradient boosting ↔ functional gradient descent on $L^2$.** Boosting iteratively descends on the empirical-risk *functional* in $L^2(\text{data manifold})$: at each step you add the function (a tree) that best approximates the functional gradient. XGBoost's second-order Taylor expansion is Newton's method in function space — the same idea as Hartree–Fock SCF in quantum chemistry (functional gradient + curvature on a one-particle space).

Keep these bridges live; W5 (autograd / backprop ≡ adjoint method) and W12 (PINN via functional gradient on the residual) reuse the "gradient descent on a function space" pattern explicitly.
