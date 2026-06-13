# Week 2 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — Ridge bias / variance on a 1-D problem

Toy data-generating process: $y = \beta^\star x + \varepsilon$ with
$\beta^\star = 1$, $x \in \{-2, -1, 0, 1, 2\}$, $\varepsilon \sim
\mathcal{N}(0, \sigma^2)$, $\sigma = 0.3$.

Ridge estimator with regulariser $\lambda$:

$$
\hat\beta_\lambda = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda} = \frac{10 \beta^\star + \sum_i x_i \varepsilon_i}{10 + \lambda}.
$$

(The $\sum x_i^2 = 4 + 1 + 0 + 1 + 4 = 10$.)

### Bias

$$
\mathbb{E}[\hat\beta_\lambda] = \frac{10 \beta^\star}{10 + \lambda} = \frac{10}{10 + \lambda}, \qquad \text{Bias} = \mathbb{E}[\hat\beta_\lambda] - \beta^\star = -\frac{\lambda}{10 + \lambda}.
$$

Bias$^2 = \lambda^2 / (10 + \lambda)^2$ — monotonically increasing in $\lambda$.

### Variance

$$
\operatorname{Var}(\hat\beta_\lambda) = \frac{\sigma^2 \sum_i x_i^2}{(10 + \lambda)^2} = \frac{10 \sigma^2}{(10 + \lambda)^2} = \frac{0.9}{(10 + \lambda)^2}.
$$

Monotonically decreasing in $\lambda$.

### Optimal $\lambda$

Total MSE = Bias$^2$ + Variance. Differentiate, set to zero:

$$
\frac{d}{d\lambda}\!\left[ \frac{\lambda^2 + 10 \sigma^2}{(10 + \lambda)^2} \right] = \frac{2 \cdot 10\,(\lambda - \sigma^2)}{(10 + \lambda)^3} = 0 \quad \Longrightarrow \quad \lambda^\star = \sigma^2 = 0.09.
$$

For this problem the optimal $\lambda^\star = \sigma^2 = 0.09$ — small but
nonzero, despite the "true" model being unregularised. Numerical sanity check:

| $\lambda$ | Bias² | Variance | MSE |
|---|---|---|---|
| 0    | 0.0000 | 0.0090 | 0.0090 |
| 0.09 | 0.0001 | 0.0088 | 0.0089 |
| 0.5  | 0.0023 | 0.0082 | 0.0104 |
| 5    | 0.1111 | 0.0040 | 0.1151 |

Bias$^2 = 0.0000$ at $\lambda = 0$ is correct (OLS is unbiased), but the MSE
is minimised at $\lambda^\star = 0.09$, where the small variance reduction
just outweighs the bias introduced. The margin over OLS is tiny here because
the noise level is small and the problem is well-conditioned; on a harder
problem (larger $\sigma$ or $X^\top X$ closer to singular) ridge wins clearly.

---

## Example 2 — Bayesian Bernoulli with Beta prior

Setup: $y_1, \ldots, y_N \sim \mathrm{Bernoulli}(\theta)$ with prior
$\theta \sim \mathrm{Beta}(\alpha, \beta)$. Conjugate update:

$$
\theta \mid y_{1:N} \sim \mathrm{Beta}(\alpha + s, \beta + N - s), \qquad s = \sum_i y_i.
$$

### MAP

$\mathrm{Beta}(a, b)$ has mode at $(a - 1)/(a + b - 2)$ for $a, b > 1$.
So

$$
\hat\theta_\text{MAP} = \frac{\alpha + s - 1}{\alpha + \beta + N - 2}.
$$

### MLE

$\hat\theta_\text{MLE} = s / N$ (drop the prior — equivalent to $\alpha = \beta = 1$, the uniform prior).

### Numerical: $\alpha = \beta = 2$ (mode at $\tfrac{1}{2}$), 3 heads in 5 flips

$$
\hat\theta_\text{MAP} = \tfrac{2 + 3 - 1}{2 + 2 + 5 - 2} = \tfrac{4}{7} \approx 0.571.
$$
$$
\hat\theta_\text{MLE} = 3/5 = 0.600.
$$

The MAP is pulled towards the prior mean $0.5$. **As $N \to \infty$ the
$- 1$ and $-2$ are negligible and MAP → MLE → $\theta^\star$.** Try $N
= 50$, $s = 30$: MAP $= 31/52 \approx 0.596$, MLE $= 0.600$ — almost
identical.

### Connection to ridge

The Beta prior plays exactly the role of a Gaussian prior in ridge:
strong prior beliefs (high $\alpha + \beta$) ↔ large $\lambda$ ↔
shrinkage toward the prior mean. The "data swamps the prior" rate is
$\mathcal{O}(1/N)$ in both cases.

---

## Example 3 — K-fold cross-validation picks $\lambda$

Generate 100 samples from $y = 1 + 0.5 x + \varepsilon$, $x \sim
\mathcal{N}(0, 1)$, $\sigma = 1$. Augment with 9 useless features
$x_2, \ldots, x_{10}$ drawn $\mathcal{N}(0, 1)$. So true model is
sparse; ridge will overfit if $\lambda$ too small, underfit if
too large.

### Procedure

For $\lambda \in \{0.01, 0.1, 1, 10, 100\}$:
- 5-fold CV, refit on each 80-sample training fold.
- Predict on the 20-sample validation fold, compute MSE.
- Average MSE across folds.

### Expected shape

Train MSE: monotonically increasing in $\lambda$ (more regularisation
= worse fit).
Test (CV) MSE: U-shaped — high at $\lambda = 0.01$ (overfit, high
variance), high at $\lambda = 100$ (underfit, high bias), minimum
near $\lambda \approx 1$.

### Numerical check (representative seed)

| $\lambda$ | Train MSE | CV MSE |
|---|---|---|
| 0.01 | 0.93 | 1.32 |
| 0.1  | 0.94 | 1.21 |
| 1    | 0.97 | **1.15** |
| 10   | 1.10 | 1.18 |
| 100  | 1.46 | 1.51 |

CV picks $\lambda = 1$. Train MSE alone would have picked $\lambda =
0.01$ — that's the **leakage of generalisation information** that CV
prevents. This is the artifact you ship in `portfolio/02_numpy_linreg/`.

### Pitfall: feature scaling inside the fold

If you `StandardScaler.fit_transform(X_train)` on the *full* training
set before splitting into folds, the validation fold sees scaling
parameters that depend on its own data — that's leakage. Fit the
scaler **inside each fold's training portion only**, then apply to
the validation portion. Same rule for mean imputation, PCA, anything
data-dependent.

---

## What to do with these examples

Re-do Example 1 with $\sigma = 1$ (instead of $0.3$) and watch ridge
become clearly preferable. Re-do Example 2 with $\alpha = \beta = 10$
(strong prior) and see how many more flips you need to overcome
prior bias. For Example 3, implement the CV loop yourself in NumPy —
your portfolio artifact this week is the full version.
