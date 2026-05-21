# Week 3 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — One IRLS step on a tiny 2-D logistic problem

Data: three points in 2-D, labels $y$:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 0 | 0 | 0 |

Start at $\beta = (0, 0)$. Then $p_i = \sigma(0) = 0.5$ for all $i$,
$W = \mathrm{diag}(0.25, 0.25, 0.25)$, residual $p - y = (-0.5, -0.5, 0.5)$.

### Newton step

$X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$, so

$X^\top W X = 0.25 \cdot X^\top X = \begin{pmatrix} 0.25 & 0 \\ 0 & 0.25 \end{pmatrix}$.

Gradient: $X^\top (p - y) = (-0.5, -0.5)^\top$.

Newton update:

$$
\beta^{(1)} = \beta^{(0)} - (X^\top W X)^{-1} X^\top (p - y) = (0,0) - \begin{pmatrix} 4 & 0 \\ 0 & 4 \end{pmatrix}\begin{pmatrix} -0.5 \\ -0.5 \end{pmatrix} = (2, 2)^\top.
$$

### Second iteration

Now $p = (\sigma(2), \sigma(2), \sigma(0)) = (0.881, 0.881, 0.5)$,
residual $(p - y) = (-0.119, -0.119, 0.5)$,
$W = \mathrm{diag}(0.105, 0.105, 0.25)$.

Plug in:

$$
X^\top W X = \begin{pmatrix} 0.105 & 0 \\ 0 & 0.105 \end{pmatrix}, \quad X^\top (p - y) = (-0.119, -0.119)^\top.
$$

$\beta^{(2)} = (2, 2) - (1/0.105) (-0.119, -0.119) = (2 + 1.13, 2 + 1.13) = (3.13, 3.13)$.

Continuing converges quadratically toward the (unconstrained) MLE.
For these three points the data is *separable* and the MLE drifts to
infinity — exactly the kind of failure mode that L2-regularised
logistic (ridge logistic) prevents.

---

## Example 2 — SVM dual on three points

Two positive ($y = +1$) and one negative ($y = -1$):

| $x$ | $y$ |
|---|---|
| $(2, 0)$ | $+1$ |
| $(0, 2)$ | $+1$ |
| $(0, 0)$ | $-1$ |

Hard-margin SVM (no slack), linear kernel. Compute the Gram matrix:

$K = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 4 & 0 \\ 0 & 0 & 0 \end{pmatrix}$.

The dual: $\max_\alpha \alpha_1 + \alpha_2 + \alpha_3 - \tfrac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j K_{ij}$
subject to $\sum_i \alpha_i y_i = 0$ and $\alpha_i \ge 0$. The
constraint is $\alpha_1 + \alpha_2 = \alpha_3$.

Plug in: objective becomes

$$
\alpha_1 + \alpha_2 + \alpha_3 - 2 \alpha_1^2 - 2 \alpha_2^2.
$$

Substitute $\alpha_3 = \alpha_1 + \alpha_2$:

$$
L(\alpha_1, \alpha_2) = 2(\alpha_1 + \alpha_2) - 2(\alpha_1^2 + \alpha_2^2).
$$

First-order conditions: $\partial L / \partial \alpha_1 = 2 - 4\alpha_1 = 0$,
so $\alpha_1 = 1/2$. Similarly $\alpha_2 = 1/2$, $\alpha_3 = 1$.

### Reconstruct $w$ and $b$

$w = \sum_i \alpha_i y_i x_i = 0.5 \cdot 1 \cdot (2,0) + 0.5 \cdot 1 \cdot (0,2) + 1 \cdot (-1) \cdot (0,0) = (1, 1)$.

$b$ from any support vector with $0 < \alpha_i < C$: $y_i(w^\top x_i + b) = 1$.
Take the negative point: $-1 \cdot (0 + b) = 1$, so $b = -1$.

Decision boundary: $x_1 + x_2 = 1$. The margin is $\|w\|^{-1} = 1/\sqrt 2$.
**All three points are support vectors** — they all sit exactly on
the margin (compute: $1 \cdot (2 + 0 - 1) = 1$ ✓ for $(2, 0)$).

---

## Example 3 — Gini vs information gain on a tiny split

Parent: 10 samples, 6 positives + 4 negatives, $\mathbf{p} = (0.6, 0.4)$.

Candidate split: left child $(4+, 1-)$ (5 samples), right child
$(2+, 3-)$ (5 samples).

### Entropy / information gain

$H(\text{parent}) = -0.6 \log_2 0.6 - 0.4 \log_2 0.4 \approx 0.971$ bits.
$H(L) = -0.8 \log_2 0.8 - 0.2 \log_2 0.2 \approx 0.722$.
$H(R) = -0.4 \log_2 0.4 - 0.6 \log_2 0.6 \approx 0.971$.

$\text{IG}_\text{entropy} = 0.971 - 0.5 \cdot 0.722 - 0.5 \cdot 0.971 = 0.124$ bits.

### Gini

$G(\text{parent}) = 1 - 0.6^2 - 0.4^2 = 0.48$.
$G(L) = 1 - 0.8^2 - 0.2^2 = 0.32$.
$G(R) = 1 - 0.4^2 - 0.6^2 = 0.48$.

$\text{IG}_\text{Gini} = 0.48 - 0.5 \cdot 0.32 - 0.5 \cdot 0.48 = 0.080$.

### XGBoost second-order gain

For squared-error gradient boosting, the gain from splitting a leaf
(with sum of gradients $G$, sum of Hessians $H$) into two children
$(G_L, H_L)$ and $(G_R, H_R)$ with regulariser $\lambda$ and split
cost $\gamma$ is

$$
\text{Gain} = \tfrac{1}{2}\!\left[\tfrac{G_L^2}{H_L + \lambda} + \tfrac{G_R^2}{H_R + \lambda} - \tfrac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.
$$

The pleasing detail: this is the same algebraic structure as
$\text{Var}(\bar X_L) + \text{Var}(\bar X_R) - \text{Var}(\bar X)$
weighted by inverse Hessian — i.e. **second-order Gini-like
information gain** in function space. That's why XGBoost is often
called "Gini-on-steroids" — it's the same decomposition but with the
proper second-order curvature instead of the first-order Bernoulli
variance.

### Sanity check

All three give the same *ordering* on most splits (you can verify by
plotting the three measures vs $p$ on a binary classification curve).
The "Gini vs entropy" choice almost never moves the test AUC by more
than the noise on a 5-fold CV — that's why XGBoost defaults are
robust.

---

## What to do with these examples

For Example 1, try regularised logistic (add $\lambda \beta$ to the
gradient and $\lambda I$ to the Hessian) and watch it converge to a
finite solution. For Example 2, swap the negative to $(1, 1)$ and
recompute — fewer support vectors, harder margin. For Example 3, plot
the three impurity measures as functions of $p \in [0, 1]$ and
overlay; you'll see they're all peaked at $p = 0.5$ but with
different curvatures.
