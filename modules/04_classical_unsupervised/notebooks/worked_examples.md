# Week 4 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — PCA on a 4-sample, 2-feature toy

Data (rows = samples, already centered):

$$
X = \begin{pmatrix} 1 & 2 \\ -1 & -2 \\ 1 & 1 \\ -1 & -1 \end{pmatrix}.
$$

### Step 1. Sample covariance

$$
S = \tfrac{1}{N} X^\top X = \tfrac{1}{4}\begin{pmatrix} 4 & 6 \\ 6 & 10 \end{pmatrix} = \begin{pmatrix} 1 & 1.5 \\ 1.5 & 2.5 \end{pmatrix}.
$$

### Step 2. Eigendecomposition

Characteristic polynomial: $(1 - \lambda)(2.5 - \lambda) - 2.25 = \lambda^2 - 3.5\lambda + 0.25 = 0$.

$\lambda = (3.5 \pm \sqrt{12.25 - 1}) / 2 = (3.5 \pm \sqrt{11.25})/2 \approx (3.5 \pm 3.354)/2$.

$\lambda_1 \approx 3.427$ (≈ 93% of variance), $\lambda_2 \approx 0.073$.

### Step 3. Top eigenvector

$(S - \lambda_1 I) v_1 = 0$:

$$\begin{pmatrix} 1 - 3.427 & 1.5 \\ 1.5 & 2.5 - 3.427 \end{pmatrix} v_1 = 0.$$

First row: $-2.427 v_{1,1} + 1.5 v_{1,2} = 0 \Rightarrow v_{1,2} = 1.618 v_{1,1}$. Normalise: $v_1 \approx (0.526, 0.851)^\top$ — the golden-ratio direction, because this data was contrived to make $v_1$ point along $(1, \phi)$.

### Step 4. Verify by SVD

$X = U \Sigma V^\top$ with the same $V$. Singular values $\sigma_i = \sqrt{N \lambda_i}$, so $\sigma_1 \approx \sqrt{4 \cdot 3.427} \approx 3.70$, $\sigma_2 \approx 0.54$.

The first PC explains $\lambda_1 / (\lambda_1 + \lambda_2) \approx 97.9\%$ of the variance — almost a 1-D problem. In real data this is what "rank concentrates" looks like: a few large $\lambda$, a long tail.

---

## Example 2 — k-means on 6 points, k = 2

Points in 1-D (for clarity):
$x = (0, 1, 2, 10, 11, 12)$.

### Iteration 0 (random init)

Centres $\mu_1 = 0.5$, $\mu_2 = 1.5$ (both at the wrong cluster). Assignments by nearest: $(0, 1) \to 1$, $(2, 10, 11, 12) \to 2$.

Distortion: $(0 - 0.5)^2 + (1 - 0.5)^2 + (2 - 1.5)^2 + \dots = 0.25 + 0.25 + 0.25 + (10-1.5)^2 + (11-1.5)^2 + (12-1.5)^2 = 0.75 + 72.25 + 90.25 + 110.25 = 273.5$.

### Iteration 1 (update step)

$\mu_1 = \text{mean}(0, 1) = 0.5$. $\mu_2 = \text{mean}(2, 10, 11, 12) = 8.75$.

Now reassign: $0, 1, 2 \to \mu_1 = 0.5$ (distance vs $|2 - 8.75| = 6.75$), $10, 11, 12 \to \mu_2 = 8.75$ (distance vs $|10 - 0.5| = 9.5$).

### Iteration 2

$\mu_1 = \text{mean}(0, 1, 2) = 1$. $\mu_2 = \text{mean}(10, 11, 12) = 11$.

Distortion: $1 + 0 + 1 + 1 + 0 + 1 = 4$. Assignments unchanged → converged.

### Lesson

Decrease in distortion: $273.5 \to 4$, two iterations. The bound "distortion non-increasing" (theory problem 3) holds *strictly* on every step here. Worse init can stall in a local minimum with all 6 points in one cluster — try $\mu_1 = -5$, $\mu_2 = 6$: you'll converge to a non-trivial split, but not the global one if the init is symmetric. **This is why k-means++ matters.**

---

## Example 3 — GMM-EM, one full iteration on a 4-point dataset

Points $x = (-1, 0, 4, 5)$. Init: $\mu_1 = -1$, $\mu_2 = 5$, $\sigma_1^2 = \sigma_2^2 = 1$, $\pi_1 = \pi_2 = 0.5$.

### E-step: responsibilities

$\gamma_{ik} = \pi_k \mathcal{N}(x_i; \mu_k, \sigma_k^2) / \sum_{k'} \pi_{k'} \mathcal{N}(x_i; \mu_{k'}, \sigma_{k'}^2)$.

| $x_i$ | $\mathcal{N}(\cdot; \mu_1=-1)$ | $\mathcal{N}(\cdot; \mu_2=5)$ | $\gamma_{i1}$ | $\gamma_{i2}$ |
|---|---|---|---|---|
| -1 | 0.399 | $\sim 10^{-9}$ | 1.000 | 0.000 |
| 0  | 0.242 | $\sim 10^{-6}$ | 1.000 | 0.000 |
| 4  | $\sim 10^{-6}$ | 0.242 | 0.000 | 1.000 |
| 5  | $\sim 10^{-9}$ | 0.399 | 0.000 | 1.000 |

(The Gaussian densities at 6+ units away are vanishingly small at $\sigma = 1$, so responsibilities collapse to hard assignments.)

### M-step: parameter updates

$N_1 = \sum_i \gamma_{i1} = 2$, $N_2 = 2$.

$\mu_1 = (1 \cdot (-1) + 1 \cdot 0 + 0 \cdot 4 + 0 \cdot 5) / 2 = -0.5$.
$\mu_2 = (0 + 0 + 4 + 5) / 2 = 4.5$.

$\sigma_1^2 = (1 \cdot ( -1 - (-0.5))^2 + 1 \cdot (0 - (-0.5))^2) / 2 = (0.25 + 0.25)/2 = 0.25$.
$\sigma_2^2 = (1 \cdot (4 - 4.5)^2 + 1 \cdot (5 - 4.5)^2) / 2 = 0.25$.

$\pi_1 = \pi_2 = 0.5$ unchanged.

### Observation

After one EM iteration the model tightens dramatically: variances drop $1 \to 0.25$, means refine, and the next iteration's responsibilities will be even sharper. The **monotonic log-likelihood** property (theory problem 2) means this only ever goes one way — your iteration is making progress in $\ell(\theta)$, just maybe to a local maximum.

### Sanity check

If the responsibilities had been "soft" (say 0.5, 0.5 for all points), the M-step would have shrunk both means toward the global mean = 2, killing the cluster structure. Sharp responsibilities preserve it. This is the "winner-take-all" character of EM with well-separated clusters, and the failure mode for poorly-initialised mixtures.

---

## What to do with these examples

For Example 1, vary the data slightly — add a third feature column or
shift one row — and watch the eigenvalues redistribute. For Example
2, try a poor init ($\mu_1 = 0.5$, $\mu_2 = 1.5$) and see how far
distortion drops in one step. For Example 3, increase $\sigma_1^2 =
\sigma_2^2 = 9$ and recompute the responsibilities; you'll see they
no longer collapse to hard assignments — that's the regime where
soft EM is meaningfully different from k-means.
