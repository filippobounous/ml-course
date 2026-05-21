# Week 1 — Worked examples

Concrete numerical walk-throughs to accompany `lecture_notes.md`. Each
example is small enough to do on paper in 10–15 minutes; the point is to
*feel* the mechanics, not to be impressed by the answer.

---

## Example 1 — SVD of a rank-deficient $3 \times 3$ matrix

Take

$$
A = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.
$$

### Step 1. Gram matrix

$$
A^\top A = \begin{pmatrix} 2 & 2 & 0 \\ 2 & 2 & 0 \\ 0 & 0 & 0 \end{pmatrix},
$$

eigenvalues $\{4, 0, 0\}$. Singular values $\sigma_i = \sqrt{\lambda_i}$
give $\sigma_1 = 2$, $\sigma_2 = \sigma_3 = 0$. **Rank = 1.**

### Step 2. Right singular vectors (eigenvectors of $A^\top A$)

The $\lambda = 4$ eigenvector is $v_1 = \tfrac{1}{\sqrt{2}}(1, 1, 0)^\top$.
A basis for the kernel of $A^\top A$ (the null space of $A$) is
$\{(1,-1,0)^\top/\sqrt{2}, \, (0,0,1)^\top\}$. Pick these as $v_2, v_3$.

### Step 3. Left singular vectors

$u_1 = A v_1 / \sigma_1 = \tfrac{1}{\sqrt{2}}(1,1,0)^\top$. Extend to an
orthonormal basis with $u_2 = (1,-1,0)^\top/\sqrt{2}$, $u_3 = (0,0,1)^\top$.

### Step 4. Reassemble

$$
A = U \Sigma V^\top, \quad U = \begin{pmatrix} 1/\sqrt 2 & 1/\sqrt 2 & 0 \\ 1/\sqrt 2 & -1/\sqrt 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \quad \Sigma = \mathrm{diag}(2, 0, 0), \quad V = U.
$$

Verify by multiplying out — you get $A$ back.

### Step 5. Pseudoinverse

$\Sigma^+ = \mathrm{diag}(1/2, 0, 0)$, so

$$
A^+ = V \Sigma^+ U^\top = \tfrac{1}{4}\begin{pmatrix} 1 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix} = \tfrac{1}{4} A.
$$

Check: for $b = (2, 2, 0)^\top$, $A^+ b = (1, 1, 0)^\top$, and this is
the **minimum-norm** least-squares solution (any vector $(1+t, 1-t, s)$
also solves $A x = b$ exactly, but $t = s = 0$ minimises $\|x\|$).

---

## Example 2 — KL between two Gaussians

For $p = \mathcal{N}(\mu_p, \sigma_p^2)$ and $q = \mathcal{N}(\mu_q, \sigma_q^2)$
(univariate), the closed-form KL is

$$
D_\text{KL}(p \| q) = \log\frac{\sigma_q}{\sigma_p} + \frac{\sigma_p^2 + (\mu_p - \mu_q)^2}{2 \sigma_q^2} - \frac{1}{2}.
$$

### Derivation

$$
D_\text{KL}(p \| q) = \mathbb{E}_p\!\left[ \log p(x) - \log q(x) \right].
$$

Log densities: $\log p(x) = -\tfrac{1}{2}\log(2\pi\sigma_p^2) - \tfrac{(x-\mu_p)^2}{2\sigma_p^2}$
and similarly for $q$. Subtract:

$$
\log p - \log q = \tfrac{1}{2}\log\frac{\sigma_q^2}{\sigma_p^2} + \tfrac{(x-\mu_q)^2}{2\sigma_q^2} - \tfrac{(x-\mu_p)^2}{2\sigma_p^2}.
$$

Take $\mathbb{E}_p$. Using $\mathbb{E}_p[(x-\mu_p)^2] = \sigma_p^2$ and
$\mathbb{E}_p[(x-\mu_q)^2] = \sigma_p^2 + (\mu_p - \mu_q)^2$ (variance-bias decomposition):

$$
D_\text{KL}(p\|q) = \tfrac{1}{2}\log\tfrac{\sigma_q^2}{\sigma_p^2} + \tfrac{\sigma_p^2 + (\mu_p - \mu_q)^2}{2\sigma_q^2} - \tfrac{1}{2}.
$$

### Numerical check

$p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(1, 4)$:

$$
D_\text{KL} = \log 2 + \tfrac{1 + 1}{8} - \tfrac{1}{2} \approx 0.693 + 0.25 - 0.5 = 0.443.
$$

Reverse $D_\text{KL}(q\|p) = -\log 2 + \tfrac{4 + 1}{2} - \tfrac{1}{2} \approx -0.693 + 2.5 - 0.5 = 1.307$.
**Asymmetry**: this is why KL is not a metric — and why VAEs / score-matching
make a deliberate choice of which direction to minimise.

---

## Example 3 — Gradient descent on a 1-D quadratic

$f(x) = \tfrac{L}{2} x^2$ with $L = 2$. Then $\nabla f(x) = L x$, smoothness
constant $= L$, optimum at $x^\star = 0$. Step size $\eta = 1/L = 0.5$.

### Iterations

$x_{t+1} = x_t - \eta \cdot L x_t = (1 - \eta L) x_t = 0 \cdot x_t = 0$.

The exact-step GD on a quadratic converges in **one** iteration. That's
not coincidence: for $\mu$-strongly-convex $L$-smooth $f$, GD with step
$1/L$ has linear rate $(1 - \mu/L)^T$, and a pure quadratic of curvature
$L$ has $\mu = L$, so $\mu/L = 1$.

### Now make it harder

$f(x) = \tfrac{1}{2} L_1 x_1^2 + \tfrac{1}{2} L_2 x_2^2$ with $L_1 = 1$,
$L_2 = 100$. Smoothness $L = 100$, strong convexity $\mu = 1$, condition
number $\kappa = 100$. Step $\eta = 1/L = 0.01$.

Coordinate-wise: $x_1^{(t+1)} = (1 - 0.01) x_1^{(t)}$, $x_2^{(t+1)} = 0 \cdot x_2^{(t)} = 0$.

The $x_2$ coordinate hits zero in one step (its update is exact); the
$x_1$ coordinate decays at rate $0.99$ per step. After $T = 300$ steps,
$|x_1| / |x_1^{(0)}| = 0.99^{300} \approx 0.049$. The theoretical bound

$$
f(x_T) - f^\star \le (1 - \mu/L)^T (f(x_0) - f^\star) = 0.99^T (f(x_0) - f^\star)
$$

matches the empirical decay almost exactly. The lesson: it's the **worst
direction** (the one with curvature $\mu$, not $L$) that dominates the
convergence — i.e. the **condition number** rules.

---

## What to do with these examples

Re-do them on paper *before* you read the answers. The point isn't the
numerical answer; it's the muscle memory of "decompose, eigenvalues,
verify a property, sanity-check on a small case." That same workflow
appears in W4 (PCA covariance eigenvalues), W5 (autograd via the chain
rule on small graphs), and W10 (verifying the DDPM noise schedule by
running it on a single image).
