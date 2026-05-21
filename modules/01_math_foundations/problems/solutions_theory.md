# Week 1 — Theory-problem solutions

Reference proofs for the theory items in `README.md` §Theory. Work each one yourself first; consult this only to check.

## 1. SVD identities

### (a) Pseudoinverse

For $A = U\Sigma V^\top$ with $\Sigma = \operatorname{diag}(\sigma_i)$, define $\Sigma^+ = \operatorname{diag}(\sigma_i^{-1}$ if $\sigma_i > 0$, else $0)$.

Direct verification of the Moore–Penrose axioms:

- $A A^+ A = U\Sigma V^\top V\Sigma^+ U^\top U\Sigma V^\top = U\Sigma\Sigma^+\Sigma V^\top = U\Sigma V^\top = A$ (using $V^\top V = I$, $U^\top U = I$, and $\Sigma\Sigma^+\Sigma = \Sigma$).
- Similarly $A^+ A A^+ = A^+$.
- $(AA^+)^\top = (U\Sigma\Sigma^+ U^\top)^\top = AA^+$ since $\Sigma\Sigma^+$ is diagonal, hence symmetric.
- $(A^+A)^\top = A^+A$ by the same argument.

All four axioms hold, so $A^+ = V\Sigma^+ U^\top$.

### (b) Frobenius norm

$\|A\|_F^2 = \operatorname{tr}(A^\top A) = \operatorname{tr}(V\Sigma^\top U^\top U \Sigma V^\top) = \operatorname{tr}(V\Sigma^2 V^\top) = \operatorname{tr}(\Sigma^2) = \sum_i \sigma_i^2$

using cyclic property of trace and $V^\top V = I$.

### (c) Eckart–Young

Let $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$ be the truncated SVD. Claim: for any matrix $B$ of rank $\le k$,

$$\|A - A_k\|_F^2 \le \|A - B\|_F^2.$$

**Proof sketch.** Let $B = X Y^\top$ with $X \in \mathbb{R}^{m \times k}$, $Y \in \mathbb{R}^{n \times k}$ (rank-$k$ factorisation). By the Courant–Fischer min-max characterisation of singular values:

$$\sigma_{k+1}(A) = \min_{\dim S = m - k} \max_{x \in S, \|x\|=1} \|A x\|.$$

Apply this with $S = \mathcal{N}(X^\top)$, which has dimension $\ge m - k$. Any $x \in S$ satisfies $B^\top x = Y X^\top x = 0$, so $\|A - B\|_2 \ge \|(A-B)x\| = \|Ax\|$, giving $\|A - B\|_2 \ge \sigma_{k+1}(A)$.

A Weyl-style interlacing argument (singular values of $A$ vs of $A - B$) then upgrades the operator-norm bound to the Frobenius bound

$$\|A - B\|_F^2 \ge \sum_{i=k+1}^r \sigma_i(A)^2 = \|A - A_k\|_F^2.$$

Equality holds at $B = A_k$. See Strang §1.8 or Horn–Johnson Thm. 7.4.9.1 for the interlacing details.

## 2. KL properties

### Non-negativity (Gibbs)

$D_{KL}(p\|q) = \mathbb{E}_{p}\!\left[\log\frac{p(X)}{q(X)}\right] \ge -\log \mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = -\log \int q(x) \, dx = 0$ using Jensen (log is concave) and $\int q = 1$. Equality in Jensen iff $q/p$ is constant $p$-a.e., i.e. $p = q$ a.e.

### Pinsker

Let $\Delta = \{x : p(x) \ge q(x)\}$. Total variation $\|p-q\|_1/2 = \int_\Delta (p-q)\,dx =: \delta$. Define a Bernoulli reduction $P = \operatorname{Ber}(\int_\Delta p)$, $Q = \operatorname{Ber}(\int_\Delta q)$; data-processing inequality gives $D_{KL}(p\|q) \ge D_{KL}(P\|Q)$. Now prove Pinsker in the Bernoulli case by elementary calculus: $f(\delta) = 2\delta^2 \le D_{KL}(P\|Q)$. Combine → $\|p-q\|_1 \le \sqrt{2 D_{KL}(p\|q)}$.

## 3. Log-sum-exp is convex

$f(x) = \log \sum_i e^{x_i}$. Gradient: $\nabla f = p$ where $p_i = e^{x_i} / \sum_j e^{x_j}$ (softmax). Hessian: $\nabla^2 f = \operatorname{diag}(p) - p p^\top$. For any $v$: $v^\top (\operatorname{diag}(p) - pp^\top) v = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2 = \operatorname{Var}_p(v) \ge 0$ (variance of $v$ under categorical $p$). So $\nabla^2 f \succeq 0$ → convex. Singular when $v$ is constant on the support of $p$, i.e. along the $(1,\dots,1)$ direction when $p$ has full support.

## 4. Gradient-descent convergence for $L$-smooth convex $f$

$L$-smoothness: $f(y) \le f(x) + \nabla f(x)^\top (y-x) + \tfrac{L}{2}\|y-x\|^2$. With step $\eta = 1/L$ and $x_{t+1} = x_t - \eta \nabla f(x_t)$, plug $y = x_{t+1}$:

$$f(x_{t+1}) \le f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2. \quad (\star)$$

So the loss is monotone decreasing — call this **descent**.

**Step 1. Distance to optimum is non-increasing.** Expanding $\|x_{t+1} - x^\star\|^2$ and using convexity $\nabla f(x_t)^\top (x_t - x^\star) \ge f(x_t) - f^\star$:

$$\|x_{t+1} - x^\star\|^2 = \|x_t - x^\star\|^2 - \tfrac{2}{L}\nabla f(x_t)^\top (x_t - x^\star) + \tfrac{1}{L^2}\|\nabla f(x_t)\|^2.$$

Combine with $(\star)$ rewritten as $\tfrac{1}{L^2}\|\nabla f(x_t)\|^2 \le \tfrac{2}{L}(f(x_t) - f(x_{t+1}))$:

$$\|x_{t+1} - x^\star\|^2 \le \|x_t - x^\star\|^2 - \tfrac{2}{L}(f(x_{t+1}) - f^\star).$$

In particular $\|x_T - x^\star\| \le \|x_0 - x^\star\|$ — distance to optimum never grows.

**Step 2. Telescope.** Sum the above for $t = 0, \dots, T-1$:

$$\tfrac{2}{L} \sum_{t=1}^T (f(x_t) - f^\star) \le \|x_0 - x^\star\|^2 - \|x_T - x^\star\|^2 \le \|x_0 - x^\star\|^2.$$

By descent ($(\star)$ ⟹ $f(x_t)$ non-increasing), $f(x_T) - f^\star \le \tfrac{1}{T} \sum_{t=1}^T (f(x_t) - f^\star)$. Therefore

$$f(x_T) - f^\star \le \tfrac{L \|x_0 - x^\star\|^2}{2T}.$$

Rate is $\mathcal{O}(1/T)$ for smooth-convex.

**Adding strong convexity.** $\mu$-strong convexity strengthens convexity to $f(x) \ge f(x^\star) + \tfrac{\mu}{2}\|x - x^\star\|^2$ (the quadratic lower bound). Repeating Step 1 with this strengthening gives a *contraction* on $\|x_t - x^\star\|^2$:

$$\|x_{t+1} - x^\star\|^2 \le (1 - \mu/L) \|x_t - x^\star\|^2,$$

iterating to $\|x_T - x^\star\|^2 \le (1 - \mu/L)^T \|x_0 - x^\star\|^2$, hence $f(x_T) - f^\star \le \tfrac{L}{2}(1 - \mu/L)^T \|x_0 - x^\star\|^2$ — **linear convergence**, the exponential improvement. The ratio $\kappa = L/\mu$ is the **condition number** and controls how aggressive a step you can take.
