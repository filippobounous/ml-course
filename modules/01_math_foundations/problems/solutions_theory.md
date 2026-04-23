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

Let $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$. For any rank-$k$ matrix $B$, write $B = \sum_{i=1}^k x_i y_i^\top$ and apply Courant–Fischer: the best rank-$k$ approximation in Frobenius (and operator) norm has error $\|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$. The full argument goes via interlacing of singular values; see Strang §1.8.

## 2. KL properties

### Non-negativity (Gibbs)

$D_{KL}(p\|q) = \mathbb{E}_{p}\!\left[\log\frac{p(X)}{q(X)}\right] \ge -\log \mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = -\log \int q(x) \, dx = 0$ using Jensen (log is concave) and $\int q = 1$. Equality in Jensen iff $q/p$ is constant $p$-a.e., i.e. $p = q$ a.e.

### Pinsker

Let $\Delta = \{x : p(x) \ge q(x)\}$. Total variation $\|p-q\|_1/2 = \int_\Delta (p-q)\,dx =: \delta$. Define a Bernoulli reduction $P = \operatorname{Ber}(\int_\Delta p)$, $Q = \operatorname{Ber}(\int_\Delta q)$; data-processing inequality gives $D_{KL}(p\|q) \ge D_{KL}(P\|Q)$. Now prove Pinsker in the Bernoulli case by elementary calculus: $f(\delta) = 2\delta^2 \le D_{KL}(P\|Q)$. Combine → $\|p-q\|_1 \le \sqrt{2 D_{KL}(p\|q)}$.

## 3. Log-sum-exp is convex

$f(x) = \log \sum_i e^{x_i}$. Gradient: $\nabla f = p$ where $p_i = e^{x_i} / \sum_j e^{x_j}$ (softmax). Hessian: $\nabla^2 f = \operatorname{diag}(p) - p p^\top$. For any $v$: $v^\top (\operatorname{diag}(p) - pp^\top) v = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2 = \operatorname{Var}_p(v) \ge 0$ (variance of $v$ under categorical $p$). So $\nabla^2 f \succeq 0$ → convex. Singular when $v$ is constant on the support of $p$, i.e. along the $(1,\dots,1)$ direction when $p$ has full support.

## 4. Gradient-descent convergence for $L$-smooth convex $f$

$L$-smoothness means $f(y) \le f(x) + \nabla f(x)^\top (y-x) + \tfrac{L}{2}\|y-x\|^2$. With step $\eta = 1/L$, $x_{t+1} = x_t - \eta \nabla f(x_t)$. Plugging $y = x_{t+1}$:

$f(x_{t+1}) \le f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2.$

Summing over $t = 0, \dots, T-1$: $\sum_t \|\nabla f(x_t)\|^2 \le 2L(f(x_0) - f(x_T))$. Then by convexity $f(x_t) - f^\star \le \nabla f(x_t)^\top (x_t - x^\star)$ and Cauchy–Schwarz + a telescoping argument (Nesterov Lemma 1.2.3, or Bubeck §3.2) yields

$$f(x_T) - f^\star \le \frac{L\|x_0 - x^\star\|^2}{2T}.$$

The rate is $\mathcal{O}(1/T)$ for smooth-convex. Add $\mu$-strong convexity to get the exponential improvement $(1 - \mu/L)^T$.
