# Week 1 — Mathematical foundations (lecture notes)

*Reading pair: Strang Ch.1–4 · MacKay Ch.2,4,8 · Boyd Ch.2–3,9.*

These notes connect four toolkits that we will reuse for the whole course: **linear
algebra as linear maps**, **probability on exponential families and KL**,
**convex optimization**, and a short **stochastic-dynamics primer** that will
come back in Week 10 (diffusion).

---

## 1. Linear algebra as linear maps

### Matrices are functions

Given $A \in \mathbb{R}^{m \times n}$, think of it as a map $A : \mathbb{R}^n \to \mathbb{R}^m$, $x \mapsto Ax$. Every ML algorithm we will write ultimately composes such maps with nonlinearities.

The **four fundamental subspaces** decompose $\mathbb{R}^n$ and $\mathbb{R}^m$:

- $\mathcal{N}(A) \perp \mathcal{R}(A^\top)$ in $\mathbb{R}^n$ (null space ⊥ row space).
- $\mathcal{N}(A^\top) \perp \mathcal{R}(A)$ in $\mathbb{R}^m$ (left null ⊥ column space).

### Singular value decomposition (SVD)

Every $A \in \mathbb{R}^{m \times n}$ factors as $A = U \Sigma V^\top$ with orthogonal $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$, and $\Sigma$ containing the singular values $\sigma_1 \ge \dots \ge \sigma_r > 0$ on its leading diagonal (with zeros elsewhere).

Geometric picture: $A$ first rotates the input (by $V^\top$), then stretches each axis (by $\Sigma$), then rotates the output (by $U$). SVD is the right mental model whenever you see "PCA" (Week 4), "denoising" (also Week 4), or "low-rank approximation".

**Eckart–Young theorem.** The best rank-$k$ approximation of $A$ in Frobenius norm is obtained by keeping the top-$k$ singular values. Proof: orthogonally project onto the top-$k$ singular subspace; compare to any other rank-$k$ $B$ by a dimension-counting argument.

### Moore–Penrose pseudoinverse

The pseudoinverse $A^+ = V \Sigma^+ U^\top$, where $\Sigma^+$ inverts the nonzero singular values and leaves zeros as zeros, is the *minimum-norm least-squares* solver: it returns the smallest-norm $x$ minimizing $\|Ax - b\|_2$.

When $A$ is full column rank, $A^+ = (A^\top A)^{-1} A^\top$, which is the OLS normal equation. Ridge regression adds a regulariser, equivalent to replacing $\Sigma^+$ with $\Sigma / (\Sigma^2 + \lambda I)$ — this is why ridge stabilises rank-deficient problems.

### Matrix calculus cheat-sheet

- $\nabla_x (a^\top x) = a$
- $\nabla_x (x^\top A x) = (A + A^\top) x$; if $A$ is symmetric, $2 A x$.
- $\nabla_x \|A x - b\|^2 = 2 A^\top (A x - b)$.
- $\nabla_X \log |\det X| = X^{-\top}$ (useful for normalising flows).

---

## 2. Probability: exponential families, entropy, KL

### Exponential families

A distribution $p(x \mid \theta) = h(x) \exp\{\theta^\top T(x) - A(\theta)\}$ is an exponential family with natural parameter $\theta$, sufficient statistic $T$, and log-partition $A(\theta)$. Bernoulli, Gaussian (with known variance), Poisson, multinomial — all exponential families. The log-partition is **convex** in $\theta$ and satisfies

$$\nabla_\theta A(\theta) = \mathbb{E}_{p_\theta}[T(X)], \qquad \nabla^2_\theta A(\theta) = \operatorname{Cov}_{p_\theta}(T(X)).$$

This is why MLE of exponential families has a clean closed-form in terms of empirical moments.

### Entropy, cross-entropy, KL

For a density $p$ on a discrete alphabet,

$$H(p) = -\sum_x p(x) \log p(x), \qquad H(p, q) = -\sum_x p(x) \log q(x),$$

$$D_\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p).$$

**KL is non-negative** (Gibbs' inequality) and equals zero iff $p = q$ almost everywhere. Pinsker: $\|p - q\|_1 \le \sqrt{2 D_\text{KL}(p \| q)}$.

In ML the standard classification loss $-\sum_i \log q_\theta(y_i \mid x_i)$ is exactly $N \cdot H(\hat p, q_\theta)$ — minimising cross-entropy is minimising KL between the empirical and model distributions.

### Mutual information and Fisher information

$I(X; Y) = D_\text{KL}(p(x, y) \| p(x) p(y)) = H(X) - H(X \mid Y)$; quantifies how much knowing $Y$ reduces uncertainty about $X$. InfoNCE (CLIP, Week 10) is a tractable lower bound on MI.

The Fisher information matrix $\mathcal{F}(\theta) = \mathbb{E}_{p_\theta}[\nabla_\theta \log p_\theta(X) \, \nabla_\theta \log p_\theta(X)^\top]$ gives the Cramér–Rao lower bound on estimator variance and, up to the Hessian of $-\log p$, the natural-gradient preconditioner.

---

## 3. Convex optimization

### Convex sets and functions

A set $C$ is convex if $\lambda x + (1 - \lambda) y \in C$ for $x, y \in C$ and $\lambda \in [0, 1]$. A function $f$ is convex iff its epigraph is convex; equivalently iff $f(\lambda x + (1 - \lambda) y) \le \lambda f(x) + (1 - \lambda) f(y)$. Twice-differentiable $f$ is convex iff $\nabla^2 f \succeq 0$.

**Convex ML losses:** squared error, cross-entropy, logistic loss, hinge loss. Non-convex: any neural network with more than one layer — which is why deep-learning optimisation is a distinct subject.

### Gradient descent: convergence

For $L$-smooth convex $f$ with step size $\eta = 1/L$,

$$f(x_T) - f^\star \le \frac{L \|x_0 - x^\star\|^2}{2 T}.$$

Add $\mu$-strong convexity and you get linear convergence $f(x_T) - f^\star \le (1 - \mu/L)^T [f(x_0) - f^\star]$. The **condition number** $\kappa = L / \mu$ controls how fast you can run.

**Stochastic gradient descent** replaces the exact gradient with an unbiased estimate; the optimal step size scales as $1/\sqrt{T}$ in the general convex case, with momentum and adaptive methods (Adam, AdamW) accelerating in practice.

### Lagrangian duality (as needed)

For $\min f(x)$ s.t. $g_i(x) \le 0$, the Lagrangian is $L(x, \lambda) = f(x) + \sum_i \lambda_i g_i(x)$ and the dual is $d(\lambda) = \inf_x L(x, \lambda)$. Strong duality holds under Slater's condition for convex problems; SVMs (Week 3) live entirely in this framework.

---

## 4. Stochastic dynamics primer (for Week 10)

### Brownian motion and Langevin SDE

The overdamped Langevin SDE

$$d X_t = -\nabla U(X_t) \, dt + \sqrt{2} \, dW_t$$

has (under regularity) stationary measure $\pi(x) \propto e^{-U(x)}$. Intuition: gradient descent driven by thermal noise, equilibrating to a Gibbs measure at temperature $1$.

### Fokker–Planck equation

The density $p(x, t)$ of $X_t$ evolves as

$$\partial_t p = \nabla \cdot (p \nabla U) + \Delta p.$$

Diffusion models (Week 10) are built on exactly this equation with a *reversed* time direction and a learned score $\nabla \log p$.

### Euler–Maruyama simulation

Discretisation: $X_{t+1} = X_t - \nabla U(X_t) \Delta t + \sqrt{2 \Delta t} \, \xi_t$, $\xi_t \sim \mathcal{N}(0, I)$. We will simulate this for a double-well potential in the problem set.

---

## Glossary (for later reference)

- **Frobenius norm.** $\|A\|_F^2 = \sum_{ij} A_{ij}^2 = \operatorname{tr}(A^\top A) = \sum_i \sigma_i^2$.
- **Rayleigh quotient.** $R(x) = x^\top A x / x^\top x$; equals eigenvalues at eigenvectors.
- **Jensen's inequality.** For convex $f$, $f(\mathbb{E} X) \le \mathbb{E} f(X)$.
- **Cauchy–Schwarz.** $|\langle x, y \rangle| \le \|x\| \, \|y\|$.

## What to do with these notes

Work the problem set in `../problems/README.md`. In particular: derive every inequality you use, implement the pseudoinverse and Gaussian MLE in NumPy (reference code in `../problems/solutions.py`), and simulate the double-well Langevin SDE.
