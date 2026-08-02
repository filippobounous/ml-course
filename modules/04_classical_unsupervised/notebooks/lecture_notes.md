# Week 4 — Classical unsupervised learning (lecture notes)

*Reading pair: ESL Ch.14 · Bishop Ch.9, 12 · Avellaneda & Lee 2008.*

---

## 1. PCA: three equivalent views

Let $X \in \mathbb{R}^{N \times p}$ have centered columns. PCA can be defined in three equivalent ways.

### (a) Variance maximisation

Find $w \in \mathbb{R}^p$, $\|w\| = 1$ maximising $w^\top S w$ where $S = X^\top X / N$. Lagrangian → $w$ is the top eigenvector of $S$.

### (b) Reconstruction error minimisation

Find rank-$k$ projection $P$ minimising $\|X - X P\|_F^2$. Solution: project onto the top-$k$ eigenspace of $S$. This is the Eckart–Young theorem again.

### (c) Probabilistic PCA (Tipping & Bishop 1999)

Model $z \sim \mathcal{N}(0, I_k)$, $x = W z + \mu + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$. MLE of $W$ recovers the span of the top-$k$ PCs. In the limit $\sigma^2 \to 0$ the MLE is exactly standard PCA.

### Computing PCA

Always via **SVD** on the centered data: $X = U \Sigma V^\top$, PC directions are columns of $V$, scores are $U \Sigma$. Eigendecomposition of $X^\top X$ is equivalent but numerically worse.

## 2. k-means and GMM/EM

### k-means as coordinate descent

Alternating:
- **Assignment step**: $z_i = \arg\min_j \|x_i - \mu_j\|^2$.
- **Update step**: $\mu_j = \operatorname{mean}(\{x_i : z_i = j\})$.

Each step decreases $\sum_i \|x_i - \mu_{z_i}\|^2$ (distortion). Converges to a local minimum; sensitive to initialisation — use **k-means++** seeding.

### Gaussian mixture model

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k).$$

**E-step**: responsibilities $\gamma_{ik} = \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) / \sum_{k'} \pi_{k'} \mathcal{N}(x_i \mid \mu_{k'}, \Sigma_{k'})$.

**M-step**:

$$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i \gamma_{ik}}, \quad \pi_k = \frac{1}{N} \sum_i \gamma_{ik}.$$

**Convergence.** EM monotonically increases the observed-data log-likelihood because each iteration maximises a Jensen lower bound that is tight at the current parameters.

k-means is the zero-variance limit of EM on an isotropic, equal-mixing-weight GMM.

## 3. Density estimation

- **KDE.** $\hat p(x) = \frac{1}{N h^d} \sum_i K\!\left(\frac{x - x_i}{h}\right)$. Bandwidth $h$ trades bias vs variance; Silverman's rule of thumb is a decent default for Gaussian kernels.
- **Normalising flows** (preview for Week 10). Parameterise an invertible map $f : \mathbb{R}^d \to \mathbb{R}^d$; by change of variables $p_X(x) = p_Z(f(x)) |\det J_f(x)|$. MLE on $\{x_i\}$. RealNVP / Glow / NICE enforce tractable Jacobians by factorising $f$ as coupling layers.

## 4. Application — PCA statistical arbitrage

Given returns $R \in \mathbb{R}^{T \times p}$ (assets × periods):

1. Center and scale each column.
2. PCA on the rolling covariance window to identify the top-$k$ market / factor components.
3. For each asset, regress returns on the top-$k$ PCs → residuals = idiosyncratic return.
4. Compute a rolling **z-score** $z_t = (r_t^{\text{resid}} - \mu) / \sigma$; trade the mean-reverting residual.
5. Evaluate with **walk-forward splits**: only information up to time $t$ used for decisions at $t+1$.

This is the core of Avellaneda & Lee (2008), with many variants: OU-calibrated half-life, sector/industry neutralisation, beta-hedging to the market.

**Honest reporting.** Report in-sample and out-of-sample Sharpe, turnover, max drawdown, and the breakdown of return across calendar years. Apply a realistic transaction-cost model (e.g. 5–10 bp per side for liquid equities).

## What to do with these notes

Work the problem set in `../problems/README.md`. Implement GMM-EM in NumPy in `../problems/starter.py` (reference in `../problems/_reference/solutions.py`). Build the portfolio artifact in `../../../portfolio/04_pca_statarb/` — a walk-forward PCA stat-arb backtest on simulated returns (and optionally Ken French industry data when offline).

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three numerical exercises (PCA on a 4-sample toy, k-means on 6 points by hand, one full GMM-EM iteration on a 4-point dataset).

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1 PCA | 4 | Three equivalent views; do the toy SVD by hand; verify sklearn agrees. |
| §2 k-means + GMM/EM | 5 | Lloyd's algorithm on 6 points; ELBO + Jensen for EM monotonicity. |
| §3 Density estimation | 2 | KDE bandwidth selection; normalising-flow preview. |
| §4 PCA stat-arb | 5 | Avellaneda–Lee residual construction; walk-forward backtest with realistic transaction costs. |
| Problem set | 3 | 2 theory + 2 implementation, test-graded via `tests/week_04/`. |
| Office hours / review | 1 | Check proofs against `problems/solutions_theory.md`. |

## Self-assessment rubric

Before moving to Week 5, you should be able to answer "yes" to all of:

1. Can I derive PCA three ways (variance maximisation, reconstruction-error minimisation, probabilistic PCA in the zero-noise limit) and explain why SVD is the preferred numerical route?
2. Can I derive the GMM E-step and M-step updates and prove EM monotonically improves the log-likelihood via the ELBO + Jensen?
3. Can I show Lloyd's algorithm decreases distortion at every step and explain why k-means++ matters for initialisation?
4. Can I implement PCA two ways (SVD vs power iteration) and explain when each is preferred (numerical stability vs runtime for large-$N$, small-$k$)?
5. Can I construct a PCA stat-arb residual signal, run a walk-forward backtest, and explain why turnover and transaction-cost modelling determine whether the IS Sharpe survives OOS?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **PCA ↔ principal axes of inertia.** The sample covariance matrix is the **inertia tensor** of the centered data cloud, treating each sample as a unit mass at $x_i$. Diagonalising $S$ produces the principal moments (singular values $\sigma_i^2$) and principal axes (eigenvectors $v_i$). Same algebra you've done a hundred times for rigid-body dynamics.
- **GMM-EM ↔ mean-field iteration on a soft-assignment free energy.** The ELBO is a **variational free-energy functional** $\mathcal{F}(q, \theta) = -\langle \log p \rangle_q - H(q)$. The E-step minimises $\mathcal{F}$ over the variational distribution $q$ (mean field); the M-step minimises $\mathcal{F}$ over physical parameters $\theta$. EM monotonicity is the second law in this micro-system — free energy never increases.
- **k-means ↔ zero-temperature limit of GMM-EM.** Soft Boltzmann responsibilities $\gamma_{ik} \propto e^{-\|x_i - \mu_k\|^2 / 2T}$ collapse to hard assignments as $T \to 0$. The deterministic-annealing algorithm interpolates between the two by lowering $T$ gradually — same idea as simulated annealing on the cluster-assignment free energy.
- **PCA stat-arb ↔ Marčenko–Pastur / random matrix theory.** Splitting eigenvalues into "signal" (factor modes) vs "noise" (idiosyncratic residuals) is exactly the spectral-density argument from RMT: bulk MP distribution sets the noise floor, eigenvalues outside it are the systematic factors. This is the rigorous justification for choosing the truncation level $k$.

Keep these bridges live; W10 (diffusion ≡ reverse-time Langevin on a learned potential) and W12 (PINN ≡ functional-gradient descent on a PDE residual) extend the variational / functional-gradient lens.
