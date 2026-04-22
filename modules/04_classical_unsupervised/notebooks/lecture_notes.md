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

Work the problem set in `../problems/README.md`. Implement GMM-EM in NumPy (reference in `../problems/solutions.py`). Build the portfolio artifact in `../../../portfolio/04_pca_statarb/` — a walk-forward PCA stat-arb backtest on simulated returns (and optionally Ken French industry data when offline).
