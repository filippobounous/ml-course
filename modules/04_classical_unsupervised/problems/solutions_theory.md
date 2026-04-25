# Week 4 — Theory-problem solutions

## 1. Probabilistic PCA → PCA in the zero-noise limit

Model: $z \sim \mathcal{N}(0, I_k)$, $x = W z + \mu + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_p)$. Marginal: $x \sim \mathcal{N}(\mu, C)$ with $C = WW^\top + \sigma^2 I$.

MLE of $\mu$ is the sample mean $\bar x$. For $W$, Tipping & Bishop (1999) show the MLE is

$\hat W_\text{ML} = U_k (\Lambda_k - \sigma^2 I)^{1/2} R$

where $U_k$ are top-$k$ eigenvectors of the sample covariance $S$, $\Lambda_k$ is a diagonal of the top-$k$ eigenvalues, and $R$ is any $k\times k$ orthogonal matrix. As $\sigma^2 \to 0$: $\hat W \to U_k \Lambda_k^{1/2} R$; the column space is exactly the top-$k$ PC subspace. Hence PPCA reduces to PCA.

## 2. EM for GMM monotonically improves the likelihood

Let $\ell(\theta) = \log \sum_z p_\theta(x, z) = \log \int p_\theta(x|z) p_\theta(z) dz$. Introduce a variational posterior $q(z)$:

$\ell(\theta) = \underbrace{\mathbb{E}_{q}[\log p_\theta(x,z)] + H(q)}_{\mathcal{F}(q, \theta)} + D_{KL}(q \| p_\theta(\cdot|x))$

(this is the ELBO decomposition). $\mathcal{F}$ is a lower bound on $\ell$; equality iff $q(z) = p_\theta(z|x)$.

**E-step.** Hold $\theta^{(t)}$ fixed; set $q^{(t+1)}(z) = p_{\theta^{(t)}}(z|x)$. This makes $\mathcal{F}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$.

**M-step.** Maximise $\mathcal{F}(q^{(t+1)}, \theta)$ over $\theta$. By construction $\mathcal{F}(q^{(t+1)}, \theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})$.

Since $\ell(\theta^{(t+1)}) \ge \mathcal{F}(q^{(t+1)}, \theta^{(t+1)})$ always, we conclude $\ell(\theta^{(t+1)}) \ge \ell(\theta^{(t)})$: **EM monotonically improves the observed-data log-likelihood**.

For GMM the M-step reduces to

$\mu_k \leftarrow \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \Sigma_k \leftarrow \frac{\sum_i \gamma_{ik}(x_i - \mu_k)(x_i-\mu_k)^\top}{\sum_i \gamma_{ik}}, \quad \pi_k \leftarrow \frac{1}{N}\sum_i \gamma_{ik},$

where $\gamma_{ik} = p_{\theta^{(t)}}(z_i = k | x_i)$ are the E-step responsibilities.

## 3. Lloyd's algorithm decreases distortion

Distortion $D(\mu, z) = \sum_i \|x_i - \mu_{z_i}\|^2$. Lloyd alternates:

- **Assignment.** $z_i \leftarrow \arg\min_k \|x_i - \mu_k\|^2$. This can only decrease $D$ holding $\mu$ fixed (each term's new value is $\le$ its old value).
- **Update.** $\mu_k \leftarrow \arg\min_\mu \sum_{i: z_i = k} \|x_i - \mu\|^2 = \operatorname{mean}(\{x_i : z_i = k\})$. Optimal holding $z$ fixed; decreases $D$ on each cluster.

$D$ is bounded below by 0 and strictly decreases while assignments change, so Lloyd converges in finitely many iterations to a local minimum. It is not guaranteed global — initialisation matters, which is why k-means++ seeding is standard.

## Avellaneda–Lee cross-reference

For the stat-arb portfolio: each asset's residual is modelled as OU with mean-reversion speed $\kappa$ estimated from the rolling-window AR(1) coefficient $\phi$: $\kappa = -\log\phi / \Delta t$. Half-life $\tau_{1/2} = \log 2 / \kappa$. A trade is opened when the z-score of the cumulative residual crosses a threshold and closed when it mean-reverts toward zero. The portfolio-level Sharpe is dominated by how well the half-life estimate holds out of sample.
