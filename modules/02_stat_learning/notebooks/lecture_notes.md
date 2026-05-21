# Week 2 — Statistical learning and ERM (lecture notes)

*Reading pair: Bishop PRML Ch.1 & 3 · Shalev-Shwartz & Ben-David Ch.2–5 · Murphy PML-1 Ch.4, 11.*

---

## 1. The learning framework

We are given $N$ i.i.d. samples $\{(x_i, y_i)\}_{i=1}^N$ drawn from an unknown joint distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$. We choose a **hypothesis class** $\mathcal{H}$ of predictors $h : \mathcal{X} \to \mathcal{Y}$, a **loss** $\ell : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0}$, and look for $h^\star$ minimising the risk

$$R(h) = \mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(h(x), y)].$$

Since $\mathcal{D}$ is unknown, we instead minimise the **empirical risk**

$$\hat R(h) = \frac{1}{N} \sum_{i=1}^N \ell(h(x_i), y_i),$$

hence the name **ERM**. The gap $R(h) - \hat R(h)$ is the generalisation error; bounding it is what statistical learning theory is about.

## 2. Bias–variance decomposition

For squared loss and a fixed test point $x$ with $y = f(x) + \varepsilon$, $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Var}(\varepsilon) = \sigma^2$:

$$\mathbb{E}[(h_\mathcal{S}(x) - y)^2] = \underbrace{(\mathbb{E}[h_\mathcal{S}(x)] - f(x))^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}(h_\mathcal{S}(x))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}},$$

where the expectation is over the training set $\mathcal{S}$. Ridge regression trades bias for variance: as $\lambda \uparrow$, variance drops and bias grows.

## 3. MLE, MAP, and the Bayesian picture

For a parametric family $p(y \mid x, \theta)$,

- **MLE:** $\hat\theta = \arg\max_\theta \prod_i p(y_i \mid x_i, \theta)$.
- **MAP:** $\hat\theta = \arg\max_\theta \prod_i p(y_i \mid x_i, \theta) \, \pi(\theta)$.
- **Posterior:** $p(\theta \mid \mathcal{D}) \propto \prod_i p(y_i \mid x_i, \theta) \, \pi(\theta)$.

**Ridge = MAP.** For linear regression with Gaussian likelihood $\mathcal{N}(y \mid x^\top \beta, \sigma^2)$ and Gaussian prior $\mathcal{N}(\beta \mid 0, \tau^2 I)$, the MAP estimator is

$$\hat\beta_\text{MAP} = (X^\top X + \lambda I)^{-1} X^\top y, \qquad \lambda = \sigma^2 / \tau^2.$$

**Lasso = MAP under Laplace prior** $\pi(\beta) \propto e^{-\|\beta\|_1 / b}$.

## 4. Linear regression: closed form and SGD

For $y = X\beta + \varepsilon$ with homoscedastic Gaussian noise,

$$\hat\beta_\text{OLS} = (X^\top X)^{-1} X^\top y = X^+ y.$$

When $N < p$ or $X^\top X$ is ill-conditioned, use ridge (λ-smoothed normal equations) or the SVD-based pseudoinverse from Week 1. For very large $N$, use **SGD**:

$$\beta^{(t+1)} = \beta^{(t)} - \eta (x_i^\top \beta^{(t)} - y_i) x_i.$$

Step-size choice: $\eta = 1 / L$ with $L$ the largest eigenvalue of $X^\top X / N$ gives convergence in expectation; momentum / Adam accelerate. Ridge adds $-\eta \lambda \beta^{(t)}$ (weight decay).

## 5. Cross-validation

K-fold CV partitions the data into $K$ folds, trains on $K-1$ folds and evaluates on the held-out fold, averaging the $K$ estimates. LOO-CV ($K = N$) is approximately unbiased but has high variance; $K \in \{5, 10\}$ is the usual trade-off.

Two pitfalls to internalise:
1. **Leakage.** Any transformation fitted on the full training set (e.g. mean imputation, feature scaling, PCA) must be refitted *inside* each fold.
2. **Nested CV.** If you tune hyperparameters on the CV folds, you need an outer loop of CV to get an unbiased estimate of generalisation error.

## 6. Where this leads

- **Logistic regression** (Week 3) is MLE under a Bernoulli likelihood with logit link.
- **Gradient boosting** (Week 3) is functional-gradient descent on the empirical risk.
- **Neural networks** (Week 5) are MLE / MAP with a richer parametric family — but the loss landscape is non-convex and ERM no longer has generalisation guarantees.

## 7. MDP primer (for Week 11)

A finite MDP is a tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ with transition kernel $p(s' \mid s, a)$, reward $r(s, a)$, and discount $\gamma \in [0, 1)$. A policy $\pi(a \mid s)$ induces a return $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$. The Bellman operator on state-value functions is

$$(T^\pi V)(s) = \mathbb{E}_{a \sim \pi, s' \sim p}[r(s, a) + \gamma V(s')].$$

$T^\pi$ is a $\gamma$-contraction in sup-norm, so value iteration converges. Everything else in Week 11 is extensions of this.

## What to do with these notes

Work the problem set in `../problems/README.md`. The portfolio artifact this week is the NumPy linear-models library in `../../../portfolio/02_numpy_linreg/` — it implements closed-form OLS, SGD, ridge, lasso (coordinate descent), and K-fold CV, with full tests.

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three numerical exercises (ridge bias/variance on a 1-D problem, Bernoulli–Beta Bayesian update, 5-fold CV picking $\lambda$) that build the intuition the open-ended problems need.

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1–2 ERM + bias–variance | 4 | Derive bias–variance for squared loss; plot the U-shape across $\lambda$ on a toy ridge problem. |
| §3 MLE / MAP / Bayes | 3 | Ridge ≡ MAP under Gaussian prior; Bernoulli + Beta conjugate update on paper. |
| §4 Linear regression | 3 | Closed-form OLS + unbiasedness + covariance; SGD convergence on a toy quadratic. |
| §5 Cross-validation | 2 | K-fold + LOO bias; the leakage pitfall (scale-inside-the-fold). |
| §7 MDP primer | 2 | Bellman optimality contraction; value iteration on a 5-state chain. |
| Problem set | 4 | 2 theory + 2 implementation + 1 applied, test-graded via `tests/week_02/`. |
| Office hours / review | 2 | Check proofs against `problems/solutions_theory.md`. |

## Self-assessment rubric

Before moving to Week 3, you should be able to answer "yes" to all of:

1. Can I derive the bias–variance decomposition for squared loss, and identify which term ridge increases and which it decreases?
2. Can I prove that ridge regression equals MAP under a Gaussian prior, and state the prior variance in terms of $\lambda$ and $\sigma^2$?
3. Can I derive $\mathbb{E}[\hat\beta_\text{OLS}] = \beta$ and $\operatorname{Var}(\hat\beta_\text{OLS}) = \sigma^2(X^\top X)^{-1}$ from the closed-form?
4. Can I explain why LOO-CV is approximately (not exactly) unbiased and why it has high variance?
5. Can I prove the Bellman optimality operator $T^\star$ is a $\gamma$-contraction in sup-norm, and run tabular value iteration on a 5-state grid world to convergence?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **ERM ↔ zero-temperature Gibbs.** The empirical risk $\hat R(h)$ is a "potential" on hypothesis space; choosing $\arg\min \hat R$ is the $T \to 0$ limit of $p(h) \propto \exp(-\hat R(h)/T)$. **PAC-Bayes** bounds (W2 stretch reading) live at $T > 0$ and recover the "entropy of the posterior" as a regulariser — the **variational free energy** $\hat R + T \cdot D_\text{KL}(\text{posterior} \| \text{prior})$. ELBO (used in W10 diffusion) is this object minimised at fixed $T$.
- **Bias–variance ↔ fluctuation-dissipation.** Bias$^2$ is the deterministic response of the predictor to the true signal; variance is the susceptibility (mean fluctuation under dataset resampling). The total MSE behaves like internal energy = mean$^2$ + fluctuation$^2$. Increasing model capacity is like increasing degrees of freedom: lower bias, higher variance.
- **Ridge ↔ harmonic oscillator on parameter space.** The Gaussian prior $\beta \sim \mathcal{N}(0, \tau^2 I)$ is a harmonic potential $V(\beta) = \tfrac{1}{2\tau^2}\|\beta\|^2$. Adding the data term gives a quadratic Hamiltonian with shifted minimum (the ridge estimator) and inverse-Hessian covariance — exactly the linear-response analogue.
- **Value iteration ↔ Lax–Friedrichs sweep.** The Bellman operator is a discrete-time backward sweep that propagates the value function leftwards through state space; the $\gamma$-contraction is a numerical stability condition equivalent to a CFL constraint on a discretised HJB equation. Continuous-time limit recovers the HJB PDE we'll meet in W12 (PINN capstone).

Keep these bridges live; W4 (PCA ≡ modes of a centred Gaussian, susceptibility ≡ variance) and W11 (RL Bellman ≡ Hamilton–Jacobi–Bellman) extend them.
