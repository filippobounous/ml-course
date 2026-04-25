# Week 2 — Theory-problem solutions

## 1. Bias–variance decomposition (squared loss)

Let $h_\mathcal{S}(x)$ be the predictor trained on a random dataset $\mathcal{S}$, and $y = f(x) + \varepsilon$, $\mathbb{E}\varepsilon = 0$, $\operatorname{Var}\varepsilon = \sigma^2$, $\varepsilon \perp \mathcal{S}$.

$\mathbb{E}_\mathcal{S,\varepsilon}[(h_\mathcal{S}(x) - y)^2]$ — add and subtract $\bar h(x) := \mathbb{E}_\mathcal{S}[h_\mathcal{S}(x)]$ and $f(x)$:

$(h_\mathcal{S}(x) - y)^2 = (h_\mathcal{S}(x) - \bar h(x))^2 + (\bar h(x) - f(x))^2 + \varepsilon^2$ + 2(cross-terms).

Taking expectations, the cross-terms vanish because (i) $\mathbb{E}_\mathcal{S}[h_\mathcal{S}(x) - \bar h(x)] = 0$, (ii) $\mathbb{E}[\varepsilon] = 0$, (iii) $\varepsilon \perp \mathcal{S}$. Survivors:

$\mathbb{E}[\cdot] = \underbrace{\operatorname{Var}_\mathcal{S}(h_\mathcal{S}(x))}_\text{variance} + \underbrace{(\bar h(x) - f(x))^2}_{\text{bias}^2} + \underbrace{\sigma^2}_\text{irreducible}.$

For ridge, $\bar\beta_\lambda = (X^\top X + \lambda I)^{-1} X^\top X \beta^\star$ shrinks toward 0 as $\lambda$ grows → bias$^2$ grows. The variance term $\operatorname{Var}(\hat\beta_\lambda) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1}$ shrinks as $\lambda$ grows. The optimal $\lambda$ balances the two.

## 2. Ridge ≡ MAP

Likelihood $p(y | X, \beta) = \mathcal{N}(X\beta, \sigma^2 I)$. Prior $p(\beta) = \mathcal{N}(0, \tau^2 I)$. Posterior is proportional to

$\exp\!\left(-\tfrac{1}{2\sigma^2}\|y - X\beta\|^2 - \tfrac{1}{2\tau^2}\|\beta\|^2\right).$

Taking $-\log$ and dropping constants gives $\tfrac{1}{2\sigma^2}\|y-X\beta\|^2 + \tfrac{1}{2\tau^2}\|\beta\|^2$. This is $\tfrac{1}{2\sigma^2}\big(\|y-X\beta\|^2 + \lambda \|\beta\|^2\big)$ with $\lambda = \sigma^2/\tau^2$. The MAP is the minimiser, i.e. the ridge estimator.

## 3. Closed-form OLS

Minimising $\|y - X\beta\|^2$: $\nabla_\beta = -2X^\top(y - X\beta) = 0 \Rightarrow \hat\beta = (X^\top X)^{-1} X^\top y$ when $X$ has full column rank.

**Unbiasedness.** $\mathbb{E}[\hat\beta] = (X^\top X)^{-1} X^\top \mathbb{E}[y] = (X^\top X)^{-1} X^\top X \beta = \beta$.

**Covariance.** $\operatorname{Var}(\hat\beta) = (X^\top X)^{-1} X^\top \operatorname{Var}(y) X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$ using $\operatorname{Var}(y) = \sigma^2 I$.

## 4. LOO-CV bias

Denote $\mathcal{D}$ of size $N$. LOO estimator: $\hat R_\text{LOO} = \tfrac{1}{N}\sum_i \ell(h_{\mathcal{D}\setminus i}(x_i), y_i)$. Each term is an unbiased estimator of $\mathbb{E}_{\mathcal{D}'}[R(h_{\mathcal{D}'})]$ where $|\mathcal{D}'| = N-1$ (i.e. risk of predictors trained on $N-1$ samples). So $\mathbb{E}[\hat R_\text{LOO}] = \mathbb{E}[R(h_{\mathcal{D}_{N-1}})]$, not $\mathbb{E}[R(h_{\mathcal{D}_N})]$. For stable learners (where predictor changes little when one sample is removed) the gap is $\mathcal{O}(1/N)$, i.e. approximately unbiased for $N$ large; more formally this is Kearns–Ron-style stability bounds. Variance is high because the $N$ leave-one-out predictors are highly correlated.

## MDP primer

A tabular optimal-policy result for free: write the Bellman operator $T^\star V(s) = \max_a [r(s,a) + \gamma \sum_{s'} p(s'|s,a) V(s')]$. Contraction: $\|T^\star V - T^\star W\|_\infty = \max_s |\max_a(\dots V) - \max_a(\dots W)| \le \max_s \max_a |\gamma \sum_{s'} p(s'|s,a)(V(s') - W(s'))| \le \gamma \|V - W\|_\infty$. Contraction + completeness → Banach → unique fixed point $V^\star$, and value iteration $V_{k+1} = T^\star V_k$ converges geometrically.
