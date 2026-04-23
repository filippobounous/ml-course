# Week 3 — Theory-problem solutions

## 1. Logistic negative log-likelihood is convex

With $p_i = \sigma(x_i^\top \beta)$, NLL is $L(\beta) = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$. First derivative:

$\nabla_\beta L = \sum_i (p_i - y_i) x_i = X^\top (p - y)$.

Second derivative via chain rule ($\sigma'(z) = \sigma(z)(1-\sigma(z))$):

$\nabla^2_\beta L = X^\top W X,$ where $W = \operatorname{diag}(p_i(1-p_i))$.

Since $W \succeq 0$, $X^\top W X \succeq 0$ for any $X$ → $L$ is convex.

**IRLS step.** Newton: $\beta^+ = \beta - (X^\top W X)^{-1} X^\top (p-y)$. Reparametrise as a weighted least-squares problem: let $z = X\beta + W^{-1}(y-p)$. Then $\beta^+ = (X^\top W X)^{-1} X^\top W z$. At each iteration you solve a WLS problem with weights $W$ and response $z$ — hence "iteratively reweighted least squares".

## 2. Soft-margin SVM dual

Primal: $\min_{w,b,\xi} \tfrac12 \|w\|^2 + C \sum_i \xi_i$ s.t. $y_i(w^\top x_i + b) \ge 1 - \xi_i$, $\xi_i \ge 0$.

Lagrangian: $\mathcal{L} = \tfrac12 \|w\|^2 + C\sum_i \xi_i - \sum_i \alpha_i[y_i(w^\top x_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i$, with $\alpha_i, \mu_i \ge 0$.

Stationarity:

- $\partial_w: w = \sum_i \alpha_i y_i x_i$.
- $\partial_b: \sum_i \alpha_i y_i = 0$.
- $\partial_{\xi_i}: C - \alpha_i - \mu_i = 0 \Rightarrow \alpha_i \le C$.

Substitute back: $\mathcal{L}(\alpha) = \sum_i \alpha_i - \tfrac12 \sum_{ij} \alpha_i \alpha_j y_i y_j \langle x_i, x_j\rangle$, subject to $0 \le \alpha_i \le C$ and $\sum_i \alpha_i y_i = 0$.

**KKT complementary slackness:** $\alpha_i = 0 \Rightarrow y_i (w^\top x_i + b) \ge 1$; $0 < \alpha_i < C \Rightarrow$ on-margin; $\alpha_i = C \Rightarrow \xi_i > 0$ (inside margin or misclassified).

## 3. Info-gain vs Gini

For a split with $K$ children, parent class distribution $\mathbf{p}$, child distributions $\mathbf{p}_k$ and weights $w_k$:

$\text{IG}_\text{entropy} = H(\mathbf{p}) - \sum_k w_k H(\mathbf{p}_k)$,

$\text{IG}_\text{Gini}  = G(\mathbf{p}) - \sum_k w_k G(\mathbf{p}_k)$,

with $H(\mathbf{p}) = -\sum_c p_c \log p_c$ and $G(\mathbf{p}) = 1 - \sum_c p_c^2 = \sum_c p_c(1-p_c)$.

On 2-class problems, $H(p) = -p\log p - (1-p)\log(1-p)$ and $G(p) = 2p(1-p)$. Both are concave, both peak at $p=0.5$, both are zero at $p \in \{0,1\}$. They aren't monotone transforms of each other pointwise (the ratio $H/G$ varies with $p$), but they induce the **same ordering** on simple binary splits in practice. The choice rarely matters for predictive performance; XGBoost and random forests default to Gini or entropy interchangeably.
