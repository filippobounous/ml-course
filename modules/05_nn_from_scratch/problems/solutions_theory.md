# Week 5 — Theory-problem solutions

## 1. Backprop for a 2-layer MLP with tanh + softmax + cross-entropy

Forward pass on a single example $x \in \mathbb{R}^{d_0}$, class label $y \in \{0,\dots,K-1\}$:

- $z^{(1)} = W^{(1)} x + b^{(1)}, \quad h = \tanh(z^{(1)})$
- $z^{(2)} = W^{(2)} h + b^{(2)}, \quad p = \operatorname{softmax}(z^{(2)})$
- $\ell = -\log p_y$.

Backward pass, starting from $\partial \ell / \partial z^{(2)}$:

$\partial \ell / \partial z^{(2)} = p - e_y$ (standard softmax+CE gradient identity).

Then:

- $\partial \ell / \partial W^{(2)} = (p - e_y) h^\top$
- $\partial \ell / \partial b^{(2)} = p - e_y$
- $\partial \ell / \partial h = W^{(2)\top}(p - e_y)$
- $\partial \ell / \partial z^{(1)} = (1 - h^2) \odot (\partial \ell / \partial h)$ since $\tanh'(z) = 1 - \tanh^2(z)$.
- $\partial \ell / \partial W^{(1)} = (\partial \ell / \partial z^{(1)}) x^\top$
- $\partial \ell / \partial b^{(1)} = \partial \ell / \partial z^{(1)}$.

On a batch of $N$ examples, each gradient is the mean over examples.

## 2. Glorot preserves variance (linear activations)

Consider a linear layer $y = Wx + b$, $W_{ij}$ iid with mean 0 and variance $\sigma_W^2$, $x$ iid with variance $\operatorname{Var}(x)$. Ignoring $b$ (mean 0 too):

$\operatorname{Var}(y_i) = \operatorname{Var}\!\left(\sum_j W_{ij} x_j\right) = \sum_j \operatorname{Var}(W_{ij} x_j) = n_\text{in} \sigma_W^2 \operatorname{Var}(x).$

Preserve variance → $\sigma_W^2 = 1 / n_\text{in}$.

For the backward pass, gradient w.r.t. $x$ is $W^\top \partial\ell/\partial y$, same argument gives $\sigma_W^2 = 1 / n_\text{out}$.

Glorot's compromise: average of the two — $\sigma_W^2 = 2 / (n_\text{in} + n_\text{out})$. He's variant for ReLU (which kills half the signal on average, multiplying $\operatorname{Var}(y)$ by $1/2$): $\sigma_W^2 = 2 / n_\text{in}$.

## 3. Adam bias correction

Update rules: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, $m_0 = v_0 = 0$.

Expand: $m_t = (1-\beta_1) \sum_{s=1}^t \beta_1^{t-s} g_s$. If $g_s$ were iid draws from a stationary distribution with mean $\mathbb{E}[g]$:

$\mathbb{E}[m_t] = (1-\beta_1) \sum_{s=1}^t \beta_1^{t-s} \mathbb{E}[g] = (1 - \beta_1^t) \mathbb{E}[g].$

So $\mathbb{E}[m_t / (1 - \beta_1^t)] = \mathbb{E}[g]$: the correction removes the initial "warmup" bias caused by $m_0 = 0$.

Same reasoning for $v_t$: divide by $1 - \beta_2^t$. Effect is most pronounced in the first $~1/(1-\beta)$ steps; beyond that, corrections tend to 1.

**Why it matters in practice.** Without bias correction, the effective learning rate is tiny for the first hundred steps or so, and training stalls. Warmup schedules can paper over the problem, but bias correction is the principled fix.
