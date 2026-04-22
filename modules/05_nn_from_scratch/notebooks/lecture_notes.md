# Week 5 — Neural networks from scratch (lecture notes)

*Reading pair: Goodfellow Ch.6 & 8 · Karpathy micrograd + *Yes you should understand backprop*.*

---

## 1. Computational graphs and reverse-mode autodiff

A neural network is a composition $f = f_L \circ \dots \circ f_1$ of differentiable operations organised as a **directed acyclic graph (DAG)**. At each node $v$ we store a value $\text{val}(v)$ and, after the backward pass, a gradient $\nabla v$.

Two orders to compute Jacobians:

- **Forward mode** computes $\partial v / \partial \theta_k$ one parameter at a time. Cost $\mathcal{O}(|\theta| \cdot |\text{graph}|)$.
- **Reverse mode** propagates $\partial L / \partial v$ backwards from the scalar loss. Cost $\mathcal{O}(|\text{graph}|)$ *per backward pass* — constant-cost-per-parameter, which is why it dominates DL.

Reverse mode = backpropagation. It is only $\mathcal{O}(\text{graph})$ because we stored intermediate values during the forward pass; space-for-time trade.

## 2. Chain rule, node by node

For a node $v$ with parents $u_1, \dots, u_n$ and local Jacobian $\partial v / \partial u_i$,

$$\frac{\partial L}{\partial u_i} \mathrel{+}= \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial u_i}.$$

The `+=` handles shared subexpressions (DAGs, not trees). In code, each operation defines:

- `forward(*parents) → child value`
- a **local backward** that, given $\partial L / \partial \text{child}$, increments each parent's gradient.

This is exactly what micrograd and (at a much larger scale) PyTorch do.

## 3. Common local gradients

Let $L$ be the final scalar loss; we always have $\partial L / \partial v$ available at node $v$.

- **Add**: $v = a + b \Rightarrow \nabla a += \nabla v$, $\nabla b += \nabla v$.
- **Multiply**: $v = a \cdot b \Rightarrow \nabla a += b \cdot \nabla v$, $\nabla b += a \cdot \nabla v$.
- **Power**: $v = a^k \Rightarrow \nabla a += k \cdot a^{k-1} \cdot \nabla v$.
- **Exp**: $v = e^a \Rightarrow \nabla a += v \cdot \nabla v$ (reusing the forward value).
- **Tanh**: $v = \tanh(a) \Rightarrow \nabla a += (1 - v^2) \cdot \nabla v$.
- **ReLU**: $v = \max(0, a) \Rightarrow \nabla a += \mathbf{1}[a > 0] \cdot \nabla v$.

## 4. Topological order matters

Before running backward we need a topological order so that when we process node $v$, every child of $v$ has already contributed to $\nabla v$. Standard recipe:

```python
def topo_sort(root):
    order, seen = [], set()
    def visit(v):
        if v in seen: return
        seen.add(v)
        for p in v._parents:
            visit(p)
        order.append(v)
    visit(root)
    return order
```

Then `order[::-1]` is a valid backward traversal.

## 5. Optimisers: SGD, momentum, Adam

**Vanilla SGD**. $\theta^{(t+1)} = \theta^{(t)} - \eta g_t$. Step size should be $\eta \le 1/L$ for stability on $L$-smooth problems.

**Momentum**. $v_t = \mu v_{t-1} + g_t$; $\theta \leftarrow \theta - \eta v_t$. Dampens oscillations along ill-conditioned directions. Nesterov: query the gradient at $\theta - \eta \mu v_{t-1}$ instead of $\theta$.

**Adam**. Per-parameter adaptive step with bias correction:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,$$

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta \leftarrow \theta - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

**AdamW** (Loshchilov & Hutter 2017) decouples weight decay from the Adam preconditioner — use this by default in PyTorch.

## 6. Initialisation (Glorot and He)

For a linear layer $y = W x$ with $W_{ij}$ i.i.d. with mean 0 and variance $\sigma_W^2$, $\text{Var}(y_i) = n_\text{in} \sigma_W^2 \text{Var}(x_j)$. Preserve variance:

- **Glorot / Xavier**: for a symmetric activation (tanh), use $\sigma_W^2 = 2 / (n_\text{in} + n_\text{out})$.
- **He / Kaiming**: for ReLU (which kills half the signal on average), use $\sigma_W^2 = 2 / n_\text{in}$.

Get initialisation wrong and you get vanishing or exploding activations — and gradients — on the very first forward pass.

## 7. Normalisation (BatchNorm, LayerNorm, RMSNorm)

**BatchNorm**. $\hat x_i = (x_i - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}$, then affine rescale $\gamma \hat x + \beta$. Stats are over the batch during training and over running averages at eval. Subtle pitfalls: train vs eval mode, distributed sync, small batches.

**LayerNorm** replaces batch statistics with per-token feature statistics — the right default inside transformers (Week 8).

**RMSNorm** drops the mean subtraction: $\hat x = x / \sqrt{\tfrac{1}{d} \sum_j x_j^2}$. Used in modern LLMs (Llama, Qwen) — faster and almost as good.

## 8. Where we are going

The artifact this week is a **micrograd-style** autograd engine you implement from scratch: scalar `Value` with the local gradients above, a topological-sort backward, and `Neuron / Layer / MLP` wrappers. In Week 6 we will then port everything to PyTorch and build a reusable `Trainer` — which is when your from-scratch understanding really pays off.

## What to do with these notes

Work the problem set in `../problems/README.md`. Build the micrograd engine in
`../../../portfolio/05_micrograd/` (reference implementation lives under
`src/mlcourse/autograd/`). Make sure your unit tests compare gradients both
to numerical gradients and to `torch.autograd` for at least six small graphs.
