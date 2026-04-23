# Week 8 — Theory-problem solutions

## 1. Attention gradient

Let $A = \text{softmax}(QK^\top / \sqrt{d_k})$, $O = AV$. With upstream gradient $\partial L / \partial O = G_O$:

- $\partial L / \partial V = A^\top G_O$.
- $\partial L / \partial A = G_O V^\top$.
- $\partial L / \partial S$ where $S_{ij} = Q_i \cdot K_j / \sqrt{d_k}$: softmax Jacobian gives
  $\partial L / \partial S_{ij} = A_{ij} \, [(\partial L / \partial A)_{ij} - \sum_k A_{ik}(\partial L/\partial A)_{ik}]$.
- $\partial L / \partial Q = (\partial L / \partial S) K / \sqrt{d_k}$.
- $\partial L / \partial K = (\partial L / \partial S)^\top Q / \sqrt{d_k}$.

This is the bread and butter of FlashAttention derivations; in practice torch's `scaled_dot_product_attention` computes it efficiently in a single fused kernel.

## 2. Softmax shift invariance

$\text{softmax}(z + c \mathbf{1})_i = e^{z_i + c} / \sum_j e^{z_j + c} = e^c e^{z_i} / (e^c \sum_j e^{z_j}) = \text{softmax}(z)_i$. The $e^c$ factors cancel. Numerical stability: subtract $\max_j z_j$ before exponentiating to keep $e^{z_i - \max}$ bounded in $[0, 1]$.

## 3. Causal mask correctness

Apply mask $M_{ij} = -\infty$ for $j > i$ before softmax. Then $A_{ij} = \text{softmax}(S + M)_{ij} = 0$ for $j > i$ (because $e^{-\infty} = 0$) and rows still sum to 1. So $O_i = \sum_j A_{ij} V_j = \sum_{j \le i} A_{ij} V_j$ — position $i$ depends only on positions $\le i$.

## 4. RoPE relative-position property

RoPE rotates pairs of dimensions by an angle proportional to position: $\tilde Q_t = R_{\theta_t} Q_t$, $\tilde K_s = R_{\theta_s} K_s$ where $R_\alpha$ is a block-diagonal of 2×2 rotations at frequencies $\theta^{(i)}$.

The inner product $\tilde Q_t \cdot \tilde K_s = Q_t^\top R_{\theta_t}^\top R_{\theta_s} K_s = Q_t^\top R_{\theta_s - \theta_t} K_s$, using $R^\top R = R_{-\theta}$ and $R_\alpha R_\beta = R_{\alpha+\beta}$ for 2×2 rotations. The inner product depends only on the *relative* offset $s - t$.

That's the magic: the attention score between positions $t$ and $s$ is a function of $s - t$, not of absolute positions. Exactly what you want for generalisation to longer contexts than the training distribution — RoPE extrapolates much better than learned absolute embeddings.

**Implementation.** See `apply_rope` in `modules/08_transformers/problems/solutions.py`; the unit test there verifies the relative-position identity numerically.
