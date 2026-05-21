# Week 8 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — Scaled dot-product attention on 3 tokens, $d_k = 2$

$$
Q = K = V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}.
$$

(Three tokens, each represented as a 2-D vector — for ease of mental
arithmetic.)

### Step 1. Scores

$$
S = QK^\top / \sqrt{d_k} = \tfrac{1}{\sqrt 2}\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{pmatrix}.
$$

(Each row is one query's similarity to all three keys; diagonal terms
are self-similarity, off-diagonal are cross-similarity.)

### Step 2. Softmax row-wise

Row 1: $(\tfrac{1}{\sqrt 2}, 0, \tfrac{1}{\sqrt 2}) \approx (0.707, 0, 0.707)$. Softmax: $e^{0.707} \approx 2.028$, $e^0 = 1$, $e^{0.707} \approx 2.028$. Sum = 5.056.

$A_1 = (0.401, 0.198, 0.401)$.

By symmetry, row 2 also has weights $(0.198, 0.401, 0.401)$. Row 3:
$(0.707, 0.707, 1.414) \to e^x \approx (2.028, 2.028, 4.114)$, sum = 8.170. $A_3 \approx (0.248, 0.248, 0.504)$.

### Step 3. Output $O = AV$

Row 1: $0.401 \cdot (1, 0) + 0.198 \cdot (0, 1) + 0.401 \cdot (1, 1) = (0.802, 0.599)$.

The third token (at $(1, 1)$) had the highest similarity to *every* query, so it dominates each row of $O$ — a self-attention layer essentially "broadcasts" the most distinctive token. Same algebra as a softmax-weighted nearest-neighbour lookup, where $Q$ is the query, $K$ is the catalogue, $V$ is the payload.

### Take-home

Scaled dot-product attention is content-based addressable memory. The $1/\sqrt{d_k}$ keeps the softmax from saturating in high dimension — without it, for $d_k = 64$ the entropy of $A$ collapses and attention degenerates to argmax (a known failure mode).

---

## Example 2 — Causal mask on 4 tokens

Pre-softmax scores (we don't care about the actual values, just the shape):

$$
S = \begin{pmatrix} s_{11} & s_{12} & s_{13} & s_{14} \\ s_{21} & s_{22} & s_{23} & s_{24} \\ s_{31} & s_{32} & s_{33} & s_{34} \\ s_{41} & s_{42} & s_{43} & s_{44} \end{pmatrix}.
$$

Mask: $M_{ij} = 0$ for $j \le i$, $-\infty$ for $j > i$. So $S + M$ becomes

$$
S + M = \begin{pmatrix} s_{11} & -\infty & -\infty & -\infty \\ s_{21} & s_{22} & -\infty & -\infty \\ s_{31} & s_{32} & s_{33} & -\infty \\ s_{41} & s_{42} & s_{43} & s_{44} \end{pmatrix}.
$$

### After softmax

$e^{-\infty} = 0$, so upper-triangular entries vanish. Each row re-normalises over the surviving entries:

$$
A = \begin{pmatrix} 1 & 0 & 0 & 0 \\ a_{21} & a_{22} & 0 & 0 \\ a_{31} & a_{32} & a_{33} & 0 \\ a_{41} & a_{42} & a_{43} & a_{44} \end{pmatrix},
$$

with each row summing to 1. Then $O_i = \sum_j A_{ij} V_j$ has $O_i$ depending only on $V_1, \dots, V_i$. **Position $i$ never attends to the future.**

### Why the $-\infty$ pre-softmax, not zero post-softmax?

Naively zeroing $A_{ij} = 0$ for $j > i$ would leave rows that don't sum to 1 — your attention output would be scaled wrong. The mask-before-softmax pattern is correct because it lets softmax handle the re-normalisation.

---

## Example 3 — RoPE relative-position identity

Take $d = 2$, single rotation frequency $\theta$. RoPE rotates the
$(Q_t, K_s)$ pair by $t \theta$ and $s \theta$ respectively:

$$
\tilde Q_t = R_{t\theta} Q_t, \quad \tilde K_s = R_{s\theta} K_s, \quad R_\alpha = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}.
$$

### Inner product

$$
\tilde Q_t \cdot \tilde K_s = Q_t^\top R_{t\theta}^\top R_{s\theta} K_s = Q_t^\top R_{(s-t)\theta} K_s.
$$

The last equality uses $R_\alpha^\top = R_{-\alpha}$ and $R_\alpha R_\beta = R_{\alpha+\beta}$ (commutative for 2D rotations). **Depends only on $s - t$.**

### Numerical check

Take $Q = (1, 0)$, $K = (1, 0)$, $\theta = \pi / 4$, $t = 2$, $s = 5$.

$R_{2\theta} Q = (\cos(\pi/2), \sin(\pi/2)) = (0, 1)$.
$R_{5\theta} K = (\cos(5\pi/4), \sin(5\pi/4)) = (-\tfrac{1}{\sqrt 2}, -\tfrac{1}{\sqrt 2})$.

Inner product: $0 \cdot (-\tfrac{1}{\sqrt 2}) + 1 \cdot (-\tfrac{1}{\sqrt 2}) = -\tfrac{1}{\sqrt 2}$.

Now check the relative form: $Q^\top R_{(s-t)\theta} K = (1, 0) \cdot R_{3\pi/4} \cdot (1, 0)^\top = \cos(3\pi/4) = -\tfrac{1}{\sqrt 2}$. ✓

### Why this matters

Learned absolute embeddings train one vector per position; they can't generalise past the training sequence length. RoPE's score-vs-relative-offset property means the model implicitly handles longer contexts at inference time — the "context-length extrapolation" phenomenon that makes Llama 3 et al. work past their training cap.

---

## What to do with these examples

For Example 1, redo with $d_k = 64$ (without the $1/\sqrt{d_k}$ scaling) using NumPy random Gaussians for $Q$, $K$, $V$ — and watch the softmax become a one-hot vector. That's the failure mode the scale prevents. For Example 2, run a tiny GPT forward pass twice — once with the mask, once without — and notice the perplexity difference on autoregressive generation. For Example 3, extend to multi-dimensional $d_k$ by stacking 2×2 rotation blocks at log-spaced frequencies $\theta^{(i)} = 10000^{-2i/d_k}$ — that's the canonical RoPE setup.
