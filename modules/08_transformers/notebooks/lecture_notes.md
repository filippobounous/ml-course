# Week 8 — Transformers from scratch (lecture notes)

*Reading pair: Vaswani *Attention Is All You Need* · Karpathy *Let's build GPT* · Radford *GPT-2* · Sennrich *BPE*.*

---

## 1. Why attention

RNNs compress the entire history into a fixed-size hidden state; CNNs extend their receptive field only linearly in depth. **Attention** lets a position look directly at any other position in parallel.

For a sequence $X \in \mathbb{R}^{T \times d}$,

- **Queries** $Q = X W_Q \in \mathbb{R}^{T \times d_k}$.
- **Keys**    $K = X W_K \in \mathbb{R}^{T \times d_k}$.
- **Values**  $V = X W_V \in \mathbb{R}^{T \times d_v}$.

Scaled dot-product attention:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

The $1 / \sqrt{d_k}$ factor keeps the pre-softmax logits' variance at $\mathcal{O}(1)$ as $d_k$ grows — otherwise softmax saturates and gradients vanish.

## 2. Multi-head attention

Run $h$ attention heads in parallel, each with its own $W_Q, W_K, W_V \in \mathbb{R}^{d \times d/h}$, concatenate outputs, project back to $d$:

$$\text{MHA}(X) = \text{Concat}(\text{Attn}_1, \dots, \text{Attn}_h) W_O.$$

Each head has low-rank $d/h$-dimensional projections; the concat+project recombines them. The total parameter count is the same as a single head at full width.

## 3. Causal masking

For an autoregressive model $p(x_1, \dots, x_T) = \prod_t p(x_t | x_{<t})$, token $t$ must not attend to tokens $\ge t$. Enforce this by adding $-\infty$ to the softmax logits at positions $j \ge t$:

$$\text{logits}_{t, j} \leftarrow \text{logits}_{t, j} + M_{t, j}, \quad M_{t, j} = \begin{cases} 0 & j \le t \\ -\infty & j > t \end{cases}.$$

After softmax the forbidden positions have zero weight.

## 4. Positional encodings

Attention is permutation-equivariant — on its own it has no sense of order. Fix with positional encodings.

- **Sinusoidal** (Vaswani 2017). $\text{PE}_{t, 2i} = \sin(t / 10000^{2i/d})$, $\text{PE}_{t, 2i+1} = \cos(\dots)$. Deterministic; generalises loosely to longer contexts.
- **Learned**. $\text{PE}_t$ is a trainable vector per position. Simple; doesn't generalise past the training context.
- **RoPE** (Su 2021). Rotate $Q$ and $K$ by position-dependent 2-D rotations; the inner product $Q_t \cdot K_s$ then depends only on the relative offset $t - s$. Used by Llama, Qwen, etc.
- **ALiBi** (Press 2022). Subtract a linear bias proportional to $|t - s|$ from the logits. No embedding; trivial to extrapolate.

## 5. Tokenisation: byte-pair encoding (BPE)

Character-level is slow; word-level has huge OOV problems. BPE splits the difference:

1. Start with a byte-level vocabulary (256 entries).
2. Iteratively merge the most frequent adjacent pair into a new token.
3. Stop at a target vocabulary size (GPT-2: ≈ 50k).

The `tokenizers` library (HF) gives you a fast Rust implementation. For TinyStories (~350 MB) a 10k-vocab BPE trains in under a minute on a laptop.

**Gotcha.** `encode(decode(x))` is not always an identity if your tokenizer normalises (lowercasing, NFC). For reversibility, use a byte-level BPE without normalisation — exactly what GPT-2 and nanoGPT do.

## 6. The decoder-only block

A standard pre-LN GPT block:

```
x ← x + MHA(LN(x))
x ← x + FFN(LN(x))
```

with `FFN(z) = GELU(z W_1) W_2` and widths $d \to 4d \to d$. Pre-LN (LayerNorm before attention/FFN, not after) trains more stably at depth than Vaswani's original post-LN.

Full decoder-only transformer: embedding → positional encoding → $N$ blocks → final LN → linear head. Tying the embedding weights to the output head (`lm_head.weight = embedding.weight`) saves parameters and helps generalisation.

## 7. Training and generation

### Training
- Autoregressive loss: shift inputs right by one, cross-entropy over the next-token logits.
- AdamW with β₁ = 0.9, β₂ = 0.95, weight decay 0.1 (the GPT-2 recipe).
- Warmup + cosine LR schedule.
- Gradient clipping at 1.0 is almost free insurance.

### Generation
- **Greedy**: pick the argmax. Deterministic but dull.
- **Temperature**: divide logits by $T$ before softmax. Lower $T$ → sharper.
- **Top-k / nucleus**: keep only the top $k$ or top-$p$ probability mass, renormalise, sample.
- **Beam search**: reasonable for short-answer tasks, disastrous for open-ended generation (mode-seeking).

## 8. Scaling and attention maps

Per-layer **attention maps** — the softmax(QK^T / √d) weights — are interpretable(ish) for small models. In Week 8 we will plot them on generated continuations and look for heads tracking syntactic structure. At scale, mechanistic-interpretability research has identified induction heads, bigrams, and more (Elhage et al. 2021).

Scaling laws (Kaplan 2020, Chinchilla 2022) predict that at fixed compute $C$ the optimal (params, tokens) pair has them growing roughly together; see Week 9 for the details and the tiny-scale reproductions we can actually run on a laptop.

## 9. Capstone kickoff

By Friday of Week 8, draft a one-page capstone proposal in `capstone/proposal.md`:

- Goal (one sentence).
- Dataset (link + size).
- Primary metric (one number).
- Compute budget (hours on MPS / CPU).
- Risks and mitigations.

The capstone then runs in parallel with Weeks 9–12.

## What to do with these notes

Work the problem set in `../problems/README.md`. Build the tiny-GPT artifact
in `../../../portfolio/08_tinygpt/` — multi-head attention from scratch, BPE
tokenizer, ~10M-param transformer trained on TinyStories to a coherent-text
regime.

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three paper-doable exercises (scaled dot-product attention on 3 tokens with $d_k = 2$, causal mask on 4 tokens, RoPE relative-position identity with numerical check).

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1 Attention | 4 | Derive $\partial L / \partial Q, K, V$; compute one attention layer by hand on a 3-token toy. |
| §2 Multi-head + masking | 3 | Multi-head as multiple parallel heads on split channels; causal-mask correctness proof. |
| §3 Position encodings | 2 | Compare absolute / relative / RoPE; prove the RoPE identity. |
| §4 BPE tokenization | 3 | Train a BPE tokenizer with `tokenizers`; round-trip test. |
| §5 Tiny GPT training | 6 | Pre-LN + GELU + RoPE + weight tying; train on TinyStories to loss $\lesssim 2.0$. |
| Problem set + viz | 1 | Attention-map plot of 3 prompts; identify syntactic heads. |
| Office hours / review | 1 | Cross-check against `problems/solutions_theory.md`. |

## Self-assessment rubric

Before moving to Week 9, you should be able to answer "yes" to all of:

1. Can I derive $\partial L / \partial Q, \partial L / \partial K, \partial L / \partial V$ for scaled dot-product attention from the softmax + matmul chain rule?
2. Can I prove softmax shift invariance and explain why the max-subtraction trick is what makes a softmax implementation numerically stable?
3. Can I prove that causal-masked attention at position $t$ depends only on positions $\le t$ — and explain why the $-\infty$-before-softmax pattern is correct while $0$-after-softmax would be wrong?
4. Can I state RoPE mathematically and prove the inner product of two RoPE-rotated vectors depends only on the *relative* position $s - t$?
5. Can I train a ~10M-parameter GPT on TinyStories to loss $\lesssim 2.0$ and generate samples coherent enough to identify subject–verb agreement and closing punctuation?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **Attention ↔ pair-correlation / two-body interaction.** The matrix $S_{ij} = Q_i \cdot K_j / \sqrt{d_k}$ is a "potential" between token-$i$ and token-$j$; the softmax over $j$ turns it into a Boltzmann weight $A_{ij} \propto e^{-(-S_{ij})}$ — a Gibbs measure over keys at inverse-temperature $1$. The full attention output is then the **expected $V$ under this measure**, exactly the structure of an ensemble average over a two-body system.
- **Softmax over keys ↔ partition function.** The normalising sum $\sum_j e^{S_{ij}}$ is the partition function for the $i$-th query; $\log \sum_j e^{S_{ij}}$ is the **free energy**. The $1/\sqrt{d_k}$ scale plays the role of temperature: in the high-temperature limit attention becomes uniform; in the low-temperature limit it becomes a hard argmax (the "attention collapse" failure mode).
- **Causal mask ↔ retarded propagator.** Forbidding $j > i$ in $A_{ij}$ is the discrete-time analogue of imposing the retarded boundary condition $G^R(t, t') = 0$ for $t < t'$ — causality on the time arrow, with the mask playing the role of the Heaviside theta function.
- **RoPE ↔ Galilean / translation invariance in the embedding manifold.** Encoding position by a rotation makes the dot-product $\tilde Q_t \cdot \tilde K_s$ depend only on $s - t$, exactly the relative-position structure of a translation-invariant system. Same trick as the Bloch theorem in solid-state: physical observables in a periodic lattice depend on $k$, not on absolute position. RoPE's frequency ladder $\theta^{(i)} = 10000^{-2i/d_k}$ is a Fourier basis on the embedding manifold.
- **Multi-head attention ↔ multiple irreducible representations.** Each head sees a different $d_h$-dimensional projection of $Q$, $K$, $V$; they're concatenated and re-mixed. Think of each head as a separate **irrep channel** of a global symmetry — the model can simultaneously attend on syntactic, semantic, and positional features without forcing them into one shared subspace.

Keep these bridges live; W9 (DPO ≡ contrastive log-likelihood update) and W10 (diffusion ≡ reverse-time SDE) reuse the Gibbs / free-energy lens.
