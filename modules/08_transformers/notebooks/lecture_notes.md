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
