# Week 9 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — DPO loss on a single preference pair

Setup: one prompt $x$, chosen completion $y_w$, rejected $y_l$. After
running both completions through $\pi_\theta$ and $\pi_\text{ref}$, you
have summed log-probabilities (per token, summed across tokens):

| | $\log \pi_\theta(y \mid x)$ | $\log \pi_\text{ref}(y \mid x)$ |
|---|---|---|
| $y_w$ | $-12.5$ | $-13.0$ |
| $y_l$ | $-10.5$ | $-12.0$ |

Hyperparameter $\beta = 0.1$ (typical).

### Step 1. Log-ratios

$$
r_w = \log\!\frac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)} = -12.5 - (-13.0) = 0.5.
$$

$$
r_l = \log\!\frac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)} = -10.5 - (-12.0) = 1.5.
$$

The reference is *more sure* of $y_w$ vs $y_l$ than $\pi_\theta$ is.

### Step 2. Margin

$$
\Delta = \beta (r_w - r_l) = 0.1 \cdot (0.5 - 1.5) = -0.1.
$$

### Step 3. Loss

$$
\mathcal{L}_\text{DPO} = -\log \sigma(\Delta) = -\log \sigma(-0.1) = -\log(0.4750) \approx 0.7444.
$$

(Recall $\sigma(0) = 0.5$ so $-\log 0.5 \approx 0.693$ is the "indifferent"-loss anchor; anything *above* that means the model is *worse* than reference at preferring $y_w$.)

### Step 4. Gradient sign

$\partial \mathcal{L} / \partial r_w = -\beta \sigma(-\Delta) < 0$ (positive on the chosen-loss side ⟹ push $r_w$ up).

$\partial \mathcal{L} / \partial r_l = +\beta \sigma(-\Delta) > 0$ (push $r_l$ down).

The gradient *increases* the chosen-completion log-probability and *decreases* the rejected one — exactly what RLHF wants, with no reward model and no PPO.

### Sanity check

If $r_w = 1.5$, $r_l = 0.5$ (preference correctly captured), $\Delta = 0.1$, $\mathcal{L} = -\log \sigma(0.1) \approx 0.6444$ — below the 0.693 indifference anchor. The further the chosen log-ratio is above the rejected one, the lower the loss.

---

## Example 2 — LoRA parameter count for a 7B model

Suppose a 7B transformer has $L = 32$ layers, with 4 linear projections
per attention block (Q, K, V, O), all of shape $d \times d = 4096 \times 4096$.

### Full fine-tuning

Per projection: $d^2 = 4096^2 = 16{,}777{,}216$ params.
Per layer: $4 \cdot 16.78\text{M} = 67.1\text{M}$.
All layers: $32 \cdot 67.1\text{M} = 2.15\text{B}$ trainable.

(That's not "all of 7B" — FFN layers, embeddings, layernorms add the rest. But it's representative for the attention path.)

### LoRA at rank $r = 16$

Per projection: $r (d_\text{in} + d_\text{out}) = 16 \cdot (4096 + 4096) = 131{,}072$.
Per layer: $4 \cdot 131{,}072 = 524{,}288 \approx 0.5\text{M}$.
All layers: $32 \cdot 524{,}288 = 16.8\text{M}$ trainable.

### Reduction

$16.8\text{M} / 2.15\text{B} \approx 0.78\%$ — under 1% of the original
attention parameters. **And** the LoRA weights are mergeable at
deployment ($W' = W + BA$) so inference has zero overhead.

### Sanity check the $r = 16$ choice

Real-world: $r \in \{8, 16, 32, 64\}$ are common. Lower $r$ → fewer
trainable params, lower update capacity, faster training. The "right"
$r$ for a given task is usually picked by ablation on a held-out set
(DPO win-rate, downstream eval). Anthropic's published numbers suggest
$r = 32$–$64$ for hard tasks, $r = 8$–$16$ for style/format tuning.

---

## Example 3 — Chinchilla-optimal compute budget for three model sizes

Empirical rule: $D^\star \approx 20 N^\star$ at the compute-optimal point.
Total FLOPs $C \approx 6 N D$ (forward + backward + optimiser).

| Params $N$ | Optimal tokens $D$ | FLOPs $C$ | H100-hours @ 50% util |
|---|---|---|---|
| 1B   | 20B    | $1.2 \times 10^{20}$  | $\sim 67$ |
| 7B   | 140B   | $5.9 \times 10^{21}$  | $\sim 3{,}300$ |
| 70B  | 1.4T   | $5.9 \times 10^{23}$  | $\sim 330{,}000$ |

(Assuming $\sim 1 \times 10^{15}$ FLOPs/s at 50% utilisation for an H100 in bf16.)

### Read

- 1B is a *weekend project* on a single GPU (~$200 of rented compute).
- 7B is a small-team week (~$5k–$10k).
- 70B is industrial-scale (a million dollars or more).

### Where Chinchilla is wrong

Hoffmann et al.'s fit was for *pretraining* only. Modern Llama-3 / Qwen-2 train *past* the Chinchilla optimum (e.g. Llama-3-8B trained on 15T tokens vs the Chinchilla optimum of $\sim 160$B) because **inference cost dominates the total**: under-train slightly, deploy widely, recover the over-training cost via cheaper inference. The lesson is that "optimal" depends on the deployment regime.

### Connection to W9's tiny-scale reproduction

You won't pretrain a 7B model. But you can verify the **scaling exponents** by running a sweep of TinyLlama-scale models on TinyStories — losses scale as $L = A \cdot D^{-\alpha}$ with $\alpha \approx 0.34$ empirically. That's the calculation behind the slow-tier integration test in W9.

---

## What to do with these examples

For Example 1, modify so $\pi_\theta$ already strongly prefers $y_w$
($r_w = 5, r_l = -5$) and notice $\mathcal{L}_\text{DPO} \to 0$: the
model has nothing to learn from this pair. For Example 2, drop $r$ to
4 and recompute — sometimes adequate for style tuning. For Example 3,
use the OpenAI scaling-law fit ($\alpha \approx 0.5$ for the
loss-vs-compute exponent) and see how dramatically the compute budget
changes for the same target loss.
