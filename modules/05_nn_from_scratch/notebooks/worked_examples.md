# Week 5 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — Backward pass on $f(x) = x \cdot x \cdot x$ by hand

Build the DAG:

```
     x ────┐
     │     ├─→ (x · x) ─┐
     │     ┘            ├─→ ((x · x) · x) = f
     └────────────────────┘
```

Two `Mul` nodes; same leaf $x$ feeds into all three slots. At $x = 2$:
$x \cdot x = 4$, then $4 \cdot x = 8$. So $f(2) = 8$. ✓

### Topological order

`[x, (x·x), f]` (children after parents). Backward order: reverse it.

### Backward pass

Seed: $\partial f / \partial f = 1$.

**Process $f = (x \cdot x) \cdot x$**:

- $\partial f / \partial (x \cdot x) \mathrel{+}= x \cdot 1 = 2$.
- $\partial f / \partial x \mathrel{+}= (x \cdot x) \cdot 1 = 4$.

**Process $(x \cdot x)$**:

Local op is $a \cdot b$ with $a = b = x$. Incoming upstream gradient is 2 from the step above.

- $\partial f / \partial x \mathrel{+}= x \cdot 2 = 4$ (the "left-x" of the multiply).
- $\partial f / \partial x \mathrel{+}= x \cdot 2 = 4$ (the "right-x").

**Total** at $x$: $4 + 4 + 4 = 12$. ✓ (Analytical: $\partial_x x^3 = 3 x^2 = 12$ at $x = 2$.)

### Lesson

The `+=` matters. If you wrote `=` instead, you'd lose the contribution from the second use of $x$ and get $\nabla x = 8$ — silently wrong. The "broadcast across all uses of a leaf" is the heart of reverse-mode autodiff on a DAG.

---

## Example 2 — Adam bias correction on a constant gradient

Suppose $g_t = 1$ at every step (constant gradient), and $\beta_1 = 0.9$, $m_0 = 0$.

| Step $t$ | $m_t = 0.9 \cdot m_{t-1} + 0.1$ | $1 - 0.9^t$ | $\hat m_t = m_t / (1 - 0.9^t)$ |
|---|---|---|---|
| 1 | 0.100 | 0.100 | 1.000 |
| 2 | 0.190 | 0.190 | 1.000 |
| 3 | 0.271 | 0.271 | 1.000 |
| 5 | 0.410 | 0.410 | 1.000 |
| 10 | 0.651 | 0.651 | 1.000 |
| 20 | 0.878 | 0.878 | 1.000 |

The **bias-corrected** $\hat m_t = 1$ at every step (since the true mean is $g = 1$). Without correction, the effective step size is $m_t / (\sqrt{v_t} + \epsilon)$ where $m_t$ starts near zero — the optimiser is dramatically *under-stepping* for the first $\sim 1/(1 - \beta_1) = 10$ steps.

This is why naive momentum-based optimisers often need a warmup schedule (linear ramp of the LR); Adam's bias correction is the principled fix that obviates it.

### Sanity check on $v_t$

Same arithmetic with $\beta_2 = 0.999$ and $g_t^2 = 1$:

$v_5 = 0.005$ (very small), $1 - 0.999^5 = 0.005$, $\hat v_5 = 1$. ✓

Without bias correction, $\sqrt{v_5} \approx 0.07$ → the adaptive denominator $\sqrt{\hat v_5} + \epsilon \approx 1 + \epsilon$ via correction, vs $0.07 + \epsilon$ without → step size 14× too large in the first few iterations. Both bias corrections matter.

---

## Example 3 — Glorot variance across a 5-layer linear net

Setup: input $x \in \mathbb{R}^{100}$, $\text{Var}(x_j) = 1$. Five linear layers of widths $100 \to 80 \to 60 \to 40 \to 20 \to 1$, all with tanh activations (linear regime for the variance analysis — tanh ≈ identity near zero).

### Naive init: $W_{ij} \sim \mathcal{N}(0, 1)$ (no scaling)

Variance after layer 1: $\text{Var}(y^{(1)}_i) = 100 \cdot 1 \cdot 1 = 100$.
After layer 2: $\text{Var}(y^{(2)}) = 80 \cdot 1 \cdot 100 = 8\,000$.
After layer 3: $\text{Var}(y^{(3)}) = 60 \cdot 1 \cdot 8\,000 = 480\,000$.

Activations **explode**. Tanh saturates → gradients vanish on the saturated portion → no learning. Disaster.

### Glorot: $\sigma_W^2 = 2 / (n_\text{in} + n_\text{out})$

Layer 1: $\sigma_W^2 = 2 / (100 + 80) = 1/90$. $\text{Var}(y^{(1)}_i) = 100 \cdot (1/90) \cdot 1 = 100/90 \approx 1.11$.
Layer 2: $\sigma_W^2 = 2 / (80 + 60) = 1/70$. $\text{Var}(y^{(2)}_i) = 80 \cdot (1/70) \cdot 1.11 \approx 1.27$.
Layer 3: $\sigma_W^2 = 2 / (60 + 40) = 1/50$. $\text{Var}(y^{(3)}_i) = 60 \cdot (1/50) \cdot 1.27 \approx 1.52$.
Layer 4: $\sigma_W^2 = 2 / (40 + 20) = 1/30$. $\text{Var}(y^{(4)}_i) \approx 2.03$.
Layer 5: $\sigma_W^2 = 2 / (20 + 1) = 2/21$. $\text{Var}(y^{(5)}_i) \approx 3.86$.

Variance drifts slightly because Glorot is the *average* of the forward and backward variance-preserving constants. Pure $\sigma_W^2 = 1/n_\text{in}$ would preserve variance exactly forward; pure $1/n_\text{out}$ would preserve it exactly backward. In practice the small drift is fine — much better than the $\times 100$ explosion of naive init.

### He for ReLU

ReLU kills half the signal on expectation, so the per-layer variance gets multiplied by $1/2$ each layer if you used Glorot. Compensate by using $\sigma_W^2 = 2/n_\text{in}$ instead — variance preserved forward through ReLU layers.

### Lesson

Init is not a hyperparameter; it's a **dynamical-isometry condition** (Pennington et al. 2017). Get it wrong by a factor of 2 across 50 layers and you've multiplied your activation variance by $2^{50}$.

---

## What to do with these examples

For Example 1, swap `*` for an `Exp` node and re-run the backward — you'll need to look up the $\partial_a e^a = e^a$ rule and reuse the forward value. For Example 2, plot $\hat m_t$ vs $m_t$ for $\beta_1 \in \{0.5, 0.9, 0.99\}$ — the warmup-bias gap is dramatic for $\beta_1 = 0.99$. For Example 3, simulate the variance propagation in NumPy with a small fixed seed and verify the calculation empirically (starter recipe: draw `W ~ Glorot`, push a standard-normal input through `L` linear+tanh layers, and print `x.var()` per layer — it should stay ≈ constant under correct Glorot and shrink/explode with the wrong fan).
