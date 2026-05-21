# Week 12 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — PINN residual loss on a 1-D heat-equation toy

PDE: $u_t = \nu u_{xx}$ on $x \in [-1, 1]$, $t \in [0, T]$, $\nu = 0.01$.

Network: $u_\theta(x, t)$ — tiny MLP, say 2 hidden layers of width 32.

### Residual at a single collocation point $(x, t) = (0, 0.5)$

Forward pass: $u_\theta(0, 0.5) = c$ (some scalar).

Autograd derivatives:

```python
import torch
x = torch.tensor(0.0, requires_grad=True)
t = torch.tensor(0.5, requires_grad=True)
u = model(x, t)
u_x  = torch.autograd.grad(u, x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
u_t  = torch.autograd.grad(u, t, create_graph=True)[0]
residual = u_t - nu * u_xx                 # scalar at one point
```

Note the **second derivative** $u_{xx}$ needs `create_graph=True` on
the first derivative — otherwise the second `autograd.grad` has
nothing to differentiate. This is the standard PINN pattern; PyTorch
makes it trivial.

### Full loss at $N$ collocation, $N_0$ IC, $N_b$ BC points

$$
\mathcal{L} = \lambda_r \cdot \tfrac{1}{N} \sum_{i} (u_t - \nu u_{xx})^2 + \lambda_0 \cdot \tfrac{1}{N_0} \sum_j (u_\theta(x_j, 0) - u_0(x_j))^2 + \lambda_b \cdot \tfrac{1}{N_b} \sum_k u_\theta(\pm L, t_k)^2.
$$

### Why fixed $(\lambda_r, \lambda_0, \lambda_b) = (1, 1, 1)$ fails

Residual points come from a 2-D set (interior); IC points from a 1-D
set; BC points from a 1-D set. Even at equal weights, the residual
loss has many more terms, dominates the gradient, and the network
ignores the boundary. Two common fixes:

1. **Hand-pick weights**: $\lambda_0, \lambda_b \gg \lambda_r$. Brittle.
2. **NTK-balanced** (Wang–Yu–Perdikaris 2022): every few steps,
   compute $\|\nabla_\theta \mathcal{L}_i\|$ for each loss term and
   set $\lambda_i \propto 1 / \|\nabla_\theta \mathcal{L}_i\|$ so each
   term contributes the same gradient magnitude. Phase E's
   `GradNormReweighter` is this idea.

The lesson generalises: any multi-term loss benefits from automatic
weighting tied to gradient norms.

---

## Example 2 — Walk-forward split with embargo on 20 samples

20 time-ordered samples, indices $0, 1, \dots, 19$. Labels are
**5-day forward returns** (so the label of sample $i$ depends on
prices through $i + 5$). Embargo $e = 2$.

### One fold

- **Train**: indices $[0, 10)$.
- **Validation**: indices $[10, 15)$.

### Purging

Any training sample whose label horizon overlaps the validation
window must be removed:

Validation window is $[10, 15)$. A label horizon ending in this range
means an index $i$ with $i + 5 \in [10, 15)$, i.e. $i \in [5, 10)$.
**Purge training indices $\{5, 6, 7, 8, 9\}$.**

Result: train = $[0, 5)$ (5 samples).

### Embargo

After the validation fold ends at $14$, drop the next $e = 2$ samples
$(15, 16)$ before they enter any future training fold. This prevents
leakage through autocorrelation — even if labels don't formally
overlap, residual autocorrelation can leak information back.

### Next fold

- Train: $[0, 12)$ — but purge $[7, 12)$ (label-horizon overlap with
  $[12, 17)$) and embargo $\{15, 16\}$ from prior fold.
- Net train: $[0, 7) \cup \{12, 13, 14\}$ (8 samples).
- Validation: $[17, 20)$ (skipping the embargo).

### Why this matters

Without purging, you'd train on samples whose labels "know" the
validation period — biased Sharpe inflated by 2–5×. Embargoing is
the second-order correction for residual autocorrelation.

---

## Example 3 — Selection-bias Sharpe inflation

Setup: $T = 252$ trading days of iid noise returns $r_t \sim
\mathcal{N}(0, 1)$. Generate $K = 500$ iid feature time-series.

### The trap

1. Compute in-sample correlation of each feature with $r$.
2. Pick the feature with highest |correlation|, say $\rho^\star \approx 0.20$.
3. Trade its sign — long when feature > 0, short when < 0.
4. Compute biased Sharpe on the same sample.

### Why $\rho^\star$ is large by chance

For $K$ independent features and $T$ samples, the *maximum* sample
correlation is roughly $\sqrt{2 \log K / T}$ even though the true
correlation is zero. At $K = 500, T = 252$: $\sqrt{2 \cdot 6.2 / 252}
\approx 0.22$. **The "20% correlation" is pure selection bias.**

### Biased vs honest Sharpe

Biased: simulate on the same sample, Sharpe ≈ $\rho^\star \cdot \sqrt{T} \approx 3.2$ annualised.

Honest: hold out another 252 days, compute Sharpe of the same feature on the new data — Sharpe ≈ 0 (because the true correlation was 0).

| Setting | Sharpe |
|---|---|
| Best of 500 features (biased) | $\sim 3.0$–$5.0$ |
| Same feature on held-out | $\sim 0$ |

### How to prevent

- **Embargo + walk-forward** for the parameter choice itself.
- **Bonferroni** or **Romano–Wolf** correction on the p-value.
- **Pre-registration** of the trading rule before seeing any data.
- The simplest defence: **never measure performance on data you used to choose hyperparameters**.

This is the same selection-bias pathology that powers irreplicable
academic results in psychology, biology, and economics. In quant
finance the consequences are real money.

---

## What to do with these examples

For Example 1, implement the PINN loss for the heat equation and
toggle `create_graph=True` off the first derivative — watch the second
derivative break with a friendly error. For Example 2, generate a
synthetic price series with autocorrelated residuals and compare
walk-forward Sharpe with and without embargo. For Example 3, vary $K$
(number of features) and observe how $\rho^\star \approx \sqrt{2 \log
K / T}$ holds across orders of magnitude.
