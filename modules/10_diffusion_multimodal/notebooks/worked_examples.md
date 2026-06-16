# Week 10 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — Forward noise schedule on a 1-D toy

Linear schedule: $\beta_t \in [10^{-4}, 0.02]$ across $T = 1000$ steps.
$\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s \le t} \alpha_s$.

### A few steps

| $t$ | $\beta_t$ | $\alpha_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ | $\sqrt{1 - \bar\alpha_t}$ |
|---|---|---|---|---|---|
| 1 | $1.0 \times 10^{-4}$ | $0.9999$ | $0.9999$ | $0.99995$ | $0.0100$ |
| 100 | $\sim 0.002$ | $\sim 0.998$ | $\sim 0.905$ | $0.9513$ | $0.3082$ |
| 500 | $\sim 0.010$ | $\sim 0.990$ | $\sim 0.094$ | $0.3066$ | $0.9518$ |
| 1000 | $0.02$ | $0.98$ | $\sim 4.0 \times 10^{-5}$ | $0.00632$ | $0.99998$ |

*Entries below row 1 are rounded to the precision shown (the $\sim$ flags the compounding
$\bar\alpha_t$); recompute from the linear schedule if you need exact values.*

### Closed form check

$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$.

At $t = 100$ with $x_0 = 1.0$: mean $= 0.9513$, std $= 0.3082$. So a clean signal $x_0 = 1$ is, after 100 steps, indistinguishable in 1-sigma sense from $\mathcal{N}(0.95, 0.31^2)$ — most of the signal preserved.

At $t = 500$: mean $= 0.31$, std $= 0.95$. Signal is now mostly noise.

At $t = 1000$: mean $\approx 0.006$, std $\approx 1.0$. Essentially pure noise — what you want as the reverse-process starting condition.

### Sanity check

$\bar\alpha_t$ decays monotonically from $\sim 1$ to $\sim 0$ across $t = 1 \to T$. At any $t$ you can recover $x_0$ in expectation via
$\mathbb{E}[x_0 | x_t] = (x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon_\theta(x_t, t)) / \sqrt{\bar\alpha_t}$ — that's the **denoising-via-prediction** identity used at inference.

---

## Example 2 — DDIM $\eta = 0$ determinism

DDIM update: $x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\, \epsilon_\theta + \sigma_t z$.

At $\eta = 0$: $\sigma_t = 0$, the $z$-term vanishes, and the update is a deterministic function of $x_t$ via $\hat x_0$ and $\epsilon_\theta$.

### Numerical verification on a single sample

```python
import torch
torch.manual_seed(0)
x_T = torch.randn(1, 1, 28, 28)         # same starting noise

x_a = ddim_sample(model, x_T, schedule, steps=50, eta=0.0)
x_b = ddim_sample(model, x_T, schedule, steps=50, eta=0.0)
assert torch.equal(x_a, x_b)            # bit-identical
```

vs $\eta = 1.0$ (full DDPM, stochastic):

```python
x_c = ddim_sample(model, x_T, schedule, steps=50, eta=1.0)
x_d = ddim_sample(model, x_T, schedule, steps=50, eta=1.0)
assert not torch.equal(x_c, x_d)        # diverges (different RNG)
```

### Why $\eta = 0$ is the **probability-flow ODE**

The reverse SDE is $dx = [-\tfrac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)] dt + \sqrt{\beta(t)} d\bar W$. Dropping the noise term gives the **probability-flow ODE**

$$
\frac{dx}{dt} = -\tfrac{1}{2}\beta(t)x - \tfrac{1}{2}\beta(t)\nabla_x \log p_t(x),
$$

which has the same marginals $p_t(x)$ but is deterministic. DDIM with $\eta = 0$ is a 1st-order numerical integrator for this ODE — that's why **fewer steps suffice** (the discretisation–accuracy trade-off is an ODE-solver's, not a stochastic sampler's).

### Practical upshot

DDIM-50 at $\eta = 0$ often gives sample quality indistinguishable from DDPM-1000 — a 20× speedup at inference. This is what makes diffusion models deployable for image generation.

---

## Example 3 — Classifier-free guidance arithmetic

CFG samples from a "tempered" conditional density:

$$
\log \tilde p(x | y) = (1 + w) \log p(x | y) - w \log p(x).
$$

### Score → $\epsilon$

The score is $\nabla_x \log p(x | y)$ (or unconditional). DDPM trains $\epsilon_\theta$ such that $-\epsilon_\theta(x_t, y, t) / \sqrt{1 - \bar\alpha_t}$ is the score.

So the **tempered score** is

$$
\nabla_x \log \tilde p(x | y) = (1 + w) \nabla_x \log p(x | y) - w \nabla_x \log p(x).
$$

In $\epsilon$-space:

$$
\tilde \epsilon = (1 + w) \epsilon_\text{cond} - w \epsilon_\text{uncond}.
$$

(Both terms share the $-1/\sqrt{1 - \bar\alpha_t}$ factor — it cancels.)

### Numerical sanity

| $w$ | $\tilde \epsilon$ | Interpretation |
|---|---|---|
| 0 | $\epsilon_\text{cond}$ | pure conditional sampling |
| 1 | $2 \epsilon_\text{cond} - \epsilon_\text{uncond}$ | mild over-conditioning |
| 3 | $4 \epsilon_\text{cond} - 3 \epsilon_\text{uncond}$ | strong (typical for text-to-image) |
| $\infty$ | $\propto \epsilon_\text{cond} - \epsilon_\text{uncond}$ | hard projection onto the "conditional direction" |

### Why over-saturation at high $w$

Tempering by $(1 + w)$ on $\log p(x | y)$ peaks the distribution. Modes become sharper, off-mode probability mass evaporates. Concretely: very high $w$ on a Stable-Diffusion text-to-image system produces over-saturated, cartoonish outputs (the modal cat at $w = 30$ looks like an emoji), exactly what you'd expect from sampling at very low "temperature" $1/(1+w)$.

### Why $\epsilon_\text{uncond}$ is needed (not just larger $\epsilon_\text{cond}$)

Without the unconditional reference, "scaling up the conditional score" would just shift the magnitude — the geometry would be unchanged. Subtracting $w \epsilon_\text{uncond}$ is what **deletes the unconditional mass** from the conditional, leaving only the direction that's conditionally salient.

---

## What to do with these examples

For Example 1, plot $\sqrt{\bar\alpha_t}$ and $\sqrt{1 - \bar\alpha_t}$
across $t \in [1, 1000]$ — the cross-over point at $\bar\alpha_t = 0.5$
is around $t \approx 700$. That's the "signal-noise transition" of
your schedule; it sets where most of the loss comes from. For Example
2, run DDIM at $\eta = 0$ with `steps = 10` and the resulting sample
will be perceptibly worse than `steps = 50` but still recognisable —
that's the ODE-solver discretisation error showing up. For Example 3,
sweep $w \in \{0, 1, 3, 7, 15\}$ on a trained class-conditional DDPM
and watch the samples gradually over-saturate.
