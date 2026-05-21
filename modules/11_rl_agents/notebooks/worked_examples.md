# Week 11 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits on
one sheet of paper.

---

## Example 1 — Bellman contraction on a 3-state chain

Three states $\{A, B, C\}$, one action (so $T^\star = T^\pi$).
Rewards: $r_A = 1, r_B = 0, r_C = -1$. Deterministic transitions:
$A \to B$, $B \to C$, $C \to C$. Discount $\gamma = 0.9$.

### Start from $V^{(0)} = (0, 0, 0)$

$T V$ at each state:
- $V_A^{(1)} = r_A + \gamma V_B^{(0)} = 1 + 0.9 \cdot 0 = 1$.
- $V_B^{(1)} = 0 + 0.9 \cdot 0 = 0$.
- $V_C^{(1)} = -1 + 0.9 \cdot 0 = -1$.

$V^{(1)} = (1, 0, -1)$.

### Iterate

$V^{(2)} = (1 + 0.9 \cdot 0, 0 + 0.9 \cdot (-1), -1 + 0.9 \cdot (-1)) = (1, -0.9, -1.9)$.

$V^{(3)} = (1 - 0.81, -0.9 - 1.71, -1 - 1.71) = (0.19, -1.71, -2.71)$ — wait, let me redo: $V_A^{(3)} = 1 + 0.9 \cdot (-0.9) = 0.19$ ✓.

### Fixed point

$V^\star_C = -1 / (1 - 0.9) = -10$ (since C is absorbing with reward -1).
$V^\star_B = 0 + 0.9 \cdot (-10) = -9$.
$V^\star_A = 1 + 0.9 \cdot (-9) = -7.1$.

### Contraction check

$\|V^{(2)} - V^{(1)}\|_\infty = \max(|1-1|, |-0.9-0|, |-1.9-(-1)|) = 0.9$.

$\|V^{(3)} - V^{(2)}\|_\infty = \max(|0.19-1|, |-1.71-(-0.9)|, |-2.71-(-1.9)|) = 0.81 = 0.9^2$.

The infinity-norm gap shrinks by exactly $\gamma$ each step, as the proof predicts. After $k$ iterations, error is $\le \gamma^k \|V^{(0)} - V^\star\|_\infty = 0.9^k \cdot 10$. To get within $0.01$, you need $k \ge \log(0.001) / \log(0.9) \approx 65$ iterations.

---

## Example 2 — PPO clip geometry on one transition

Set $\varepsilon = 0.2$ (canonical), advantage $A = 1$ (positive — we want to increase the policy mass on this action).

### Three importance ratios

| $r = \pi_\theta / \pi_\text{old}$ | $r \cdot A$ | $\text{clip}(r, 0.8, 1.2) \cdot A$ | $L^\text{CLIP} = \min$ |
|---|---|---|---|
| 0.7 | 0.7 | 0.8 | **0.7** |
| 1.0 | 1.0 | 1.0 | 1.0 |
| 1.3 | 1.3 | 1.2 | **1.2** |
| 2.0 | 2.0 | 1.2 | **1.2** |

**Interpretation when $A > 0$:**
- $r < 1 - \varepsilon$: clip is *not* binding (the min picks $rA$). Gradient pushes $r$ up — correct, since we want more probability on a good action.
- $1 - \varepsilon \le r \le 1 + \varepsilon$: identity; standard policy gradient.
- $r > 1 + \varepsilon$: clip binds, gradient is **zero** (the min picks the clipped term, which doesn't depend on $\theta$). No incentive to push further.

The result is a **trust region**: $r$ can't profitably stray more than $\varepsilon$ from 1. Empirically this bounds $D_\text{KL}(\pi_\theta \| \pi_\text{old})$ without an explicit KL term.

### Symmetric case $A = -1$

| $r$ | $r \cdot A$ | $\text{clip}(r, 0.8, 1.2) \cdot A$ | $\min$ |
|---|---|---|---|
| 0.5 | -0.5 | -0.8 | **-0.8** |
| 0.9 | -0.9 | -0.9 | -0.9 |
| 1.1 | -1.1 | -1.1 | -1.1 |

For $A < 0$ we want $r$ small (less mass on a bad action). The clip
binds when $r < 1 - \varepsilon = 0.8$: the min selects the *more
pessimistic* $-0.8$ over $-0.5$, killing the gradient. **Asymmetric
optimism — pessimism — protects from runaway updates in both
directions.**

---

## Example 3 — GAE on a 4-step rollout

Setup: rollout of 4 transitions. TD residuals $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$:

$\delta_0 = 0.5, \delta_1 = 0.2, \delta_2 = -0.1, \delta_3 = 0.3$.

$\gamma = 0.99$.

### $\lambda = 0$: one-step TD

$A_t^{(0)} = \delta_t$:
$(A_0, A_1, A_2, A_3) = (0.5, 0.2, -0.1, 0.3)$.

Low variance (each estimate uses only one transition + the value
function). High bias if $V$ is wrong.

### $\lambda = 1$: Monte Carlo (telescoping)

$A_t^{(1)} = \sum_{k \ge 0} \gamma^k \delta_{t+k}$:

$A_3 = 0.3$.
$A_2 = -0.1 + 0.99 \cdot 0.3 = 0.197$.
$A_1 = 0.2 + 0.99 \cdot 0.197 = 0.395$.
$A_0 = 0.5 + 0.99 \cdot 0.395 = 0.891$.

Equals MC return minus $V(s_0)$ (when computed with the correct $V$).
Unbiased; high variance (uses the full sample trajectory).

### $\lambda = 0.95$: standard PPO setting

Recursion: $A_t = \delta_t + \gamma \lambda A_{t+1}$, with $A_T = 0$.

$A_3 = 0.3$.
$A_2 = -0.1 + 0.99 \cdot 0.95 \cdot 0.3 = -0.1 + 0.2822 = 0.1822$.
$A_1 = 0.2 + 0.9405 \cdot 0.1822 = 0.2 + 0.1714 = 0.3714$.
$A_0 = 0.5 + 0.9405 \cdot 0.3714 = 0.5 + 0.3493 = 0.8493$.

**Interpolation visible:** $A_0$ at $\lambda = 0.95$ ($0.849$) sits
between the TD value ($0.5$) and the MC value ($0.891$), closer to MC
as expected for high $\lambda$.

### Why $\lambda \in [0.9, 0.97]$

At low $\lambda$ you over-trust the value function; at $\lambda = 1$
you use unbiased but high-variance MC. The sweet spot is empirically
$0.9$–$0.97$ on most continuous-control tasks. CartPole / LunarLander
fit comfortably in this range.

---

## What to do with these examples

For Example 1, replace state $C$'s self-loop with $C \to A$ — value
iteration now must propagate the $-1$ reward backwards into a positive
cycle, watch convergence get harder. For Example 2, sweep
$\varepsilon \in \{0.05, 0.2, 0.5\}$ and observe how the trust-region
width affects training stability (too tight: slow; too wide: unstable
— PPO at $\varepsilon = 0.5$ behaves like vanilla policy gradient).
For Example 3, generate a synthetic rollout in code with random
$\delta_t$ and compare your hand-computed GAE to `compute_gae` in
`portfolio/11_rl_agent/ppo.py`.
