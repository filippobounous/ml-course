# Week 11 — Theory-problem solutions

## 1. Bellman optimality operator is a γ-contraction

Define $(T^\star V)(s) = \max_a [r(s,a) + \gamma \sum_{s'} p(s'|s,a) V(s')]$.

For any two $V, W$ and any state $s$:

$|(T^\star V)(s) - (T^\star W)(s)|
 = \big|\max_a [r + \gamma \sum_{s'} p V(s')] - \max_a [r + \gamma \sum_{s'} p W(s')]\big|
 \le \max_a \big|\gamma \sum_{s'} p(s'|s,a) (V(s') - W(s'))\big|
 \le \gamma \max_a \sum_{s'} p(s'|s,a) |V(s') - W(s')|
 \le \gamma \|V - W\|_\infty$

using: (i) $|\max_a f - \max_a g| \le \max_a |f-g|$, (ii) triangle inequality, (iii) $p(\cdot|s,a)$ is a probability distribution. Hence $\|T^\star V - T^\star W\|_\infty \le \gamma \|V - W\|_\infty$.

Contraction + $\mathbb{R}^S$ complete → Banach → unique fixed point $V^\star$, and $V_k = T^\star V_{k-1}$ converges geometrically: $\|V_k - V^\star\|_\infty \le \gamma^k \|V_0 - V^\star\|_\infty$.

## 2. Policy gradient theorem

For a parameterised policy $\pi_\theta$ and objective $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t \ge 0} \gamma^t r_t]$ over trajectories $\tau = (s_0, a_0, r_0, s_1, \ldots)$:

$\nabla_\theta J = \nabla_\theta \int p_\theta(\tau) R(\tau) d\tau = \int \nabla_\theta p_\theta(\tau) \cdot R(\tau) d\tau$

(pushing $\nabla_\theta$ inside the integral). Log-derivative trick:

$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau).$

The trajectory likelihood factorises as $p_\theta(\tau) = p(s_0) \prod_t \pi_\theta(a_t|s_t) p(s_{t+1}|s_t,a_t)$. The environment terms don't depend on $\theta$, so $\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t)$.

Putting it together:

$\nabla_\theta J = \mathbb{E}_\tau\!\left[R(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right] = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t\right]$

where $G_t = \sum_{k\ge 0} \gamma^k r_{t+k}$ is the return-from-$t$ (causality: action $a_t$ cannot have affected rewards before $t$).

Replace $G_t$ by $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$ to reduce variance — valid because $\mathbb{E}[\nabla \log \pi \cdot V^\pi(s)] = 0$ (control-variate).

## 3. PPO clip geometry

Importance ratio $r(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$. Clipped surrogate:

$L^\text{CLIP} = \mathbb{E}[\min(rA, \text{clip}(r, 1-\varepsilon, 1+\varepsilon) A)].$

Analyse the two cases:

- **$A > 0$.** Want $r$ large. $\min(rA, \text{clip}(r,\cdot)A) = A \min(r, \max(1-\varepsilon, r)) = A \min(r, 1+\varepsilon)$ if $r > 1+\varepsilon$, else $rA$. So the gradient is zero once $r > 1 + \varepsilon$: no incentive to drive the ratio higher.
- **$A < 0$.** Want $r$ small. Symmetric argument gives zero gradient once $r < 1 - \varepsilon$.

The clip therefore creates a **pessimistic** bound: PPO never amplifies large-deviation improvements, preventing the runaway policy updates that plague vanilla policy gradients. Empirically this bounds KL$(\pi_\theta \| \pi_\text{old})$ without an explicit KL penalty.

## 4. GAE derivation

Define TD-residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Generalised Advantage Estimation:

$A_t^{(\lambda)} = \sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k}.$

At $\lambda = 0$: $A_t^{(0)} = \delta_t$ — pure one-step TD, low variance, potentially high bias if $V$ is wrong.

At $\lambda = 1$: $A_t^{(1)} = \sum_k \gamma^k \delta_{t+k}$; the telescoping sum of TDs equals the Monte-Carlo return minus $V(s_t)$ — unbiased (given correct $V$), high variance.

For $\lambda \in (0, 1)$ GAE interpolates. The recursion $A_t = \delta_t + \gamma\lambda A_{t+1}$ (zero at terminal states) is what our implementation computes in `compute_gae` / `compute_gae_torch`.

**Rule of thumb.** $\lambda \in [0.9, 0.97]$ dominates on most continuous-control benchmarks; $\lambda = 0$ occasionally wins when the value function is well-fit.
