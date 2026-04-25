# Week 11 — Reinforcement learning and agents (lecture notes)

*Reading pair: Sutton & Barto Ch.3, 4, 6, 13 · Schulman *PPO* 2017 · Christiano *RLHF* 2017 · Yao *ReAct* 2022.*

---

## 1. The MDP framework

A (finite-horizon, discounted) Markov decision process is the tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$:

- $\mathcal{S}$: states.
- $\mathcal{A}$: actions.
- $p(s' | s, a)$: transition kernel.
- $r(s, a)$: expected immediate reward.
- $\gamma \in [0, 1)$: discount factor.

A policy $\pi(a | s)$ induces a trajectory distribution; the return is $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$, and the **state-value function** and **action-value function** are

$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s], \qquad Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a].$$

## 2. Bellman operators

The Bellman expectation operator $T^\pi$ and optimality operator $T^\star$ are

$$(T^\pi V)(s) = \mathbb{E}_{a \sim \pi, s' \sim p}[r(s, a) + \gamma V(s')],$$
$$(T^\star V)(s) = \max_a \mathbb{E}_{s'}[r(s, a) + \gamma V(s')].$$

Both are **$\gamma$-contractions** in the sup-norm $\|V\|_\infty = \max_s |V(s)|$: $\|T V - T W\|_\infty \le \gamma \|V - W\|_\infty$. By Banach's fixed-point theorem,

- value iteration $V_{k+1} = T^\star V_k$ converges to $V^\star$ geometrically,
- policy iteration (alternating policy evaluation + policy improvement) converges in finitely many steps on finite MDPs.

## 3. From tabular to function approximation

Tabular works beautifully up to a few thousand states. Past that we parameterise $V_\theta$ or $\pi_\theta$ by a neural net and *lose all the nice convergence guarantees*. The deep-RL literature is largely about managing the resulting instabilities (target networks, replay buffers, trust regions).

## 4. Policy gradients

### The policy-gradient theorem

For a parameterised policy $\pi_\theta$ and $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$,

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t \ge 0} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t\right].$$

Intuition: upweight the log-probability of actions that lead to high return. The identity $\nabla \log \pi = (\nabla \pi) / \pi$ (the "score function") is all the calculus.

### Variance reduction

The raw estimator has high variance. Standard tricks:

- **Baseline**. Subtract any $b(s)$ that doesn't depend on $a$: $\hat g_t = \nabla \log \pi(a_t | s_t)(G_t - b(s_t))$. Unbiased for any $b$. Using $b(s) = V^\pi(s)$ (advantage) is the standard choice.
- **Advantage Actor-Critic** (A2C / A3C). Learn both $\pi_\theta$ and $V_\phi$ concurrently; use $A_t = G_t - V_\phi(s_t)$ as the reduced-variance gradient weight.
- **GAE**(λ) (Schulman 2015). Blend TD and Monte-Carlo targets:
  $$A_t^\text{GAE}(\lambda) = \sum_{k \ge 0} (\gamma \lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$
  $\lambda = 0$ gives one-step TD; $\lambda = 1$ gives Monte-Carlo. $\lambda \in [0.9, 0.97]$ is the usual zone.

## 5. PPO

Policy-gradient methods can take destructively large steps. **Trust-region methods** fix this by constraining policy change; TRPO (Schulman 2015) imposes a hard KL constraint, which is effective but expensive.

**PPO** (Schulman 2017) replaces the constraint with a **clipped surrogate objective**:

$$L^\text{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) A_t\right)\right],$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_\text{old}}(a_t | s_t)$ is the importance ratio. The clip acts as a soft trust region — when the ratio drifts outside $[1 - \varepsilon, 1 + \varepsilon]$, the gradient becomes zero on the side that would push it further.

### PPO in three steps

1. Collect $N$ trajectories with the current policy.
2. Compute advantages via GAE.
3. Update $\pi_\theta$ and $V_\phi$ for $K$ epochs on the collected batch, using $L^\text{CLIP}$ plus a value-MSE and an entropy bonus.

Hyperparameters that matter: $\varepsilon \in \{0.1, 0.2\}$, $\gamma = 0.99$, $\lambda = 0.95$, 4–10 epochs per rollout, small batches (often 64–256 transitions).

*"The 37 Implementation Details of PPO"* (Huang et al. 2022) is essential reading — the gap between a working and a non-working PPO implementation is several small bookkeeping choices (observation normalisation, learning-rate annealing, advantage normalisation).

## 6. Agents from first principles

A classical ReAct-style agent (Yao 2022) is a loop:

```
observation := initial_state
while not done:
    thought, action := policy(observation, history)
    if action is final_answer:
        return action
    observation := tool(action)
    history.append((thought, action, observation))
```

For a minimal from-scratch implementation you need three pieces:

1. **Tools.** Callables that take a query, return an observation. Start with a calculator and a small retrieval tool.
2. **Policy.** An LLM (or a hand-written policy for unit tests) that takes the history and emits the next thought + action.
3. **Eval harness.** Deterministic grading: a fixed set of prompts with reference answers, exact-match or rubric grading, a failure taxonomy.

This sits at the boundary of RL and LLM research: the policy is usually learned via SFT / DPO on agent trajectories, not PPO-on-the-environment. But the scaffolding is the same.

## 7. RLHF in one paragraph

RLHF (Christiano 2017, Ouyang 2022) is PPO on a reward model trained from pairwise human preferences. It works but is painful (three models in memory, ad-hoc KL penalties). Modern practice mostly uses **DPO** (Week 9): skip the reward model entirely, optimise the preference likelihood directly. PPO-RLHF is still used at the top of the capability scale (GPT-4-class models), but for a course project DPO wins every time.

## What to do with these notes

Work the problem set in `../problems/README.md`. The portfolio artifact is in
`../../../portfolio/11_rl_agent/`: a CleanRL-style single-file PPO on a custom
1-D physics / market-making environment, plus a torch-free ReAct-style tool-use
agent with a deterministic eval harness.
