# Problem set — Week 11

## Theory

1. **Bellman contraction.** Prove that the Bellman optimality operator is a $\gamma$-contraction in the sup norm. Conclude value iteration converges.
2. **Policy gradient theorem.** Derive $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) A^{\pi_\theta}(s_t, a_t)]$ starting from $J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t]$.
3. **PPO clip.** Analyse the geometry of PPO's clipped objective. Show exactly when the clip is active and why that tends to bound policy KL.
4. **GAE.** Derive the GAE$(\lambda)$ advantage estimator and show its bias–variance interpolation between TD and Monte Carlo.

## Implementation (portfolio)

5. Implement **PPO from scratch** (single-file, CleanRL-style). Train on CartPole-v1 to convergence, then on Pendulum-v1 or LunarLanderContinuous-v2.
6. Build a **custom gymnasium environment** — either a simple market-maker (inventory + spread) or a 1-D physics sim (cart, harmonic oscillator control). Train PPO; produce training curves.

## Applied (portfolio)

7. Build a **from-scratch tool-use agent** (ReAct loop) with at most two tools (e.g. Python eval + calculator, or retrieval over a 100-document corpus). Ship a deterministic **eval harness**: 20 curated tasks with reference answers; compute success rate under a fixed seed.

## Grading

Tests in `tests/week_11/` check: Bellman update on a tiny tabular MDP matches the closed form; PPO loss with zero advantage is zero; the agent harness reports the right schema.
