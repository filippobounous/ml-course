# Week 11 — Reinforcement learning and agents

## Learning objectives

1. State **MDPs** and Bellman operators; derive **value iteration** and **policy iteration** in the tabular case.
2. Derive the **policy gradient theorem**; implement REINFORCE, actor–critic, and **PPO** from scratch.
3. Train PPO on a **custom environment** (simple market-maker or 1-D physics sim), not just stock CartPole.
4. Build a minimal **tool-use agent loop** (observation → thought → action → observation) from first principles, with a deterministic evaluation harness.

## Topics

- MDPs, returns, discounting, state / state-action value functions.
- Value iteration, policy iteration, Q-learning (brief).
- Policy gradients: score function estimator, variance reduction, GAE.
- PPO: clipped surrogate objective, importance sampling, geometry.
- Agents: ReAct-style tool-use loops, termination, eval harnesses, prompt leakage pitfalls.

## Deliverables

- Portfolio artifact: `portfolio/11_rl_agent/` — **custom-env PPO** + **from-scratch tool-use agent** with eval harness. Designed to stand out vs stock CartPole.

## Reading plan

See `readings.md`.
