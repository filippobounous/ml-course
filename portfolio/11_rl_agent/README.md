# 11 — Custom-env PPO + from-scratch tool-use agent

> Populated in Week 11. See `modules/11_rl_agents/`.

## Problem
Two-part artifact: (A) a PPO implementation from scratch applied to a
**custom environment** (not just stock CartPole) and (B) a small from-scratch
tool-use agent with a deterministic evaluation harness.

## Method
- Part A: CleanRL-style single-file PPO on a custom gymnasium env (simple
  market-maker or 1-D physics sim).
- Part B: ReAct-style loop with two tools (Python eval + retrieval), 20
  curated eval tasks, fixed seed, exact-match + rubric grading.

## Results
*PPO training curve on the custom env; agent task-success rate table.*

## Reproduce
```bash
make -C portfolio/11_rl_agent reproduce
```

## Why this matters
Goes beyond the stock CartPole artifact most candidates ship — highlights
ability to design environments and evaluation loops.
