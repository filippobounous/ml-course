# 11 — Custom-env PPO + from-scratch tool-use agent

Two artifacts in one folder.

## Part A — PPO on a custom market-making environment

- `market_env.py` — `SimpleMarketMakerEnv` (gymnasium `Env`): choose a bid-ask
  spread level, earn PnL on hits, pay a quadratic inventory penalty.
  Deliberately simple but non-trivial (reward has intertemporal structure
  and a clear exploration-exploitation tension).
- `ppo.py` — CleanRL-style single-file PPO: 3-layer actor-critic MLP, GAE
  advantages, clipped surrogate objective, entropy bonus, gradient clipping.
- `train_ppo.py` — end-to-end training driver.

### Reproduce

```bash
python -m pip install -e ".[dl,rl,ops]"
python portfolio/11_rl_agent/train_ppo.py --quick      # 4k steps smoke
python portfolio/11_rl_agent/train_ppo.py              # 200k steps (~30 min MPS)
```

Target: mean episode return crosses zero within ~100k steps, climbs to a
consistent +10 to +30 by 200k. A no-quote baseline earns exactly 0 per step.

## Part B — From-scratch ReAct-style agent + eval harness

- `agent.py` — torch-free:
  - `calculator` tool (AST-restricted arithmetic),
  - `build_retriever(corpus)` tool (keyword-score lookup),
  - `run_agent(prompt, policy, config)` — ReAct loop,
  - `evaluate_agent(tasks, policy, config)` — deterministic grading,
  - a handful of hand-written policies (`make_math_and_lookup_policy`,
    `make_numeric_policy`, `make_fixed_trace_policy`) used by the tests.

### Reproduce (offline, deterministic)

```bash
python portfolio/11_rl_agent/agent.py
```

Prints a success-rate summary for 4 canned tasks. Drop in an LLM-backed
policy (Claude / GPT / local TinyLlama) to exercise the harness for real.

## Why this stands out

The CartPole / LunarLander variant every candidate ships is not a signal.
The combination here — (a) a custom market-making env you designed, (b) a
from-scratch PPO single-file implementation, (c) a deterministic agent
eval harness — is much more distinctive.

## Tests

`tests/week_11/`:
- Tabular value iteration converges to a known $V^\star$ on a 5-state chain.
- Bellman contraction factor stays ≤ $\gamma$ on random $V, W$.
- GAE matches a hand-rolled reference on a 3-step rollout.
- PPO-clip loss is zero at $\pi = \pi_\text{old}$ and positive off-ratio.
- Agent loop honours `max_steps`, `final` short-circuits correctly,
  calculator handles pathological inputs, and `evaluate_agent` grades
  numeric + text tasks.

## What I learned

*To be filled in after training PPO to convergence on the market-making env.*
