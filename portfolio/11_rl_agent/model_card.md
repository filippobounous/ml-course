# Model card — PPO on a custom market-maker env (+ tool-use agent)

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

## Model details

- **Two artifacts in this directory:**

### A. PPO policy + value network

- **Architecture.** Two-head 3-layer MLP (actor + critic), $\sim 5 k$
  parameters. Discrete action space (spread levels).
- **Algorithm.** CleanRL-style PPO with the Huang-2022 "37 details"
  subset: observation normalisation (Welford running stats), advantage
  normalisation, linear LR anneal, value clipping.
- **Framework.** PyTorch; **not** via `mlcourse.Trainer` (rollout →
  GAE → K-epoch minibatch update doesn't fit the DataLoader shape;
  documented in PR #11).
- **Hyperparameters.** $\gamma = 0.99$, $\lambda = 0.95$,
  $\varepsilon = 0.2$, 4 epochs/rollout, 200k env steps target.

### B. ReAct-style tool-use agent

- **Architecture.** Torch-free. Hand-written policies +
  `calculator` tool (AST-restricted arithmetic) + keyword
  retriever. Deterministic given the policy.
- **Framework.** Pure Python. Designed to be plug-in for an
  external LLM (Claude / GPT / TinyLlama) as the policy.

## Intended use

### A. PPO

- **Primary.** Show end-to-end PPO on a custom non-trivial env
  (market-making with quadratic inventory penalty).
- **Out-of-scope.** Live trading. The market-maker env is a
  toy with hand-tuned reward; gen-rich production envs differ
  dramatically.

### B. Agent

- **Primary.** Demonstrate the **eval-harness** for tool-use
  agents — deterministic grading, failure taxonomy, 20 canned tasks.
- **Out-of-scope.** Acting as a "real" agent. The hand-written
  policies are for unit-testing the harness; only with an LLM
  backend (W13) does this become a meaningful agent.

## Metrics

### A. PPO

- **Mean episode return** (over last 10 episodes).
- **A no-quote baseline** earns 0 per step; convergence threshold
  is mean episode return crossing zero by 100k steps.

### B. Agent

- **Success rate** on 20 canned tasks under a fixed seed.

## Training / evaluation data

### A. PPO

- **`SimpleMarketMakerEnv`** — synthetic, hand-designed
  (intertemporal reward, inventory penalty, hit dynamics). No PII.

### B. Agent

- **20 canned tasks** (math + lookup). No PII.

## Quantitative analyses

| Artifact | Metric | Target | Verified |
|---|---|---|---|
| PPO | Mean return at 100k steps | > 0 | ⏳ |
| PPO | Mean return at 200k steps | +10 to +30 | ⏳ |
| Agent | Success rate on canned tasks | 100% (hand-policies) | ✅ |

## Caveats

- **PPO**: market-maker reward is synthetic. Don't extrapolate to
  real markets — a working PPO here is *necessary but not sufficient*
  for live trading.
- **PPO**: seed variance is *large* — quote standard deviation
  across ≥ 4 seeds before declaring convergence.
- **Agent**: success on hand-written policies says nothing about
  LLM-policy success. The harness is the contribution, not any
  particular agent score.
