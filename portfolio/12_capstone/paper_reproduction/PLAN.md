# Paper reproduction plan — PPO clip ablation (Schulman 2017)

## Target

- **Paper.** Schulman, Wolski, Dhariwal, Radford, Klimov (2017) —
  *Proximal Policy Optimization Algorithms* (arXiv:1707.06347).
- **Specific figure to reproduce.** Figure 6 of the paper:
  performance vs $\varepsilon$ (the PPO clip parameter), reported as
  mean episode reward on continuous-control tasks.

## Data

- **Dataset / environment.** `CartPole-v1` from gymnasium. Smaller than
  the paper's MuJoCo tasks but the qualitative shape — performance
  degrading at both very-small and very-large $\varepsilon$ — should
  reproduce.
- **Preprocessing.** None; CartPole is a 4-D state vector and a
  Bernoulli action.

## Setup

- **Model.** The same `mlcourse` CleanRL-style PPO from
  `portfolio/11_rl_agent/ppo.py` (3-layer MLP actor + critic).
- **Hyperparameters.** Defaults from `PPOConfig`:
  - `total_steps = 50_000` (tiny — Schulman uses $\approx 1$ M).
  - `steps_per_rollout = 1024`.
  - $\gamma = 0.99$, $\lambda_\mathrm{GAE} = 0.95$, `update_epochs = 4`.
  - Obs normalisation **on**, advantage normalisation **on**,
    LR anneal **on**, value clip **on** (Huang 2022 "37 details" subset).
  - 4 seeds per config (CartPole has high seed-to-seed variance).

## Ablation

- **Variable.** PPO clip parameter $\varepsilon$.
- **Values.** $\{0.1, 0.2, 0.3, \text{no clip}\}$. No-clip implemented
  via $\varepsilon = 10^{6}$ — effectively vanilla policy gradient
  with GAE advantage.
- **Hypothesis.** $\varepsilon = 0.2$ should win (paper's recommended
  default). $\varepsilon = 0.1$ should be slower to learn
  (over-conservative). $\varepsilon = 0.3$ should be noisier
  (over-aggressive). No-clip should be unstable / diverge.

## Compute budget

- **Target total compute.** 4 configs × 4 seeds × $\approx 60$ s per
  run $\approx 16$ minutes wall-clock on a modern laptop CPU
  (no MPS / CUDA needed).
- **Cap.** 30 minutes — abandon and fall back to 2 seeds if exceeded.

## Success criteria

- **Qualitative match.** Mean-reward curve at $\varepsilon = 0.2$
  reaches $\gtrsim 400$ (CartPole-v1 max = 500) within 50 k steps.
  No-clip diverges or plateaus low.
- **Quantitative match.** Final mean reward at $\varepsilon = 0.2$
  within 20% of the $\varepsilon = 0.3$ value (the paper's near-tie
  on most tasks).
- **Deliverable.** `figure_clip_ablation.png` + `findings.md` with the
  ablation table and a paragraph "what I saw that surprised me".

## Risks

- **CartPole is easier than MuJoCo.** The "no-clip" instability may
  not show up at this scale — the toy env is too forgiving.
  Mitigation: state this as a known limitation in `findings.md`; if
  no-clip still wins, run a continuous-control env (`Pendulum-v1`) as
  a follow-up.
- **CPU seed variance.** Single-seed runs may not separate $0.1$ from
  $0.2$. Mitigation: 4 seeds per config + report standard deviation.
