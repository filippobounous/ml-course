# Paper reproduction (bonus W12 artifact)

Reproduce a figure or table from a target paper at tiny scale, with at least
one extra configuration beyond the original.

## What's here

- `PLAN.md` — the concrete plan (paper, figure, dataset, hyperparameters,
  ablation, success criteria). Filled in for **Schulman 2017 PPO** by
  default; rewrite if you pick a different target.
- `ppo_clip_ablation.py` — runs PPO with $\varepsilon \in \{0.1, 0.2, 0.3, \text{no-clip}\}$
  on `CartPole-v1`, 4 seeds per config. Imports the W11 `mlcourse` PPO
  from `portfolio/11_rl_agent/ppo.py`. Writes `findings.md` and
  `figure_clip_ablation.png`.
- `findings.md` — placeholder until the reproduction runs.
- `Makefile` — `make reproduce` (full) and `make quick` (CI smoke).

## Reproduce

```bash
# Full reproduction (~15 min on CPU).
make -C portfolio/12_capstone/paper_reproduction reproduce

# CI smoke (~30 s, 1 seed × 4 k steps per config).
make -C portfolio/12_capstone/paper_reproduction quick
```

## Target paper options (pick one for your own pass)

- **PPO** (Schulman 2017) — clip-parameter ablation on CartPole / Pendulum.
  **Default reproduction shipped in this directory.**
- **LoRA** (Hu 2021) — adapter-rank ablation on a small SFT task.
- **DDPM** (Ho 2020) — noise-schedule figure on FashionMNIST.
- **PINNs** (Raissi 2019) — Burgers' equation residual figure with
  alternative loss weighting.

If you pick one of the other three, rewrite `PLAN.md` first; the W11 / W9 / W10 / W12
artifacts in this repo already provide the underlying machinery.

## Why PPO?

It's the most CPU-friendly of the four (CartPole trains in a minute), uses
W11's own implementation end-to-end, and Schulman's Figure 6 is a clean
single-variable ablation. The other three options are equally valid choices
but heavier on dependencies (HF, InceptionV3) or runtime.

