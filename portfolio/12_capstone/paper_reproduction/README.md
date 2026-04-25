# Paper reproduction (bonus artifact)

> Populated during Week 12. See `modules/12_applied_capstone/`.

Reproduce a figure or table from a target paper at tiny scale, with at least
one extra configuration beyond the original (an honest ablation).

## Plan

See `PLAN.md` (authored at the start of Week 12; contains dataset, metric,
compute budget, and the specific figure/table to reproduce).

## Target paper options

- LoRA (Hu et al., 2021) — adapter-rank ablation on a small SFT task.
- DDPM (Ho et al., 2020) — noise-schedule figure on FashionMNIST.
- PPO (Schulman et al., 2017) — clip-parameter ablation on CartPole / Pendulum.
- PINNs (Raissi et al., 2019) — Burgers' equation residual figure with
  alternative loss weighting.

## Reproduce
```bash
make -C portfolio/12_capstone/paper_reproduction reproduce
```
