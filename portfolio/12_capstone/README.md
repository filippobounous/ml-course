# 12 — Capstone

> Developed in parallel with Weeks 9–12, delivered in Week 12. See `modules/12_applied_capstone/`.

Pick one primary track for the capstone. The other can be touched as a
secondary notebook for portfolio breadth.

## Track A — Physics / scientific ML

**Reference capstone:** Solve **Burgers' equation** with a PINN and compare to
the analytical (Cole–Hopf) solution. Report pointwise and $L^2$ error over
the domain, loss-decomposition plots, and a short discussion of when PINNs
fail to train.

## Track B — Quantitative finance

**Reference capstone:** Stat-arb backtest on Ken French 49 industry portfolios
with PCA / factor model, residual z-score strategy, walk-forward splits, and
transaction-cost-aware Sharpe / drawdown / turnover.

## Paper reproduction bonus

Inside `paper_reproduction/`, reproduce one figure or table from a paper
(LoRA, DDPM, PPO, or a PINN paper) at tiny scale, with an ablation table
that adds at least one extra configuration beyond the original.

## Reproduce
```bash
make -C portfolio/12_capstone reproduce
```
