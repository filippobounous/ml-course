# Week 12 — Applied tracks + capstone delivery

Pick one primary track (A or B) for the capstone. Touch the other to breadth of portfolio.

## Track A — Physics / scientific ML

### Learning objectives

1. Formulate a **Physics-Informed Neural Network (PINN)** for a PDE; choose collocation points, boundary / initial conditions, loss weighting.
2. Fit **Neural ODEs** via the adjoint method (`torchdiffeq`) on a dynamical system.
3. Discover governing equations via **symbolic regression** with **PySR**.

### Reference capstone

Solve **Burgers' equation** $u_t + u u_x = \nu u_{xx}$ with a PINN. Compare to the analytical / Cole–Hopf solution. Report pointwise error and a loss-decomposition plot.

## Track B — Quantitative finance

### Learning objectives

1. Construct **factor models** and evaluate them with walk-forward splits.
2. Fit a small **time-series transformer** / N-BEATS-lite to forecast returns; compare to ARIMA / GARCH baselines.
3. Build a **backtest harness** with transaction costs and a realistic execution model.

### Reference capstone

**Stat-arb backtest** on Ken French 49 industry portfolios: dimension reduction (PCA / factor model) + residual z-score strategy + walk-forward validation + transaction-cost-aware Sharpe.

## Deliverables

- Portfolio artifact: `portfolio/12_capstone/` — capstone project (PINN **or** stat-arb).
- Bonus artifact: `portfolio/12_capstone/paper_reproduction/` — reproduce a figure from a paper (LoRA, DDPM, PPO, or a PINN paper) at tiny scale with an ablation table.

## What ships as reference vs what you build

The objectives above are broader than what the repo implements — only the **reference
capstones** ship as runnable code. Treat the rest as topics you implement *if your capstone
needs them* (consistent with the README "taught vs implemented" policy).

| Component | Status |
|---|---|
| Track A — PINN for Burgers' (`portfolio/12_capstone/pinn_burgers.py`, `demo_pinn.py`) | **reference** — read, run, extend |
| Track A — Neural ODEs (`torchdiffeq`), PySR symbolic regression | **taught only** — implement if you choose them |
| Track B — PCA / factor stat-arb (`portfolio/12_capstone/statarb_walkforward.py`, `demo_statarb.py`) | **reference** — read, run, extend |
| Track B — time-series transformer / N-BEATS-lite, GARCH baselines | **taught only** — implement if you choose them |
| Paper reproduction (`paper_reproduction/ppo_clip_ablation.py`) | **reference scaffold** — adapt to your chosen paper |

Your capstone = pick **one** track's reference, then add a contribution on top (an ablation,
a new baseline, a different dataset). Grading rubric: [`capstone/README.md`](../../capstone/README.md).

## Reading plan

See `readings.md`.
