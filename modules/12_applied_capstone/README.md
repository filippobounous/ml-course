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

## Reading plan

See `readings.md`.
