# Problem set — Week 12

Pick the subset matching your capstone track.

## Theory (both tracks)

1. **Paper-reproduction plan.** Pick a target paper (LoRA, DDPM, PPO, a PINN paper). Identify one figure or table to reproduce at tiny scale. Write a one-page plan with dataset, metric, and compute budget. Deliver in `portfolio/12_capstone/paper_reproduction/PLAN.md`.

## Track A — Scientific ML

2. **PINN loss derivation.** For the heat equation $u_t = \nu u_{xx}$, write the full PINN loss (residual + boundary + initial) explicitly. Describe a sensible loss-weighting scheme.
3. **Adjoint method correctness.** Sketch the proof that the adjoint-method gradient (used by Neural ODEs) matches direct backpropagation through the ODE solver.
4. **Implementation.** Solve **Burgers' equation** with a PINN. Compare to the Cole–Hopf analytical solution on a grid; report $L^2$ error and residual decomposition. Deliver in `portfolio/12_capstone/` (if Track A is chosen).

## Track B — Quantitative finance

2b. **Walk-forward splitting.** Implement a walk-forward split honouring embargoing / purging (López de Prado Ch. 7). Unit-test it.
3b. **Backtest trap.** Demonstrate how in-sample leakage inflates Sharpe on a simulated AR(1) return series; then show the correct walk-forward result.
4b. **Implementation.** Run a **PCA-based stat-arb backtest** on Ken French industry portfolios with transaction costs; report Sharpe, turnover, drawdown. Deliver in `portfolio/12_capstone/` (if Track B is chosen).

## Paper reproduction (both tracks)

5. Execute the reproduction planned in (1). Produce an ablation table with one extra configuration beyond the paper. Deliver in `portfolio/12_capstone/paper_reproduction/`.

## Grading

Tests in `tests/week_12/` check: (Track A) PDE residual decreases monotonically over training on a toy problem; (Track B) walk-forward split has no temporal leakage; (both) the paper-reproduction directory contains `PLAN.md` + a results notebook.
