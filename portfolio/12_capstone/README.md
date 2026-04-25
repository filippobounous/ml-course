# 12 — Capstone

Two tracks — pick **one as your primary** and touch the other for breadth.
Plus a **paper reproduction** bonus artifact.

## Track A — Physics / scientific ML

**Headline artifact.** Solve Burgers' equation with a PINN and compare to the
analytical Cole-Hopf solution.

- `pinn_burgers.py` — PINN model (6-layer tanh MLP), residual computed via
  autograd (`u_t + u u_x - ν u_xx`), IC / BC enforced via weighted MSE.
- `demo_pinn.py` — trains the PINN (Adam, 4k iters on MPS ≈ 20 min), evaluates
  against the Cole-Hopf solution on a 50×101 grid, saves `pinn_vs_exact.png`
  (exact | PINN | error panels) and a one-line error report.

### Reproduce

```bash
python -m pip install -e ".[dl,sciml,ops]"
python portfolio/12_capstone/demo_pinn.py --quick      # 200 iters smoke
python portfolio/12_capstone/demo_pinn.py              # full 4k iters
```

Target: relative $L^2$ error ≤ $10^{-2}$ (vs. Raissi's $\sim 10^{-3}$ at much
longer training). Sup-norm error in single digits of %.

## Track B — Quantitative finance

**Headline artifact.** Walk-forward stat-arb on an OU-residual factor model,
with purging + embargo and transaction-cost-aware Sharpe.

- `statarb_walkforward.py` — reuses the Week-4 backtest engine, wraps it in
  walk-forward folds with an embargo, reports gross / net Sharpe, max
  drawdown, mean turnover, and per-fold metrics.
- `demo_statarb.py` — runs the backtest on simulated returns and writes
  `statarb_results.md`.

### Reproduce

```bash
python portfolio/12_capstone/demo_statarb.py
```

Runs offline in seconds. Swap in `pca_statarb.load_ken_french_industries()`
for live data.

## Paper reproduction (bonus)

Scaffold under `paper_reproduction/`:

- `PLAN.md` — the one-page template you fill in *before* coding.
- `README.md` — reproducibility contract for the recruiter.

Pick a paper with a reproducible headline figure at tiny scale — LoRA, DDPM,
PPO, or a PINN paper are all good candidates and each has groundwork from
earlier weeks.

## Tests

`tests/week_12/` covers:
- Cole-Hopf reference at `t = 0` recovers the IC.
- Walk-forward splitter produces non-overlapping, ordered folds and honours
  the embargo.
- TCA Sharpe: zero-turnover case returns the gross Sharpe; non-zero turnover
  drops Sharpe monotonically in cost.
- Sharpe-leakage demo: biased Sharpe >> honest Sharpe on pure noise.
- Torch-gated PINN forward-pass + residual shape.

## What I learned

*To be filled in after capstone delivery. This is the piece to send to
recruiters — make it count.*
