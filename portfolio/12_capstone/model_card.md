# Model card — Applied capstone (PINN or stat-arb)

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

The W12 capstone has two **independent** tracks; pick one. This card
covers both at a high level.

---

## Track A — PINN for Burgers' equation

### Model details

- **Architecture.** 4-layer fully-connected NN with tanh
  activations, ~5 k parameters. Input $(x, t) \in [-1, 1] \times [0, T]$,
  output $u_\theta(x, t)$.
- **Loss.** Multi-term PINN loss: PDE residual + initial condition +
  boundary condition. **NTK-balanced weighting** (Wang–Yu–Perdikaris
  2022) keeps gradient norms balanced across the three terms.
- **Framework.** PyTorch (autograd is the whole point — second
  derivatives via `create_graph=True`).
- **Hyperparameters.** Adam LR 1e-3, ~20k iterations,
  $\nu = 0.01/\pi$ viscosity.

### Intended use

- **Primary.** Demonstrate that a NN trial function can solve a
  nonlinear PDE without a mesh, with provable error bounds against
  the Cole–Hopf analytical solution.
- **Out-of-scope.** Production PDE-solving. For most problems a
  finite-element solver dominates PINNs in both accuracy and
  runtime; PINNs win mostly on (1) inverse problems, (2) PDEs with
  unknown coefficients you want to learn.

### Metrics

- $L^2$ error vs Cole–Hopf analytical solution.
- Residual decomposition: PDE / IC / BC norms separately.

### Quantitative analyses

| Metric | Target | Verified |
|---|---|---|
| $L^2(u_\theta, u_\text{exact})$ | $\le 10^{-2}$ | ⏳ |
| Training runtime (MPS) | ~20 min | ⏳ |

### Caveats

- Without NTK reweighting, the same loss converges to a degenerate
  solution that satisfies the PDE residual but not the boundary
  conditions. Don't run with fixed weights and claim success.
- The Cole–Hopf comparison is for $\nu \ge 10^{-2}$. At smaller
  viscosities, shocks form and the analytical solution becomes
  ill-conditioned; the PINN often fails before then.

---

## Track B — Walk-forward PCA stat-arb

### Model details

- **Algorithm.** Avellaneda–Lee 2008 residual model on K-French 49
  industry portfolios. Walk-forward splits with **embargo + purging**
  (López de Prado Ch. 7).
- **Decision rule.** Trade signs of standardised residual
  z-scores; open at $|z| > 2$, close at $|z| < 0.5$.
- **Framework.** NumPy + pandas. No deep learning.

### Intended use

- **Primary.** Demonstrate **honest** quant backtesting:
  walk-forward CV, embargoing, realistic transaction costs,
  drawdown reporting alongside Sharpe.
- **Out-of-scope.** Live trading. The model is decades old; modern
  stat-arb at scale uses many more factors and far more sophisticated
  risk management.

### Metrics

- **Net Sharpe** (annualised, post-TCA).
- **Turnover** (fraction of book traded per period).
- **Maximum drawdown** and **Sortino**.

### Quantitative analyses

| Metric | Target (simulated returns) | Verified |
|---|---|---|
| IS Sharpe | $\approx 3.2$ | ✅ (sim) |
| OOS Sharpe | $\approx 2.9$ | ✅ (sim) |
| OOS Sharpe (real K-French) | TBD | ⏳ |

### Caveats

- The IS/OOS Sharpe targets are for **simulated** AR(1)-with-noise
  returns. On real K-French data the Sharpe is likely lower and
  may flip sign across decades.
- The selection-bias inflation $\rho^\star \sim \sqrt{2 \log K / T}$
  applies — make sure embargoing + walk-forward are actually
  enforced before quoting any number.

---

## Both tracks

### Ethical considerations

- **Track A**: PINNs do not produce uncertainty estimates by
  default — quoting a single $L^2$ error without a CI is misleading.
- **Track B**: backtests that look great on simulated returns rarely
  survive deployment. Be honest in your write-up about what you
  haven't measured (slippage, market impact, regime change).
