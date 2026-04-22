# Week 12 — Applied tracks + capstone (lecture notes)

Two tracks; pick one as the capstone depth, touch the other for breadth.

*Reading pair (A): Raissi *PINNs* 2019 · Chen *Neural ODEs* 2018 · Cranmer *PySR* 2023 · Lu *DeepXDE* 2021.*

*Reading pair (B): López de Prado *Advances in Financial ML* Ch.2, 3, 7 · Fama & French 2015 · Oreshkin *N-BEATS* 2019.*

---

## Track A — Physics / scientific ML

### Physics-informed neural networks (PINNs)

We want to solve a PDE $\mathcal{F}[u](x, t) = 0$ with boundary and initial conditions. Classical solvers (finite differences, finite elements) discretise in space and time. PINNs parameterise $u$ as a neural net $u_\theta$ and minimise

$$\mathcal{L}(\theta) = \lambda_r \mathcal{L}_\text{res}(\theta) + \lambda_b \mathcal{L}_\text{bc}(\theta) + \lambda_0 \mathcal{L}_\text{ic}(\theta),$$

where

- $\mathcal{L}_\text{res}$ enforces the PDE residual at collocation points,
- $\mathcal{L}_\text{bc}$ enforces the boundary conditions,
- $\mathcal{L}_\text{ic}$ enforces the initial condition.

Derivatives like $u_t$ and $u_{xx}$ are obtained **for free by autograd** — this is the real engineering win over classical solvers. Loss-weighting $\lambda_*$ matters a lot (Wang, Yu, Perdikaris 2022): NTK-based weightings work better than fixed ones for most problems.

### Burgers' equation (the canonical PINN demo)

$$u_t + u u_x = \nu u_{xx}, \quad x \in [-1, 1], \quad t \in [0, 1].$$

With $u(x, 0) = -\sin(\pi x)$ and $u(\pm 1, t) = 0$. The **Cole–Hopf transform** gives an analytical solution for this BC / IC pair, which is what makes it a great benchmark: we can compute the exact pointwise error.

### Neural ODEs

Chen et al. (2018) frame a residual network as a discrete-time approximation of an ODE and push it to the continuous limit: $dh/dt = f_\theta(h, t)$. Training uses the **adjoint method** to backprop through an ODE solver with O(1) memory in the depth.

Practical toolkit: `torchdiffeq` gives `odeint(func, h0, t)` with Dopri5 / Heun solvers. Good for continuous-time series, normalising flows, and as a stand-alone research tool.

### Symbolic regression

For low-dimensional scientific problems, symbolic regression (SR) finds **closed-form** expressions that fit the data. `PySR` (Cranmer 2023) uses genetic programming over a user-configurable operator set; runs fast on multi-core CPU via Julia under the hood.

A representative workflow: fit a neural net on noisy physics data → use SR on the net's predictions to recover an interpretable equation.

---

## Track B — Quantitative finance

### Walk-forward validation (and why K-fold is wrong)

In an i.i.d. classification problem, K-fold CV estimates generalisation. In finance, returns have **serial dependence** and the future of training data leaks into the past of validation data. Two consequences:

- K-fold CV massively overestimates out-of-sample performance.
- Cross-validated hyperparameters overfit to calendar effects.

Use **walk-forward splits**: train on $[0, t)$, validate on $[t, t + \Delta)$, increment $t$. And apply **purging + embargoing** (López de Prado Ch. 7): remove samples whose label horizon overlaps the validation window, and drop a small embargo period immediately after the validation window to prevent information leakage through autocorrelation.

### Transaction-cost-aware evaluation

For any live strategy:

1. Simulate trades with a **realistic slippage model** (e.g. 5–10 bp per side for liquid equities; higher for less-liquid instruments).
2. Annualise net returns, report net **Sharpe** and **Sortino**.
3. Compute **turnover** (fraction of portfolio value traded per period) and show that gross → net Sharpe degrades roughly proportionally to turnover × cost.
4. Plot equity curves and **drawdowns**. Big drawdown + high Sharpe is not the same quality as small drawdown + high Sharpe.

### Factor models in the DL era

Classical factors (market, size, value, momentum) explain a lot of returns. Two ways to extend them with ML:

- **Factor timing.** Use ML to predict which factors will perform in the next period.
- **Residual / alpha modelling.** Regress residuals (after factor regression) on any side information; model residuals with a small neural net trained on rolling windows.

Time-series transformers (Informer, PatchTST) and N-BEATS are the modern deep baselines for pure return forecasting; on short series they rarely beat a careful AR(1)-GARCH + elastic-net baseline, so make the baseline fight.

---

## Paper reproduction (bonus artifact)

Pick a paper with a reproducible headline figure at tiny scale. Examples:

- **LoRA** — adapter-rank ablation on a small SFT task (Week 9 groundwork in place).
- **DDPM** — noise-schedule comparison on FashionMNIST (Week 10 groundwork in place).
- **PPO** — clip-parameter ablation on CartPole (Week 11 groundwork in place).
- **PINNs** — Raissi et al. 2019 Figure 2 (Burgers' residual vs training time).

Commit to one figure **or** table to reproduce. Write a one-page plan
with dataset, metric, compute budget, and what you expect to see. Add at
least one configuration the paper didn't ablate, and report it honestly.

## What to do with these notes

Work the problem set in `../problems/README.md`. Build the capstone in
`../../../portfolio/12_capstone/` (primary track) and the bonus paper
reproduction under `portfolio/12_capstone/paper_reproduction/`.
