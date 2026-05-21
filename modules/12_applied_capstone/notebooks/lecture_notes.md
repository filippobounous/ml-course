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

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three concrete exercises (PINN residual via PyTorch autograd with the `create_graph=True` pattern, walk-forward split with explicit 5-day purging + embargo, selection-bias Sharpe inflation showing $\rho^\star \sim \sqrt{2\log K / T}$).

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| Paper-repro plan | 1 | One-page `PLAN.md` for the chosen paper (LoRA / DDPM / PPO / PINN). |
| **Track A** PINN | 6 | Burgers' setup; PINN loss; GradNorm reweighting; train to $L^2 \le 10^{-2}$. |
| **Track A** adjoint method | 2 | Pontryagin sketch; run `torchdiffeq.odeint_adjoint`. |
| **Track B** walk-forward | 6 | Embargo + purging; PCA stat-arb residuals; honest Sharpe vs leakage. |
| **Track B** backtest hygiene | 2 | Transaction-cost-aware Sharpe / turnover / drawdown. |
| Capstone write-up | 3 | Final figures, honest reporting, model card. |
| Paper reproduction | 2 | Execute the PLAN; commit one extra ablation beyond the paper. |

(Pick Track A *or* Track B for the primary capstone; the paper-reproduction sub-artifact is required either way.)

## Self-assessment rubric

Before declaring the course complete, you should be able to answer "yes" to all of:

1. **(Track A)** Can I derive the full PINN loss for the heat equation and explain why fixed loss weights $(1, 1, 1)$ underweight the boundary terms?
2. **(Track A)** Can I sketch the adjoint-method correctness proof (Pontryagin / Lagrange multipliers) and explain its $\mathcal{O}(1)$-memory advantage over backprop-through-solver?
3. **(Track B)** Can I implement walk-forward splits with embargo + purging and verify they have no temporal leakage via a synthetic-noise selection-bias demonstration?
4. **(Track B)** Can I produce a transaction-cost-aware backtest report (net Sharpe, turnover, drawdown) and explain why high turnover degrades gross-to-net Sharpe roughly proportionally?
5. **(Both)** Can I write a one-page paper-reproduction `PLAN.md`, execute the chosen figure or table at tiny scale, and add **at least one** extra ablation configuration the paper didn't run?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **PINN ↔ weak / variational formulation of a PDE.** Minimising the squared residual $\int |R[u_\theta]|^2 \, d\mu$ over a discrete sample is a Galerkin-style weak formulation — same idea as the finite-element method, but with neural-net trial functions instead of piecewise polynomials. Loss weighting via NTK is the analogue of choosing a metric on the space of test functions.
- **Neural-ODE adjoint ↔ Pontryagin maximum principle.** The adjoint state $a(t) = \partial L / \partial h(t)$ is precisely the **costate** in optimal-control theory; the backward ODE $\dot a = -a^\top \partial f / \partial h$ is the Hamiltonian adjoint equation. The "free lunch" of $\mathcal{O}(1)$ memory comes from integrating both equations together — same trick used in time-reversible classical-mechanics solvers.
- **Burgers' equation ↔ inviscid limit ↔ shock formation.** As viscosity $\nu \to 0$ in $u_t + u u_x = \nu u_{xx}$, smooth initial data forms shocks in finite time. The Cole–Hopf transformation linearises Burgers' into the heat equation — the same trick as the Bäcklund transformations in soliton theory. Your PINN should reproduce the shock structure; if it blurs it, the residual weighting is wrong.
- **Selection bias ↔ Berkson's paradox / multiple-comparisons inflation.** Picking the best of $K$ noisy estimators inflates the apparent effect by $\sqrt{2 \log K / T}$ — same combinatorial inflation as the maximum of $K$ independent Gaussians (extreme-value theory). The cure is **pre-registration** (decide the rule before looking) — exactly what a particle-physics blinded analysis does for the same reason.
- **Walk-forward embargo ↔ retarded propagator / causality on the time arrow.** The embargo zone is the analogue of a coherence time after which residual autocorrelations decay; sampling outside the embargo is sampling outside the (residual) light cone of the validation set.

Diffusion (W10), RL (W11), and PINN / stat-arb (W12) all rest on the same continuous-time stochastic / control-theoretic toolkit. The hardest part of the course is also where the physics analogies pay off most.
