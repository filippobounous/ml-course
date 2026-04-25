# Week 12 — Theory-problem solutions

## 1. Paper-reproduction plan

Open-ended — this is the item you actually write out in `portfolio/12_capstone/paper_reproduction/PLAN.md`. A good plan names a specific figure/table, a specific dataset slice, a primary metric, a compute cap, and at least one extra ablation configuration beyond the paper's.

## 2 (Track A). PINN loss for the heat equation

Heat equation: $u_t = \nu u_{xx}$, $u(x, 0) = u_0(x)$, Dirichlet $u(\pm L, t) = 0$.

PINN losses with neural net $u_\theta$:

- **Residual** $\mathcal{L}_r = \mathbb{E}_{(x,t) \sim \text{colloc}} [(u_{\theta,t} - \nu u_{\theta,xx})^2]$ at interior collocation points.
- **Initial** $\mathcal{L}_0 = \mathbb{E}_{x \sim \text{IC pts}} [(u_\theta(x, 0) - u_0(x))^2]$.
- **Boundary** $\mathcal{L}_b = \mathbb{E}_{t \sim \text{BC pts}}[u_\theta(\pm L, t)^2]$.

Total $\mathcal{L} = \lambda_r \mathcal{L}_r + \lambda_0 \mathcal{L}_0 + \lambda_b \mathcal{L}_b$. **Weights matter.** Fixed $(1, 1, 1)$ leaves the IC / BC losses under-weighted because they're sampled from lower-dimensional sets than the residual. NTK-based weighting (Wang–Yu–Perdikaris 2022) keeps the gradient norms of each loss comparable at every training step; the Phase-E `GradNormReweighter` is a simple instance of the same idea.

Gradients $u_{\theta,t}$ and $u_{\theta,xx}$ are computed via PyTorch autograd (that's the selling point of PINNs).

## 3 (Track A). Adjoint-method correctness for Neural ODEs

Forward: $h(t_1) = h(t_0) + \int_{t_0}^{t_1} f_\theta(h(t), t) dt$. Scalar loss $L(h(t_1))$. The adjoint state $a(t) = \partial L / \partial h(t)$ satisfies

$\frac{da}{dt} = -a(t)^\top \partial f_\theta / \partial h(t),$

integrated **backwards** from $t = t_1$ to $t_0$ with terminal condition $a(t_1) = \partial L / \partial h(t_1)$. The parameter gradient is

$\frac{dL}{d\theta} = -\int_{t_0}^{t_1} a(t)^\top \frac{\partial f_\theta(h(t), t)}{\partial \theta} dt.$

**Proof sketch.** Standard Pontryagin/Lagrange argument: add Lagrange multipliers enforcing the ODE at every $t$, take first-order conditions, identify $a(t)$ with the Lagrange multiplier on the constraint $\dot h = f$. The key benefit over backprop-through-solver: memory cost is O(1) in the number of forward ODE steps, because we don't need to store intermediate states — we re-integrate $h$ alongside $a$ during the backward sweep.

Chen et al. 2018 also treat $\theta$ and $t_0, t_1$ with a single augmented adjoint. `torchdiffeq.odeint_adjoint` implements this for you.

## 2 (Track B). Walk-forward with embargo

Splits: train on $[0, t_k)$, validate on $[t_k, t_k + \Delta)$, advance $t_k$. **Purge**: remove training samples whose labels are derived from information overlapping the validation window (e.g. if labels are 5-day forward returns, any training sample with an index in $[t_k - 5, t_k)$ must be purged). **Embargo**: drop an additional $e$ samples immediately after the validation window before using them for training in subsequent folds — this prevents information leakage through the autocorrelation of residuals.

Reference: López de Prado, *Advances in Financial ML*, Ch. 7. Implementation in `modules/12_applied_capstone/problems/solutions.py::walk_forward_splits`; `portfolio/12_capstone/statarb_walkforward.py` uses it.

## 3 (Track B). Backtest-leakage demonstration

Generate $T$ iid returns and 500 iid features, pure noise. Pick the feature with highest in-sample correlation to returns and trade its sign. Biased Sharpe (measured on the same sample): large. Honest Sharpe (measured on held-out data): ~0.

The implementation `sharpe_leakage_demo` in `modules/12_applied_capstone/problems/solutions.py` runs this experiment end-to-end. Expected magnitudes: biased Sharpe ~5–15 annualised, honest Sharpe near zero. The more features you select from, the larger the gap — classic selection bias / multiple-comparisons problem. Prevention: never decide hyperparameters on data you use to report performance.
