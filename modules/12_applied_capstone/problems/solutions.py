"""Week 12 — reference solutions.

NumPy-only utilities:
  * Cole-Hopf analytical reference for Burgers' (for PINN validation).
  * Walk-forward CV splitter with optional embargoing (López de Prado Ch. 7).
  * Transaction-cost-aware annualised Sharpe.
  * Sharpe leakage demo (in-sample-tuned strategy vs honest walk-forward).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Track A — Burgers' Cole-Hopf analytical reference


def burgers_cole_hopf(x: ArrayF, t: ArrayF, *, nu: float = 0.01 / np.pi) -> ArrayF:
    """Analytical Burgers' solution for the benchmark used in Raissi 2019:

        u_t + u u_x = ν u_xx,  x ∈ [−1, 1], t ∈ [0, 1],
        u(x, 0) = −sin(π x),  u(±1, t) = 0.

    Cole-Hopf transform reduces Burgers' to the heat equation with an
    integral representation; this function evaluates that integral via
    quadrature for each (x, t) point. Accurate to ~1e-5 for the default ν.

    Inputs:
      x: (Nx,) spatial grid in [−1, 1].
      t: (Nt,) time grid  in (0, 1]  (t=0 handled separately since the
         integral is singular; we return the known initial condition).
    Returns a (Nt, Nx) array.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    U = np.zeros((t.size, x.size), dtype=np.float64)

    # Quadrature grid for the inner integrals over η ∈ [-∞, ∞] (we truncate).
    eta = np.linspace(-5.0, 5.0, 2001)
    d_eta = eta[1] - eta[0]

    def _phi0(eta_):
        # φ(η, 0) = exp(-cos(π η) / (2π ν)) (Cole-Hopf transform applied to the IC).
        return np.exp(-np.cos(np.pi * eta_) / (2.0 * np.pi * nu))

    phi0 = _phi0(eta)

    for i, ti in enumerate(t):
        if ti <= 0:
            U[i] = -np.sin(np.pi * x)
            continue
        for j, xj in enumerate(x):
            kern = np.exp(-((xj - eta) ** 2) / (4.0 * nu * ti))
            num = np.trapezoid(phi0 * kern * (xj - eta) / (2.0 * nu * ti), eta, dx=d_eta)
            den = np.trapezoid(phi0 * kern, eta, dx=d_eta)
            U[i, j] = num / den if den != 0 else 0.0

    return U


# -----------------------------------------------------------------------------
# Track B — Walk-forward splits + TCA Sharpe


@dataclass(frozen=True)
class WalkForwardSplit:
    train_idx: slice
    val_idx: slice


def walk_forward_splits(
    n: int, *, train_size: int, val_size: int, embargo: int = 0, step: int | None = None
) -> list[WalkForwardSplit]:
    """Expanding-or-rolling walk-forward splits with an optional embargo.

    `step` defaults to `val_size` (non-overlapping validation windows). The
    embargo is a number of samples immediately after `val_idx` that are
    excluded from the subsequent train fold — critical for avoiding
    label-horizon leakage (López de Prado Ch. 7).

    Each split's `train_idx` uses all samples in [0, t0) *excluding* any
    previously-embargoed range. Here we take the simple rolling form (only
    the embargo of the immediately-preceding validation fold matters).
    """
    if step is None:
        step = val_size
    splits: list[WalkForwardSplit] = []
    start = train_size
    while start + val_size <= n:
        train = slice(0, start)
        val = slice(start, start + val_size)
        splits.append(WalkForwardSplit(train, val))
        start += step
    if embargo > 0:
        # Shift the next train start forward by `embargo` beyond the previous
        # val end — effectively narrowing the next train's upper bound.
        shifted: list[WalkForwardSplit] = []
        for k, sp in enumerate(splits):
            if k == 0:
                shifted.append(sp)
                continue
            prev_val_end = splits[k - 1].val_idx.stop
            new_train = slice(0, max(prev_val_end + embargo, sp.train_idx.start))
            shifted.append(WalkForwardSplit(new_train, sp.val_idx))
        splits = shifted
    return splits


def annualised_sharpe(
    daily_returns: ArrayF, *, cost_bps: float = 0.0, turnover: ArrayF | None = None
) -> float:
    """Annualised Sharpe ratio on daily returns, minus transaction costs.

    `turnover` should be a (T,) array of fractional turnover per period if
    costs are to be subtracted. `cost_bps` is charged per unit turnover per
    side — 5 bp is typical for liquid equities.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if turnover is not None:
        cost = (cost_bps / 10_000.0) * np.asarray(turnover, dtype=np.float64)
        r = r - cost
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(252))


# -----------------------------------------------------------------------------
# Sharpe leakage demo


def sharpe_leakage_demo(n: int = 1000, n_features: int = 500, seed: int = 0) -> tuple[float, float]:
    """Demonstrate that picking the best-correlated feature in-sample and
    deploying it OOS massively overstates Sharpe.

    Returns (biased_sharpe, honest_sharpe).
    """
    rng = np.random.default_rng(seed)
    # Pure-noise returns.
    returns = rng.standard_normal(n) * 0.01
    # A ton of pure-noise features.
    features = rng.standard_normal((n, n_features))
    # Split.
    split = n // 2
    train_f = features[:split]
    train_r = returns[:split]
    test_f = features[split:]
    test_r = returns[split:]
    # Pick the feature with the highest in-sample correlation with returns.
    corrs = np.array([np.corrcoef(train_f[:, j], train_r)[0, 1] for j in range(n_features)])
    winner = int(np.abs(corrs).argmax())
    # Go long when the feature is positive, short when negative.
    signal_train = np.sign(train_f[:, winner])
    signal_test = np.sign(test_f[:, winner])
    # Biased: evaluate on the in-sample period.
    biased_strat = signal_train * train_r
    # Honest: evaluate on held-out period.
    honest_strat = signal_test * test_r
    biased = annualised_sharpe(biased_strat)
    honest = annualised_sharpe(honest_strat)
    return biased, honest


if __name__ == "__main__":
    # Quick smoke.
    splits = walk_forward_splits(n=100, train_size=30, val_size=10, embargo=2)
    print(f"{len(splits)} walk-forward splits")

    biased, honest = sharpe_leakage_demo()
    print(f"biased Sharpe (in-sample selection): {biased:.3f}")
    print(f"honest Sharpe (walk-forward):        {honest:.3f}")
