"""Week 12 — problem-set starter. **This is the file you edit.**

`pytest tests/week_12` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_12
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
    raise NotImplementedError("burgers_cole_hopf")


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
    raise NotImplementedError("walk_forward_splits")


def annualised_sharpe(
    daily_returns: ArrayF, *, cost_bps: float = 0.0, turnover: ArrayF | None = None
) -> float:
    """Annualised Sharpe ratio on daily returns, minus transaction costs.

    `turnover` should be a (T,) array of fractional turnover per period if
    costs are to be subtracted. `cost_bps` is charged per unit turnover per
    side — 5 bp is typical for liquid equities.
    """
    raise NotImplementedError("annualised_sharpe")


# -----------------------------------------------------------------------------
# Sharpe leakage demo


def sharpe_leakage_demo(n: int = 1000, n_features: int = 500, seed: int = 0) -> tuple[float, float]:
    """Demonstrate that picking the best-correlated feature in-sample and
    deploying it OOS massively overstates Sharpe.

    Returns (biased_sharpe, honest_sharpe).
    """
    raise NotImplementedError("sharpe_leakage_demo")
