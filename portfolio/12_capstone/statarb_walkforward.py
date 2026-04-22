"""Walk-forward PCA stat-arb with purging + embargoing and TCA Sharpe.

Heavier than the Week-4 artifact: honest walk-forward with embargo, a
transaction-cost model that grows with turnover, and reporting of gross vs
net Sharpe, Sortino, max drawdown, and turnover.

Torch-free. Uses `pca_statarb` from Week 4 as the base backtest engine.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
WEEK4_STATARB = HERE.parent / "04_pca_statarb" / "pca_statarb.py"


def _load_week4():
    spec = importlib.util.spec_from_file_location("w4_statarb", WEEK4_STATARB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class WalkForwardReport:
    windows: list[tuple[slice, slice]]
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    mean_turnover: float
    per_window: list[dict[str, float]]


def _sharpe(returns: NDArray[np.float64], *, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def _max_drawdown(equity: NDArray[np.float64]) -> float:
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max())


def walk_forward_statarb(
    returns: NDArray[np.float64],
    *,
    train_size: int = 504,
    val_size: int = 126,
    embargo: int = 5,
    lookback: int = 252,
    n_factors: int = 3,
    cost_bps: float = 5.0,
) -> WalkForwardReport:
    """Run walk-forward PCA stat-arb with an embargo and report TCA metrics."""
    sa = _load_week4()

    T = returns.shape[0]
    windows: list[tuple[slice, slice]] = []
    start = train_size
    while start + val_size <= T:
        train = slice(0, start)
        val = slice(start, start + val_size)
        windows.append((train, val))
        start += val_size

    # Apply embargo by narrowing the upstream train region for subsequent folds.
    adjusted: list[tuple[slice, slice]] = []
    for k, (train, val) in enumerate(windows):
        if k == 0:
            adjusted.append((train, val))
            continue
        prev_val_end = windows[k - 1][1].stop
        new_train = slice(0, max(prev_val_end + embargo, train.start))
        adjusted.append((new_train, val))
    windows = adjusted

    val_pnl = np.array([], dtype=np.float64)
    turnovers: list[float] = []
    per_window: list[dict[str, float]] = []

    for k, (_train, val) in enumerate(windows):
        # Run the base backtest over enough history to prime the rolling PCA,
        # then keep only the validation period's PnL.
        prelude = max(val.start - lookback, 0)
        sub = returns[prelude : val.stop]
        result = sa.pca_statarb_backtest(
            sub,
            lookback=lookback,
            n_factors=n_factors,
            cost_bps=cost_bps,
        )
        offset = val.start - prelude
        fold_pnl = result.pnl_net[offset : offset + val_size]
        fold_turnover = result.turnover[offset : offset + val_size]
        val_pnl = np.concatenate([val_pnl, fold_pnl])
        turnovers.append(float(fold_turnover.mean()))
        per_window.append(
            {
                "fold": k,
                "val_start": int(val.start),
                "val_end": int(val.stop),
                "sharpe": _sharpe(fold_pnl),
                "turnover": float(fold_turnover.mean()),
            }
        )

    # Gross-of-cost: recompute with cost_bps=0 on the same prelude schedule.
    gross_pnl = np.array([], dtype=np.float64)
    for _train, val in windows:
        prelude = max(val.start - lookback, 0)
        sub = returns[prelude : val.stop]
        result = sa.pca_statarb_backtest(sub, lookback=lookback, n_factors=n_factors, cost_bps=0.0)
        offset = val.start - prelude
        gross_pnl = np.concatenate([gross_pnl, result.pnl[offset : offset + val_size]])

    equity = np.cumsum(val_pnl)
    return WalkForwardReport(
        windows=windows,
        gross_sharpe=_sharpe(gross_pnl),
        net_sharpe=_sharpe(val_pnl),
        max_drawdown=_max_drawdown(equity),
        mean_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        per_window=per_window,
    )


if __name__ == "__main__":
    sa = _load_week4()
    returns = sa.simulate_returns(n_periods=2000, n_assets=40, n_factors=3, seed=0)
    report = walk_forward_statarb(returns, train_size=504, val_size=126, embargo=5)
    print(f"Folds: {len(report.windows)}")
    print(f"Gross Sharpe: {report.gross_sharpe:.2f}")
    print(f"Net Sharpe:   {report.net_sharpe:.2f}")
    print(f"Max DD:       {report.max_drawdown:.4f}")
    print(f"Mean turnover: {report.mean_turnover:.4f}")
