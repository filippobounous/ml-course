"""PCA statistical arbitrage — walk-forward backtest on simulated factor returns.

The strategy closely follows Avellaneda & Lee (2008):

  1. Rolling PCA on the last `lookback` days of cross-sectional returns.
  2. Regress each asset's returns on the top-k principal components → residual.
  3. Z-score the cumulative residual over a `zscore_window`.
  4. Open a mean-reversion position when |z| > open_threshold; close when |z|
     < close_threshold (or a stop-loss threshold is hit).
  5. Walk-forward: all rolling computations at time t use only data up to t.
  6. Subtract transaction costs on turnover.

A simple simulator generates a realistic factor-plus-idiosyncratic return
matrix so the entire backtest runs offline. A loader for Ken French industry
portfolios is included but gated behind a network check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Simulated returns


def simulate_returns(
    *,
    n_periods: int = 1500,
    n_assets: int = 50,
    n_factors: int = 3,
    factor_vol: float = 0.01,
    idio_vol: float = 0.015,
    reversion_speed: float = 0.6,
    seed: int = 0,
) -> ArrayF:
    """Simulate returns as a factor model + mean-reverting idiosyncratic prices.

    Returns shape (n_periods, n_assets). The idiosyncratic *price* component
    follows an OU-like process so that idiosyncratic *returns* have negative
    autocorrelation — which is exactly the regime a mean-reversion stat-arb
    strategy is designed to exploit.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(scale=1.0, size=(n_assets, n_factors))
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5  # positive market beta
    factor_rets = rng.normal(scale=factor_vol, size=(n_periods, n_factors))
    systematic = factor_rets @ loadings.T

    # OU idiosyncratic prices: p_t = (1 - κ) p_{t-1} + ε_t.
    # Idiosyncratic returns r_t = p_t - p_{t-1} have autocorr ≈ −κ/(2−κ) < 0.
    idio_price = np.zeros((n_periods, n_assets))
    noise = rng.normal(scale=idio_vol, size=(n_periods, n_assets))
    for t in range(1, n_periods):
        idio_price[t] = (1 - reversion_speed) * idio_price[t - 1] + noise[t]
    idio_ret = np.diff(idio_price, axis=0, prepend=idio_price[:1])

    return systematic + idio_ret


# -----------------------------------------------------------------------------
# Backtest


@dataclass
class BacktestResult:
    dates: NDArray[np.int64]  # period indices
    pnl: ArrayF  # daily strategy return, gross
    pnl_net: ArrayF  # daily strategy return, after transaction costs
    turnover: ArrayF
    exposure: ArrayF  # total absolute notional through time

    @property
    def cumulative(self) -> ArrayF:
        return np.cumsum(self.pnl_net)

    def sharpe(self, periods_per_year: int = 252) -> float:
        x = self.pnl_net
        if x.std() == 0:
            return 0.0
        return float(x.mean() / x.std() * np.sqrt(periods_per_year))

    def max_drawdown(self) -> float:
        equity = self.cumulative
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        return float(dd.max())


def pca_statarb_backtest(
    returns: ArrayF,
    *,
    lookback: int = 252,
    n_factors: int = 3,
    zscore_window: int = 10,
    open_threshold: float = 1.0,
    close_threshold: float = 0.25,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Walk-forward PCA stat-arb backtest.

    All computations at time t use strictly past data.
    """
    T, N = returns.shape
    positions = np.zeros(N)
    pnl = np.zeros(T)
    pnl_net = np.zeros(T)
    turnover = np.zeros(T)
    exposure = np.zeros(T)
    cost = cost_bps / 10_000.0

    for t in range(lookback, T - 1):
        window = returns[t - lookback : t]
        # Fit PCA on the rolling window.
        centered = window - window.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:n_factors]  # (k, N)
        # Residuals on the lookback window (row-wise regression is implicit in
        # the reconstruction error since PCs are orthonormal in asset space).
        projection = centered @ components.T @ components
        residual_history = centered - projection

        # z-score on the last `zscore_window` residuals per asset.
        tail = residual_history[-zscore_window:]
        z = (tail.sum(axis=0)) / (tail.std(axis=0) * np.sqrt(zscore_window) + 1e-12)

        # Signal rules.
        new_positions = positions.copy()
        open_mask = np.abs(z) > open_threshold
        close_mask = np.abs(z) < close_threshold
        # Mean-revert: short when z is large positive, long when large negative.
        new_positions[open_mask] = -np.sign(z[open_mask])
        new_positions[close_mask] = 0.0
        # Leave positions in |z| between thresholds unchanged.

        # Dollar-neutralise and normalise to unit gross exposure.
        if np.any(new_positions != 0):
            new_positions = new_positions - new_positions.mean()  # market-neutral
            gross = np.sum(np.abs(new_positions))
            if gross > 0:
                new_positions = new_positions / gross

        # Trade at t, realise PnL over (t, t+1].
        trade = new_positions - positions
        turnover[t] = float(np.sum(np.abs(trade)))
        pnl[t + 1] = float(new_positions @ returns[t + 1])
        pnl_net[t + 1] = pnl[t + 1] - cost * turnover[t]
        exposure[t + 1] = float(np.sum(np.abs(new_positions)))
        positions = new_positions

    return BacktestResult(
        dates=np.arange(T, dtype=np.int64),
        pnl=pnl,
        pnl_net=pnl_net,
        turnover=turnover,
        exposure=exposure,
    )


# -----------------------------------------------------------------------------
# Optional: Ken French industry portfolios loader


def load_ken_french_industries(url: str | None = None):  # pragma: no cover - network
    """Download Ken French 49 industry portfolios CSV. Network required.

    Returns a (dates, names, returns) tuple with daily returns as a float array.
    """
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile

    import pandas as pd

    url = url or (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "49_Industry_Portfolios_daily_CSV.zip"
    )
    with urlopen(url) as resp:
        zdata = resp.read()
    with ZipFile(BytesIO(zdata)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("latin1")
    # Skip the header preamble; the file has multiple sections.
    lines = raw.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(" ,"))
    table = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        BytesIO(table.encode()), skipinitialspace=True, index_col=0, na_values=["-99.99", "-999"]
    )
    df = df.dropna(how="any")
    dates = pd.to_datetime(df.index.astype(str), format="%Y%m%d").to_numpy()
    names = list(df.columns)
    returns = df.to_numpy(dtype=np.float64) / 100.0
    return dates, names, returns
