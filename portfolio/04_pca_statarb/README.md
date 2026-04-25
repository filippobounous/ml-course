# 04 — PCA statistical arbitrage

Walk-forward PCA-based residual stat-arb, following the Avellaneda & Lee (2008)
recipe:

1. Rolling PCA over the last `lookback` days of cross-sectional returns.
2. Residualise each asset against the top-$k$ PCs.
3. Z-score the rolling-sum residual; open a mean-reversion position when the
   z-score crosses `open_threshold`; close when it falls below `close_threshold`.
4. Dollar-neutralise across assets and normalise to unit gross exposure.
5. Compute PnL with a `cost_bps` transaction-cost penalty on turnover.

All computations at time $t$ use **strictly past data** — walk-forward, no
look-ahead.

## Layout

```
portfolio/04_pca_statarb/
├── pca_statarb.py   ← backtest engine + simulated returns generator
├── demo.py          ← IS/OOS split, plots, results.md
└── README.md
```

## Reproduce

```bash
python portfolio/04_pca_statarb/demo.py
```

Runs offline in ~10 seconds on CPU using simulated returns.

## Live data

`pca_statarb.load_ken_french_industries()` fetches the Ken French 49-industry
daily portfolios (needs network; no API key). Replace the simulated returns
in `demo.py` with that dataset to rerun on real data.

## Tests

`tests/week_04/` covers: (a) GMM-EM log-likelihood monotonicity, (b) PCA
eigenvector recovery, (c) k-means convergence, (d) backtest no-look-ahead
invariants.

## What I learned

*To be filled in after completing Week 4.*
