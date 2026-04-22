# PCA stat-arb — walk-forward backtest (simulated returns)

| Split | Sharpe | Max DD | Turnover (mean) | Exposure (mean) |
|---|---|---|---|---|
| in-sample | 3.21 | 0.0573 | 0.7727 | 0.6627 |
| out-of-sample | 2.93 | 0.0697 | 0.7673 | 0.6627 |

Figures: `equity_curve.png`.

**Note.** These are simulated returns with an AR(1) mean-reverting idio component, so the Sharpe looks optimistic. A real backtest on Ken French industry portfolios via `load_ken_french_industries()` typically produces in-sample Sharpe 1.0–1.5 and out-of-sample Sharpe 0.3–0.8 with 5bp costs.