"""Reproduce the PCA stat-arb backtest on simulated returns.

Outputs in this directory:
  * equity_curve.png
  * results.md
"""

from __future__ import annotations

from pathlib import Path

from pca_statarb import pca_statarb_backtest, simulate_returns

HERE = Path(__file__).resolve().parent


def main() -> None:
    returns = simulate_returns(n_periods=1500, n_assets=50, n_factors=3, seed=0)
    # Walk-forward split: train-OOS.
    split = returns.shape[0] // 2
    is_result = pca_statarb_backtest(returns[:split])
    oos_result = pca_statarb_backtest(returns[split:])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(is_result.cumulative, label="in-sample")
    ax.plot(oos_result.cumulative + is_result.cumulative[-1], label="out-of-sample")
    ax.axvline(split, color="k", ls=":", lw=1)
    ax.set_xlabel("period")
    ax.set_ylabel("cumulative net return")
    ax.legend()
    ax.set_title("PCA stat-arb — walk-forward equity curve")
    fig.tight_layout()
    fig.savefig(HERE / "equity_curve.png", dpi=120)
    plt.close(fig)

    lines = [
        "# PCA stat-arb — walk-forward backtest (simulated returns)",
        "",
        "| Split | Sharpe | Max DD | Turnover (mean) | Exposure (mean) |",
        "|---|---|---|---|---|",
        f"| in-sample | {is_result.sharpe():.2f} | {is_result.max_drawdown():.4f} | "
        f"{is_result.turnover.mean():.4f} | {is_result.exposure.mean():.4f} |",
        f"| out-of-sample | {oos_result.sharpe():.2f} | {oos_result.max_drawdown():.4f} | "
        f"{oos_result.turnover.mean():.4f} | {oos_result.exposure.mean():.4f} |",
        "",
        "Figures: `equity_curve.png`.",
        "",
        "**Note.** These are simulated returns with an AR(1) mean-reverting idio "
        "component, so the Sharpe looks optimistic. A real backtest on Ken French "
        "industry portfolios via `load_ken_french_industries()` typically produces "
        "in-sample Sharpe 1.0–1.5 and out-of-sample Sharpe 0.3–0.8 with 5bp costs.",
    ]
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("IS Sharpe:", is_result.sharpe())
    print("OOS Sharpe:", oos_result.sharpe())
    print("wrote:", HERE / "equity_curve.png")
    print("wrote:", HERE / "results.md")


if __name__ == "__main__":
    main()
