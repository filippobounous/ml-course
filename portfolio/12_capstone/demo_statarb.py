"""Walk-forward stat-arb backtest + TCA Sharpe reporting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK4 = HERE.parent / "04_pca_statarb" / "pca_statarb.py"


def _load_week4():
    spec = importlib.util.spec_from_file_location("w4_statarb", WEEK4)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from statarb_walkforward import walk_forward_statarb

    sa = _load_week4()
    returns = sa.simulate_returns(n_periods=2000, n_assets=40, n_factors=3, seed=0)
    report = walk_forward_statarb(returns, train_size=504, val_size=126, embargo=5, cost_bps=5.0)
    lines = [
        "# Week 12 — Walk-forward stat-arb backtest",
        "",
        f"Folds: {len(report.windows)}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Gross Sharpe | {report.gross_sharpe:.2f} |",
        f"| Net Sharpe (5bp) | {report.net_sharpe:.2f} |",
        f"| Max drawdown | {report.max_drawdown:.4f} |",
        f"| Mean turnover | {report.mean_turnover:.4f} |",
        "",
        "## Per-fold summary",
        "",
        "| Fold | Val period | Sharpe | Turnover |",
        "|---|---|---|---|",
        *[
            f"| {w['fold']} | [{w['val_start']}, {w['val_end']}) | "
            f"{w['sharpe']:.2f} | {w['turnover']:.4f} |"
            for w in report.per_window
        ],
    ]
    (HERE / "statarb_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Gross Sharpe: {report.gross_sharpe:.2f}")
    print(f"Net Sharpe:   {report.net_sharpe:.2f}")
    print(f"Max DD:       {report.max_drawdown:.4f}")
    print("wrote:", HERE / "statarb_results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
