"""Reproducibility entry point: regularisation path + small benchmark.

Runs on CPU in well under a minute. Produces two files in the same directory:
  * regularisation_path.png
  * report.md  (tiny markdown summary of the comparison with scikit-learn)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from linreg import fit_lasso, fit_ols_closed_form, fit_ridge, fit_sgd

HERE = Path(__file__).resolve().parent


def _make_sparse_problem(
    n: int = 400, p: int = 40, n_informative: int = 5, noise: float = 0.3, seed: int = 0
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[:n_informative] = rng.uniform(1.0, 3.0, size=n_informative) * rng.choice(
        [-1.0, 1.0], size=n_informative
    )
    y = X @ beta_true + noise * rng.standard_normal(n)
    return X, y, beta_true


def regularisation_path(X, y, alphas):
    ridge_coefs = np.stack([fit_ridge(X, y, alpha=a).coef for a in alphas])
    lasso_coefs = np.stack([fit_lasso(X, y, alpha=a, max_iter=500).coef for a in alphas])
    return ridge_coefs, lasso_coefs


def _plot(alphas, ridge_coefs, lasso_coefs, beta_true, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, coefs, title in zip(axes, (ridge_coefs, lasso_coefs), ("Ridge", "Lasso"), strict=True):
        for j in range(coefs.shape[1]):
            ax.plot(alphas, coefs[:, j], lw=0.8, color="C0" if beta_true[j] != 0 else "0.7")
        ax.set_xscale("log")
        ax.set_xlabel("α")
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.4)
    axes[0].set_ylabel("coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _sklearn_comparison(X, y) -> str:
    try:
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
    except ImportError:
        return "scikit-learn not installed; skipping comparison."

    ours_ols = fit_ols_closed_form(X, y)
    ours_ridge = fit_ridge(X, y, alpha=1.0)
    ours_lasso = fit_lasso(X, y, alpha=0.1, max_iter=5000, tol=1e-8)

    sk_ols = LinearRegression().fit(X, y)
    sk_ridge = Ridge(alpha=1.0).fit(X, y)
    sk_lasso = Lasso(alpha=0.1, max_iter=10_000, tol=1e-8).fit(X, y)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    lines = [
        "| Model | ||β_ours − β_sk||₂ | |b_ours − b_sk| |",
        "|---|---|---|",
        f"| OLS | {np.linalg.norm(ours_ols.coef - sk_ols.coef_):.3e} | "
        f"{abs(ours_ols.intercept - sk_ols.intercept_):.3e} |",
        f"| Ridge | {np.linalg.norm(ours_ridge.coef - sk_ridge.coef_):.3e} | "
        f"{abs(ours_ridge.intercept - sk_ridge.intercept_):.3e} |",
        f"| Lasso | {np.linalg.norm(ours_lasso.coef - sk_lasso.coef_):.3e} | "
        f"{abs(ours_lasso.intercept - sk_lasso.intercept_):.3e} |",
        "",
        "All three should be ≤ 10⁻³ on this synthetic problem.",
    ]
    return "\n".join(lines)


def main() -> None:
    X, y, beta_true = _make_sparse_problem()
    alphas = np.logspace(-3, 2, 30)
    ridge_coefs, lasso_coefs = regularisation_path(X, y, alphas)
    _plot(alphas, ridge_coefs, lasso_coefs, beta_true, HERE / "regularisation_path.png")

    sgd_model = fit_sgd(X, y, lr=1e-2, n_epochs=20, batch_size=64)

    report_lines = [
        "# Week 2 — NumPy linear-models mini-library",
        "",
        "## Comparison with scikit-learn",
        "",
        _sklearn_comparison(X, y),
        "",
        "## SGD training loss",
        "",
        f"first-epoch MSE: {sgd_model.history[0]:.4f}",
        f"last-epoch MSE:  {sgd_model.history[-1]:.4f}",
        "",
        "## Support recovery (lasso, α = 0.1)",
        "",
    ]
    lasso = fit_lasso(X, y, alpha=0.1, max_iter=5000, tol=1e-8)
    support_hat = np.flatnonzero(np.abs(lasso.coef) > 1e-3)
    support_true = np.flatnonzero(beta_true != 0)
    report_lines.append(f"true support: {list(support_true)}")
    report_lines.append(f"recovered:    {list(support_hat)}")

    (HERE / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("Wrote:", HERE / "regularisation_path.png")
    print("Wrote:", HERE / "report.md")


if __name__ == "__main__":
    main()
