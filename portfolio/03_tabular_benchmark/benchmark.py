"""Tabular-classifier benchmark on UCI Adult (income >50k).

Compares logistic regression, random forest, gradient boosting, XGBoost (if
installed), and LightGBM (if installed). Reports ROC AUC, PR AUC, Brier score,
and wall-clock fit time.

Output files (in this directory):
  * results.md — a small markdown table
  * roc_pr_calibration.png — ROC, PR, and reliability diagrams

Dataset fetch uses the OpenML mirror bundled with scikit-learn, so it runs
without network access after the first cache.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


@dataclass
class Result:
    name: str
    fit_seconds: float
    roc_auc: float
    pr_auc: float
    brier: float


def _load_adult():
    from sklearn.datasets import fetch_openml

    data = fetch_openml("adult", version=2, as_frame=True, parser="liac-arff")
    X = data.data.copy()
    y = (data.target == ">50K").astype(int).to_numpy()

    # Simple pipeline: label-encode categoricals.
    import pandas as pd

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.astype(np.float64).to_numpy()
    return X, y


def _score(y_true, y_score) -> tuple[float, float, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
        float(brier_score_loss(y_true, y_score)),
    )


def _fit_predict(model, X_train, y_train, X_test) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
    return scores, elapsed


def _plot(y_test, score_map: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        axes[0].plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        axes[1].plot(rec, prec, label=name)
        frac_pos, mean_pred = calibration_curve(y_test, scores, n_bins=10, strategy="quantile")
        axes[2].plot(mean_pred, frac_pos, marker="o", label=name)

    axes[0].plot([0, 1], [0, 1], ls=":", color="k")
    axes[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    axes[1].set(title="PR", xlabel="recall", ylabel="precision")
    axes[2].plot([0, 1], [0, 1], ls=":", color="k")
    axes[2].set(title="Calibration", xlabel="predicted", ylabel="observed")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run() -> list[Result]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = _load_adult()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    models = [
        ("logistic", LogisticRegression(max_iter=500)),
        ("random_forest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
        (
            "sklearn_gbdt",
            GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=0,
                    eval_metric="logloss",
                    verbosity=0,
                ),
            )
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=300,
                    num_leaves=31,
                    learning_rate=0.1,
                    random_state=0,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )
        )
    except ImportError:
        pass

    # Scale only for logistic regression; others are tree-based and scale-invariant.
    scaler = StandardScaler().fit(X_train)
    score_map: dict[str, np.ndarray] = {}
    results: list[Result] = []
    for name, model in models:
        X_tr = scaler.transform(X_train) if name == "logistic" else X_train
        X_te = scaler.transform(X_test) if name == "logistic" else X_test
        scores, dt = _fit_predict(model, X_tr, y_train, X_te)
        roc, pr, brier = _score(y_test, scores)
        results.append(Result(name=name, fit_seconds=dt, roc_auc=roc, pr_auc=pr, brier=brier))
        score_map[name] = scores

    _plot(y_test, score_map, HERE / "roc_pr_calibration.png")
    return results


def write_report(results: list[Result]) -> Path:
    lines = [
        "# Tabular benchmark — UCI Adult (>50K)",
        "",
        "| Model | ROC AUC | PR AUC | Brier | Fit (s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.brier:.4f} | {r.fit_seconds:.2f} |"
        )
    path = HERE / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    results = run()
    path = write_report(results)
    for r in results:
        print(
            f"{r.name:14s}  ROC={r.roc_auc:.4f}  PR={r.pr_auc:.4f}  "
            f"Brier={r.brier:.4f}  fit={r.fit_seconds:.2f}s"
        )
    print("wrote:", path)
    print("wrote:", HERE / "roc_pr_calibration.png")


if __name__ == "__main__":
    sys.exit(main())
