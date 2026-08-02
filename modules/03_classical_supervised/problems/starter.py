"""Week 03 — problem-set starter. **This is the file you edit.**

`pytest tests/week_03` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_03
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# IRLS logistic regression


@dataclass
class LogisticIRLS:
    coef: ArrayF
    intercept: float
    n_iter: int
    history: list[float]

    def predict_proba(self, X: ArrayF) -> ArrayF:
        raise NotImplementedError("predict_proba")

    def predict(self, X: ArrayF, threshold: float = 0.5) -> ArrayF:
        raise NotImplementedError("predict")


def _sigmoid(z: ArrayF) -> ArrayF:
    raise NotImplementedError("_sigmoid")


def fit_logistic_irls(
    X: ArrayF, y: ArrayF, *, alpha: float = 0.0, max_iter: int = 50, tol: float = 1e-08
) -> LogisticIRLS:
    """Logistic regression via IRLS (Newton) on the augmented matrix.

    Minimises
        -Σ [y_i log σ(xᵀβ + b) + (1-y_i) log(1-σ(xᵀβ + b))] + (α/2) ||β||²
    with an unpenalised intercept. Quadratic convergence once near the optimum.
    """
    raise NotImplementedError("fit_logistic_irls")


# -----------------------------------------------------------------------------
# Information gain helpers (for decision-tree intuition)


def gini(p: ArrayF) -> float:
    raise NotImplementedError("gini")


def entropy(p: ArrayF, eps: float = 1e-12) -> float:
    raise NotImplementedError("entropy")


def information_gain(parent_probs: ArrayF, child_sets: list[tuple[float, ArrayF]]) -> float:
    """IG = H(parent) − Σ w_i H(child_i).

    `child_sets` is a list of (weight, class-probability-vector) tuples; the
    weights should sum to 1.
    """
    raise NotImplementedError("information_gain")
