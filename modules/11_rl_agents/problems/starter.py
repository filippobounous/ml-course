"""Week 11 — problem-set starter. **This is the file you edit.**

`pytest tests/week_11` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_11
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Value iteration on a finite MDP


def value_iteration(
    P: ArrayF, R: ArrayF, *, gamma: float = 0.99, tol: float = 1e-08, max_iter: int = 10000
) -> tuple[ArrayF, ArrayF, int]:
    """Value iteration on a finite MDP.

    P: (S, A, S) transition probabilities.
    R: (S, A) expected immediate reward.
    Returns (V*, optimal greedy policy π*(s) as an int array, iterations).
    """
    raise NotImplementedError("value_iteration")


def bellman_contraction_factor(V: ArrayF, W: ArrayF, TV: ArrayF, TW: ArrayF) -> float:
    """Return the empirical contraction factor ||TV − TW||∞ / ||V − W||∞.

    Proof scaffolding: for the optimality operator this is ≤ γ for any V, W
    (Banach). Useful as a test-time check.
    """
    raise NotImplementedError("bellman_contraction_factor")


# -----------------------------------------------------------------------------
# GAE


def compute_gae(
    rewards: ArrayF,
    values: ArrayF,
    dones: ArrayF,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[ArrayF, ArrayF]:
    """Generalised Advantage Estimation (Schulman 2015).

    `rewards`, `values`, `dones` are 1-D arrays of shape (T,). `values` is the
    critic's estimate at each timestep; `last_value` is V(s_T). Returns
    (advantages, returns).
    """
    raise NotImplementedError("compute_gae")


# -----------------------------------------------------------------------------
# PPO-clip loss (NumPy reference)


def ppo_clip_loss(
    log_probs: ArrayF, old_log_probs: ArrayF, advantages: ArrayF, *, clip_eps: float = 0.2
) -> float:
    """PPO's clipped surrogate objective (to *maximise*), returned as a scalar loss (to minimise)."""
    raise NotImplementedError("ppo_clip_loss")


# -----------------------------------------------------------------------------
# Example tiny MDP for tests


def tiny_chain_mdp(n: int = 5) -> tuple[ArrayF, ArrayF]:
    """n-state chain: action 0 = left, action 1 = right. Reward +1 at the right end."""
    raise NotImplementedError("tiny_chain_mdp")
