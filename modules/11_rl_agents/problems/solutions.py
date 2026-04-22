"""Week 11 — reference solutions.

NumPy-only utilities: tabular value iteration, GAE, PPO-clip NumPy reference,
Bellman-contraction demo.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Value iteration on a finite MDP


def value_iteration(
    P: ArrayF, R: ArrayF, *, gamma: float = 0.99, tol: float = 1e-8, max_iter: int = 10_000
) -> tuple[ArrayF, ArrayF, int]:
    """Value iteration on a finite MDP.

    P: (S, A, S) transition probabilities.
    R: (S, A) expected immediate reward.
    Returns (V*, optimal greedy policy π*(s) as an int array, iterations).
    """
    S = P.shape[0]
    V = np.zeros(S, dtype=np.float64)
    t = 0
    for t in range(1, max_iter + 1):  # noqa: B007 — `t` is returned as iteration count
        Q = R + gamma * np.einsum("sap,p->sa", P, V)  # (S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = Q.argmax(axis=1).astype(np.int64)
    return V, policy, t


def bellman_contraction_factor(V: ArrayF, W: ArrayF, TV: ArrayF, TW: ArrayF) -> float:
    """Return the empirical contraction factor ||TV − TW||∞ / ||V − W||∞.

    Proof scaffolding: for the optimality operator this is ≤ γ for any V, W
    (Banach). Useful as a test-time check.
    """
    denom = np.max(np.abs(V - W))
    if denom == 0:
        return 0.0
    return float(np.max(np.abs(TV - TW)) / denom)


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
    T = rewards.shape[0]
    adv = np.zeros(T, dtype=np.float64)
    running = 0.0
    next_value = last_value
    for t in range(T - 1, -1, -1):
        mask = 1.0 - dones[t]  # 0 if this timestep was terminal
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running = delta + gamma * lam * mask * running
        adv[t] = running
        next_value = values[t]
    returns = adv + values
    return adv, returns


# -----------------------------------------------------------------------------
# PPO-clip loss (NumPy reference)


def ppo_clip_loss(
    log_probs: ArrayF,
    old_log_probs: ArrayF,
    advantages: ArrayF,
    *,
    clip_eps: float = 0.2,
) -> float:
    """PPO's clipped surrogate objective (to *maximise*), returned as a scalar loss (to minimise)."""
    ratio = np.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    # Maximise the min; equivalently minimise −min.
    return float(-np.minimum(unclipped, clipped).mean())


# -----------------------------------------------------------------------------
# Example tiny MDP for tests


def tiny_chain_mdp(n: int = 5) -> tuple[ArrayF, ArrayF]:
    """n-state chain: action 0 = left, action 1 = right. Reward +1 at the right end."""
    P = np.zeros((n, 2, n), dtype=np.float64)
    R = np.zeros((n, 2), dtype=np.float64)
    for s in range(n):
        left = max(s - 1, 0)
        right = min(s + 1, n - 1)
        P[s, 0, left] = 1.0
        P[s, 1, right] = 1.0
        # Reward depends only on arrival state (right end).
        if right == n - 1:
            R[s, 1] = 1.0
    return P, R


if __name__ == "__main__":
    P, R = tiny_chain_mdp(7)
    V, pi, iters = value_iteration(P, R, gamma=0.9)
    print(f"converged in {iters} iterations")
    print("V*:", V.round(3))
    print("π*:", pi)
