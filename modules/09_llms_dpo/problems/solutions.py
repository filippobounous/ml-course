"""Week 9 — reference solutions.

Torch-free NumPy reference for the DPO loss (makes the derivation concrete
and testable); Chinchilla-style compute / optimal-token helpers; LoRA
parameter-count math.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# DPO loss (NumPy reference)


def dpo_loss(
    logp_policy_chosen: ArrayF,
    logp_policy_rejected: ArrayF,
    logp_ref_chosen: ArrayF,
    logp_ref_rejected: ArrayF,
    *,
    beta: float = 0.1,
) -> tuple[float, float]:
    """Direct-Preference-Optimization loss (Rafailov 2023).

    Inputs are **sum** log-probabilities (over response tokens) for the policy
    model π and the frozen reference π_ref, on the chosen and rejected
    responses respectively. All arrays are shape (N,).

    Returns:
        mean_loss, mean_accuracy
    where accuracy is P(π_θ prefers chosen over rejected under the DPO margin).
    """
    logp_policy_chosen = np.asarray(logp_policy_chosen, dtype=np.float64)
    logp_policy_rejected = np.asarray(logp_policy_rejected, dtype=np.float64)
    logp_ref_chosen = np.asarray(logp_ref_chosen, dtype=np.float64)
    logp_ref_rejected = np.asarray(logp_ref_rejected, dtype=np.float64)

    # Policy log-ratios: implicit reward per response.
    chosen_rewards = beta * (logp_policy_chosen - logp_ref_chosen)
    rejected_rewards = beta * (logp_policy_rejected - logp_ref_rejected)
    margins = chosen_rewards - rejected_rewards

    # NLL of σ(margin) is -log σ(margin) = log(1 + exp(-margin)).
    # Use the softplus-style stable form log1p(exp(-|m|)) + max(-m, 0).
    loss = np.log1p(np.exp(-np.abs(margins))) + np.maximum(-margins, 0.0)
    accuracy = (margins > 0).mean()
    return float(loss.mean()), float(accuracy)


def dpo_reward_margin(
    logp_policy_chosen: ArrayF,
    logp_policy_rejected: ArrayF,
    logp_ref_chosen: ArrayF,
    logp_ref_rejected: ArrayF,
    *,
    beta: float = 0.1,
) -> ArrayF:
    """Per-example reward margin (β * log ratio − β * log ratio)."""
    return beta * (
        (logp_policy_chosen - logp_ref_chosen) - (logp_policy_rejected - logp_ref_rejected)
    )


# -----------------------------------------------------------------------------
# LoRA parameter counting


def lora_param_count(d_in: int, d_out: int, rank: int) -> int:
    """Trainable parameters in a LoRA adapter A: r×d_in, B: d_out×r."""
    return rank * (d_in + d_out)


def lora_param_reduction(d: int, rank: int) -> float:
    """Fractional reduction vs. a fully fine-tuned d×d linear."""
    return lora_param_count(d, d, rank) / (d * d)


# -----------------------------------------------------------------------------
# Chinchilla scaling helpers


def chinchilla_optimal_tokens(n_params: float, ratio: float = 20.0) -> float:
    """Compute-optimal tokens for an N-parameter model under Chinchilla's 20:1."""
    return ratio * n_params


def chinchilla_flops(n_params: float, n_tokens: float) -> float:
    """Approximate training FLOPs (the 6ND rule of thumb)."""
    return 6.0 * n_params * n_tokens


if __name__ == "__main__":
    # Sanity check the DPO loss: ties give exactly ln 2.
    n = 5
    zeros = np.zeros(n)
    loss, acc = dpo_loss(zeros, zeros, zeros, zeros)
    print(f"tie loss = {loss:.4f} (expected {np.log(2):.4f})")
    print(f"tie accuracy = {acc:.4f}")

    # Fully-confident correct preference: loss → 0.
    loss2, acc2 = dpo_loss(np.full(n, 10.0), np.full(n, -10.0), np.zeros(n), np.zeros(n), beta=1.0)
    print(f"correct-preference loss = {loss2:.6f}")
    print(f"correct-preference accuracy = {acc2:.4f}")

    # LoRA reduction at standard LLM widths.
    for d, r in [(4096, 16), (4096, 64), (1024, 8)]:
        print(f"d={d} r={r}: LoRA uses {lora_param_reduction(d, r) * 100:.2f}% of fine-tune params")
