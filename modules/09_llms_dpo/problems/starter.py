"""Week 09 — problem-set starter. **This is the file you edit.**

`pytest tests/week_09` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_09
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
    raise NotImplementedError("dpo_loss")


def dpo_reward_margin(
    logp_policy_chosen: ArrayF,
    logp_policy_rejected: ArrayF,
    logp_ref_chosen: ArrayF,
    logp_ref_rejected: ArrayF,
    *,
    beta: float = 0.1,
) -> ArrayF:
    """Per-example reward margin (β * log ratio − β * log ratio)."""
    raise NotImplementedError("dpo_reward_margin")


# -----------------------------------------------------------------------------
# LoRA parameter counting


def lora_param_count(d_in: int, d_out: int, rank: int) -> int:
    """Trainable parameters in a LoRA adapter A: r×d_in, B: d_out×r."""
    raise NotImplementedError("lora_param_count")


def lora_param_reduction(d: int, rank: int) -> float:
    """Fractional reduction vs. a fully fine-tuned d×d linear."""
    raise NotImplementedError("lora_param_reduction")


# -----------------------------------------------------------------------------
# Chinchilla scaling helpers


def chinchilla_optimal_tokens(n_params: float, ratio: float = 20.0) -> float:
    """Compute-optimal tokens for an N-parameter model under Chinchilla's 20:1."""
    raise NotImplementedError("chinchilla_optimal_tokens")


def chinchilla_flops(n_params: float, n_tokens: float) -> float:
    """Approximate training FLOPs (the 6ND rule of thumb)."""
    raise NotImplementedError("chinchilla_flops")
