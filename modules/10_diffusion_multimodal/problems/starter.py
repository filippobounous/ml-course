"""Week 10 — problem-set starter. **This is the file you edit.**

`pytest tests/week_10` grades what is in here. Every function below raises
`NotImplementedError` until you implement it — that is the expected state of a
fresh clone, not a bug.

Signatures, docstrings and return types are given: they are the contract the
tests check against, so keep them. The reference implementation lives in
`_reference/solutions.py` — read it *after* you have your own version working,
or when you are genuinely stuck. To check that a failure is yours and not the
course's, run the same tests against the reference:

    MLCOURSE_SOLUTIONS=reference pytest tests/week_10
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Noise schedules


@dataclass(frozen=True)
class Schedule:
    betas: ArrayF
    alphas: ArrayF
    alpha_bars: ArrayF


def linear_schedule(T: int, *, beta_start: float = 0.0001, beta_end: float = 0.02) -> Schedule:
    raise NotImplementedError("linear_schedule")


def cosine_schedule(T: int, *, s: float = 0.008) -> Schedule:
    """Nichol & Dhariwal (2021) cosine schedule."""
    raise NotImplementedError("cosine_schedule")


def q_sample(x0: ArrayF, t: int, schedule: Schedule, noise: ArrayF) -> ArrayF:
    """x_t = sqrt(α̅_t) x_0 + sqrt(1 − α̅_t) ε."""
    raise NotImplementedError("q_sample")


# -----------------------------------------------------------------------------
# DDIM deterministic sampling


def ddim_sample(
    score_fn,
    shape: tuple[int, ...],
    schedule: Schedule,
    *,
    n_steps: int = 50,
    seed: int = 0,
    eta: float = 0.0,
) -> ArrayF:
    """DDIM sampler with an arbitrary number of inference steps.

    `score_fn(x, t_int)` should return an estimate of the noise ε_θ(x, t).
    `eta = 0` gives deterministic sampling (the probability-flow ODE limit).
    """
    raise NotImplementedError("ddim_sample")


# -----------------------------------------------------------------------------
# CLIP-style InfoNCE


def clip_infonce_loss(
    image_embeddings: ArrayF, text_embeddings: ArrayF, *, temperature: float = 0.07
) -> float:
    """Symmetric InfoNCE loss assuming embeddings are L2-normalised."""
    raise NotImplementedError("clip_infonce_loss")


def _cross_entropy_from_logits(logits: ArrayF, labels: ArrayF) -> float:
    raise NotImplementedError("_cross_entropy_from_logits")
