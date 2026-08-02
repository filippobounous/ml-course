"""Week 10 — reference solutions.

NumPy-only implementations of:
  * Forward DDPM noise schedules (linear, cosine) and the closed-form q(x_t|x_0).
  * DDIM deterministic sampling (η=0) for a known score function.
  * CLIP-style InfoNCE loss.
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
    alpha_bars: ArrayF  # cumulative product \bar α_t


def linear_schedule(T: int, *, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Schedule:
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return Schedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def cosine_schedule(T: int, *, s: float = 0.008) -> Schedule:
    """Nichol & Dhariwal (2021) cosine schedule."""
    steps = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = np.clip(1 - alpha_bars[1:] / alpha_bars[:-1], a_min=1e-8, a_max=0.999)
    alphas = 1.0 - betas
    # Recompute alpha_bars from the clipped alphas to keep them consistent.
    alpha_bars = np.cumprod(alphas)
    return Schedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def q_sample(x0: ArrayF, t: int, schedule: Schedule, noise: ArrayF) -> ArrayF:
    """x_t = sqrt(α̅_t) x_0 + sqrt(1 − α̅_t) ε."""
    ab = schedule.alpha_bars[t]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise


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
    rng = np.random.default_rng(seed)
    T = len(schedule.alpha_bars)
    # Uniformly-spaced subset of timesteps.
    indices = np.linspace(0, T - 1, n_steps, dtype=int)
    x = rng.standard_normal(shape)

    for i in range(len(indices) - 1, 0, -1):
        t = indices[i]
        t_prev = indices[i - 1]
        ab_t = schedule.alpha_bars[t]
        ab_prev = schedule.alpha_bars[t_prev]

        eps = score_fn(x, t)
        x0_hat = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)

        sigma_t = eta * np.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
        sigma_t = float(np.nan_to_num(sigma_t, nan=0.0))

        dir_term = np.sqrt(np.clip(1 - ab_prev - sigma_t**2, a_min=0.0, a_max=None)) * eps
        noise = rng.standard_normal(shape) if eta > 0 else np.zeros(shape)
        x = np.sqrt(ab_prev) * x0_hat + dir_term + sigma_t * noise

    # Final denoising step to t = 0.
    eps = score_fn(x, int(indices[0]))
    ab_0 = schedule.alpha_bars[indices[0]]
    x0_hat = (x - np.sqrt(1 - ab_0) * eps) / np.sqrt(ab_0)
    return x0_hat


# -----------------------------------------------------------------------------
# CLIP-style InfoNCE


def clip_infonce_loss(
    image_embeddings: ArrayF, text_embeddings: ArrayF, *, temperature: float = 0.07
) -> float:
    """Symmetric InfoNCE loss assuming embeddings are L2-normalised."""
    a = np.asarray(image_embeddings, dtype=np.float64)
    b = np.asarray(text_embeddings, dtype=np.float64)
    logits = a @ b.T / max(temperature, 1e-9)
    labels = np.arange(a.shape[0])
    loss_i2t = _cross_entropy_from_logits(logits, labels)
    loss_t2i = _cross_entropy_from_logits(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def _cross_entropy_from_logits(logits: ArrayF, labels: ArrayF) -> float:
    row_max = logits.max(axis=1, keepdims=True)
    shifted = logits - row_max
    log_z = np.log(np.exp(shifted).sum(axis=1)) + row_max.flatten()
    log_probs = logits[np.arange(logits.shape[0]), labels] - log_z
    return float(-log_probs.mean())


if __name__ == "__main__":
    sch = linear_schedule(1000)
    print("linear α̅[0], α̅[500], α̅[999]:", sch.alpha_bars[[0, 500, 999]])
    sch_cos = cosine_schedule(1000)
    print("cosine α̅[0], α̅[500], α̅[999]:", sch_cos.alpha_bars[[0, 500, 999]])

    # Trivial score function: pretend the score points back to the origin.
    def fake_score(x, _t):
        return x * 0.1

    sample = ddim_sample(fake_score, (1, 4), sch, n_steps=10, seed=0, eta=0.0)
    print("DDIM sample:", sample)

    # InfoNCE sanity — perfectly-matched identity embeddings → loss ≈ log(1/5) negated.
    eye = np.eye(5)
    print("InfoNCE (identity):", clip_infonce_loss(eye, eye))
