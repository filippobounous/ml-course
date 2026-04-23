"""Compact UNet-DDPM on FashionMNIST / MNIST, with optional class conditioning
and classifier-free guidance (Ho & Salimans, 2022).

~5M-parameter UNet with timestep + class embeddings. Trains to plausible
samples in ~1-3 hours on MPS, longer on CPU. Supports:

  * unconditional DDPM (pass `num_classes=None` to `SmallUNet`),
  * conditional DDPM with random label dropout during training,
  * DDPM (stochastic) and DDIM (deterministic, `eta=0`) samplers,
  * classifier-free guidance (`guidance_scale > 0`).

Setting `guidance_scale=0.0` on a conditional model returns the pure
conditional sampler; `guidance_scale > 0` extrapolates toward the
class-specific mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "This module requires PyTorch. Install with `pip install -e '.[dl,diffusion,ops]'`."
    ) from e


# -----------------------------------------------------------------------------
# Schedules


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @staticmethod
    def linear(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> DiffusionSchedule:
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return DiffusionSchedule(betas, alphas, alpha_bars)


# -----------------------------------------------------------------------------
# Sinusoidal timestep embedding


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# -----------------------------------------------------------------------------
# UNet building blocks


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(self.act(temb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """28×28-sized UNet for MNIST / FashionMNIST (1 channel).

    Optional class conditioning: pass `num_classes=K` to enable a class
    embedding with K+1 entries (the last is reserved as the "null" class
    used during CFG dropout and for the unconditional branch at sampling).
    """

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        t_dim: int = 128,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim)
        )
        if num_classes is not None:
            # +1 slot for the null/unconditional class.
            self.class_emb: nn.Embedding | None = nn.Embedding(num_classes + 1, t_dim)
            self.null_class_id = num_classes
        else:
            self.class_emb = None
            self.null_class_id = 0
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = ResBlock(base, base, t_dim)
        self.down2 = ResBlock(base, base * 2, t_dim)
        self.mid = ResBlock(base * 2, base * 2, t_dim)
        self.up2 = ResBlock(base * 4, base, t_dim)  # skip-concat doubles channels
        self.up1 = ResBlock(base * 2, base, t_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.act = nn.SiLU()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.t_dim))
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.null_class_id, device=x.device, dtype=torch.long)
            temb = temb + self.class_emb(y)
        h = self.in_conv(x)
        d1 = self.down1(h, temb)
        d2 = self.down2(self.pool(d1), temb)
        m = self.mid(self.pool(d2), temb)
        u2 = self.up2(torch.cat([self.upsample(m), d2], dim=1), temb)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), temb)
        return self.out_conv(self.act(self.out_norm(u1)))


# -----------------------------------------------------------------------------
# DDPM training + sampling


def ddpm_loss(
    model: nn.Module,
    x0: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    y: torch.Tensor | None = None,
    p_drop: float = 0.1,
) -> torch.Tensor:
    """DDPM ε-prediction loss. With a conditional model pass `y` (class labels);
    with probability `p_drop` each sample's label is replaced by the null class
    so the same network learns both the conditional and unconditional scores
    (Ho & Salimans, 2022).
    """
    B = x0.shape[0]
    t = torch.randint(0, len(schedule.alpha_bars), (B,), device=x0.device)
    noise = torch.randn_like(x0)
    ab = schedule.alpha_bars.to(x0.device)[t][:, None, None, None]
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    y_dropped = y
    null_id = getattr(model, "null_class_id", 0)
    if y is not None and getattr(model, "class_emb", None) is not None and p_drop > 0:
        drop_mask = torch.rand(B, device=x0.device) < p_drop
        y_dropped = torch.where(drop_mask, torch.full_like(y, null_id), y)

    pred = model(x_t, t, y_dropped) if y_dropped is not None else model(x_t, t)
    return nn.functional.mse_loss(pred, noise)


def _cfg_epsilon(
    model: nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    *,
    class_label: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    """ε-prediction with classifier-free guidance.

    At `guidance_scale=0.0` this returns the pure conditional prediction;
    at `guidance_scale > 0` it extrapolates away from the unconditional
    prediction: `eps = (1+w) eps_cond − w eps_uncond`.

    If the model is unconditional (`class_emb is None`) the two branches
    coincide and guidance is a no-op.
    """
    is_conditional = getattr(model, "class_emb", None) is not None
    if not is_conditional or class_label is None or guidance_scale <= 0.0:
        return model(x, t_batch, class_label) if is_conditional else model(x, t_batch)

    null_id = model.null_class_id
    y_null = torch.full_like(class_label, null_id)
    eps_cond = model(x, t_batch, class_label)
    eps_uncond = model(x, t_batch, y_null)
    return (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    shape: tuple[int, ...],
    schedule: DiffusionSchedule,
    *,
    device: str = "cpu",
    seed: int | None = None,
    class_label: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    T = len(schedule.alpha_bars)
    for t in range(T - 1, -1, -1):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = _cfg_epsilon(
            model, x, t_batch, class_label=class_label, guidance_scale=guidance_scale
        )
        beta = schedule.betas[t].to(device)
        alpha = schedule.alphas[t].to(device)
        ab = schedule.alpha_bars[t].to(device)
        mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ab)) * eps)
        x = mean + torch.sqrt(beta) * torch.randn_like(x) if t > 0 else mean
    return x


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    shape: tuple[int, ...],
    schedule: DiffusionSchedule,
    *,
    n_steps: int = 50,
    device: str = "cpu",
    eta: float = 0.0,
    seed: int | None = None,
    class_label: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    T = len(schedule.alpha_bars)
    indices = torch.linspace(0, T - 1, n_steps, dtype=torch.long)
    for i in range(len(indices) - 1, 0, -1):
        t = int(indices[i])
        t_prev = int(indices[i - 1])
        ab_t = schedule.alpha_bars[t].to(device)
        ab_prev = schedule.alpha_bars[t_prev].to(device)
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = _cfg_epsilon(
            model, x, t_batch, class_label=class_label, guidance_scale=guidance_scale
        )
        x0_hat = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        sigma = (
            eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            if eta > 0
            else torch.zeros((), device=device)
        )
        dir_term = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0)) * eps
        noise = torch.randn_like(x) if eta > 0 else 0.0
        x = torch.sqrt(ab_prev) * x0_hat + dir_term + sigma * noise
    t = int(indices[0])
    t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
    eps = _cfg_epsilon(model, x, t_batch, class_label=class_label, guidance_scale=guidance_scale)
    ab_t = schedule.alpha_bars[t].to(device)
    return (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
