"""PINN for Burgers' equation (Raissi et al. 2019 Fig. 2 reproduction).

Trains a small MLP u_θ(x, t) against three losses:
  * PDE residual at collocation points,
  * initial condition u(x, 0) = −sin(π x),
  * boundary conditions u(±1, t) = 0.

Uses autograd for u_x, u_xx, u_t — the PDE is written out literally.

Torch is required (`pip install -e '.[dl,sciml,ops]'`).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PINN requires PyTorch. Install with `pip install -e '.[dl,sciml,ops]'`."
    ) from e


@dataclass
class PINNConfig:
    n_collocation: int = 10_000
    n_ic: int = 200
    n_bc: int = 200
    hidden: int = 64
    depth: int = 6
    lr: float = 1e-3
    max_iters: int = 4_000
    nu: float = 0.01 / 3.14159265
    # Loss-weighting scheme: "fixed" uses (lambda_res, lambda_ic, lambda_bc);
    # "gradnorm" adapts the weights every `reweight_every` steps to equalise
    # per-loss gradient norms (Wang, Yu, Perdikaris 2022 §4, simplified).
    loss_weighting: str = "fixed"
    lambda_res: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    reweight_every: int = 100
    reweight_alpha: float = 0.9  # EMA smoothing on the new weights
    seed: int = 0


class PINN(nn.Module):
    """MLP with tanh activations — typical for PINNs (smooth gradients)."""

    def __init__(self, hidden: int = 64, depth: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


def pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    """r = u_t + u u_x - ν u_xx."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def sample_points(cfg: PINNConfig, device: str) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    # Collocation: uniform in [-1, 1] x [0, 1].
    x_r = torch.rand(cfg.n_collocation, 1, generator=g) * 2.0 - 1.0
    t_r = torch.rand(cfg.n_collocation, 1, generator=g)
    # Initial condition at t = 0.
    x_0 = torch.rand(cfg.n_ic, 1, generator=g) * 2.0 - 1.0
    t_0 = torch.zeros(cfg.n_ic, 1)
    u_0 = -torch.sin(3.14159265 * x_0)
    # Boundary conditions x = ±1, t ∈ [0, 1]. Half each.
    half = cfg.n_bc // 2
    t_b = torch.rand(cfg.n_bc, 1, generator=g)
    x_b = torch.cat([torch.full((half, 1), -1.0), torch.full((cfg.n_bc - half, 1), 1.0)])
    u_b = torch.zeros(cfg.n_bc, 1)
    return {
        "x_r": x_r.to(device),
        "t_r": t_r.to(device),
        "x_0": x_0.to(device),
        "t_0": t_0.to(device),
        "u_0": u_0.to(device),
        "x_b": x_b.to(device),
        "t_b": t_b.to(device),
        "u_b": u_b.to(device),
    }


class GradNormReweighter:
    """Adaptive PINN loss-weighter (Wang, Yu, Perdikaris 2022, simplified).

    Given named loss tensors, compute the mean gradient-norm of each w.r.t.
    the model parameters, and set each loss's weight so the **weighted**
    gradient norms are equal. EMA-smoothed to stabilise training.
    """

    def __init__(self, names: list[str], alpha: float = 0.9) -> None:
        self.names = list(names)
        self.alpha = alpha
        self.weights = dict.fromkeys(self.names, 1.0)

    def step(
        self, losses: dict[str, torch.Tensor], params: list[torch.Tensor]
    ) -> dict[str, float]:
        grad_norms: dict[str, float] = {}
        for name, loss in losses.items():
            grads = torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True, create_graph=False
            )
            norm = 0.0
            for g in grads:
                if g is not None:
                    norm += float((g**2).sum())
            grad_norms[name] = norm**0.5 + 1e-12
        mean_gn = sum(grad_norms.values()) / len(grad_norms)
        for name in self.names:
            target = mean_gn / grad_norms[name]
            self.weights[name] = self.alpha * self.weights[name] + (1.0 - self.alpha) * target
        return dict(self.weights)


def train(cfg: PINNConfig | None = None, device: str = "cpu") -> dict:
    cfg = cfg or PINNConfig()
    torch.manual_seed(cfg.seed)
    model = PINN(cfg.hidden, cfg.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    pts = sample_points(cfg, device)
    history: list[dict[str, float]] = []

    reweighter: GradNormReweighter | None = None
    lambdas = {"res": cfg.lambda_res, "ic": cfg.lambda_ic, "bc": cfg.lambda_bc}
    if cfg.loss_weighting == "gradnorm":
        reweighter = GradNormReweighter(list(lambdas.keys()), alpha=cfg.reweight_alpha)

    for step in range(1, cfg.max_iters + 1):
        # Residual loss.
        r = pde_residual(model, pts["x_r"], pts["t_r"], cfg.nu)
        loss_r = (r**2).mean()
        # IC loss.
        pred_0 = model(pts["x_0"], pts["t_0"])
        loss_ic = ((pred_0 - pts["u_0"]) ** 2).mean()
        # BC loss.
        pred_b = model(pts["x_b"], pts["t_b"])
        loss_bc = ((pred_b - pts["u_b"]) ** 2).mean()

        if reweighter is not None and step % cfg.reweight_every == 0:
            lambdas = reweighter.step(
                {"res": loss_r, "ic": loss_ic, "bc": loss_bc},
                list(model.parameters()),
            )

        loss = lambdas["res"] * loss_r + lambdas["ic"] * loss_ic + lambdas["bc"] * loss_bc
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "loss_r": float(loss_r),
                    "loss_ic": float(loss_ic),
                    "loss_bc": float(loss_bc),
                    "lambda_res": float(lambdas["res"]),
                    "lambda_ic": float(lambdas["ic"]),
                    "lambda_bc": float(lambdas["bc"]),
                }
            )

    return {"model_state": model.state_dict(), "history": history, "config": cfg.__dict__}
