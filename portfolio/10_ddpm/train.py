"""Train the SmallUNet-DDPM on FashionMNIST via `mlcourse.Trainer`, driven by Hydra.

The DDPM objective doesn't fit the standard `(x, y) → loss` shape: each step
samples a random timestep, adds matching noise, predicts it. We wrap the
`SmallUNet` in a thin `DDPMLossModule` whose `forward(images, labels)` returns
the scalar loss directly; `Trainer.fit(... loss_fn=None ...)` consumes it.

Outputs:
  * checkpoint.pt
  * samples.png  (16-image sample grid after training)

Hydra entry point — knobs:

    python train.py                                # defaults from week10/ddpm.yaml
    python train.py quick=true                     # CI smoke (1 epoch, 256 imgs)
    python train.py trainer.max_epochs=20 diffusion.T=500

Requires `pip install -e '.[dl,diffusion,ops]'`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

HERE = Path(__file__).resolve().parent
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "src" / "mlcourse" / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="week10/ddpm")
def main(cfg: DictConfig) -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed — skipping DDPM training demo.")
        return 0

    from ddpm import DiffusionSchedule, SmallUNet, ddpm_loss, ddpm_sample
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import FashionMNIST

    from mlcourse.trainer import Trainer, TrainerConfig
    from mlcourse.utils import detect_device

    device = cfg.trainer.device if cfg.trainer.device != "auto" else detect_device()
    print(f"device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    data_root = cfg.data.root if Path(cfg.data.root).is_absolute() else str(HERE / cfg.data.root)
    train_ds = FashionMNIST(data_root, train=True, download=True, transform=transform)
    epochs = cfg.trainer.max_epochs
    if cfg.quick:
        train_ds = torch.utils.data.Subset(train_ds, range(cfg.data.quick_subset_size))
        epochs = 1
    loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)

    if cfg.diffusion.schedule != "linear":
        raise ValueError(f"only 'linear' schedule supported, got {cfg.diffusion.schedule!r}")
    schedule = DiffusionSchedule.linear(cfg.diffusion.T)
    unet = SmallUNet(in_ch=1, base=64)

    class DDPMLossModule(torch.nn.Module):
        """Adapter: `forward(images, labels) -> scalar loss`, consumed by Trainer."""

        def __init__(self, unet: torch.nn.Module, schedule: DiffusionSchedule) -> None:
            super().__init__()
            self.unet = unet
            self.schedule = schedule

        def forward(self, images: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
            return ddpm_loss(self.unet, images, self.schedule)

    model = DDPMLossModule(unet, schedule)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=cfg.trainer.lr)
    trainer = Trainer(
        TrainerConfig(
            max_epochs=epochs,
            lr=cfg.trainer.lr,
            device=device,
            seed=cfg.trainer.seed,
            grad_clip_norm=cfg.trainer.grad_clip_norm,
        )
    )
    trainer.fit(model, loader, loss_fn=None, optimizer=optimizer)
    for epoch, tl in enumerate(trainer.history["train_loss"], start=1):
        print(f"  epoch {epoch}: train loss = {tl:.4f}")

    out = cfg.out if Path(cfg.out).is_absolute() else str(HERE / cfg.out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.unet.state_dict(), "T": cfg.diffusion.T, "schedule": "linear"}, out
    )
    print("saved:", out)

    samples = ddpm_sample(model.unet.to(device), (16, 1, 28, 28), schedule, device=device, seed=0)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = samples.clamp(-1, 1).cpu().numpy()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(grid[i, 0] * 0.5 + 0.5, cmap="gray")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(HERE / "samples.png", dpi=120)
        plt.close(fig)
        print("wrote:", HERE / "samples.png")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
