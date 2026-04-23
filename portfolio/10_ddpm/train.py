"""Train the SmallUNet-DDPM on FashionMNIST via `mlcourse.Trainer`.

The DDPM objective doesn't fit the standard `(x, y) → loss` shape: each step
samples a random timestep, adds matching noise, predicts it. We wrap the
`SmallUNet` in a thin `DDPMLossModule` whose `forward(images, labels)` returns
the scalar loss directly; `Trainer.fit(... loss_fn=None ...)` consumes it.

Outputs in this directory:
  * checkpoint.pt
  * samples.png  (16-image sample grid after training)

Requires `pip install -e '.[dl,diffusion,ops]'`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(HERE / "data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--out", default=str(HERE / "checkpoint.pt"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1

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

    device = detect_device()
    print(f"device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_ds = FashionMNIST(args.data_root, train=True, download=True, transform=transform)
    if args.quick:
        # Tiny subset for the CI smoke path.
        train_ds = torch.utils.data.Subset(train_ds, range(256))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    schedule = DiffusionSchedule.linear(args.T)
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
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=args.lr)
    trainer = Trainer(
        TrainerConfig(
            max_epochs=args.epochs,
            lr=args.lr,
            device=device,
            seed=0,
            grad_clip_norm=1.0,
        )
    )
    trainer.fit(model, loader, loss_fn=None, optimizer=optimizer)
    for epoch, tl in enumerate(trainer.history["train_loss"], start=1):
        print(f"  epoch {epoch}: train loss = {tl:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.unet.state_dict(), "T": args.T, "schedule": "linear"}, args.out
    )
    print("saved:", args.out)

    # Eyeball-quality sample grid.
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
