"""Train the small UNet-DDPM on FashionMNIST.

Produces a checkpoint and a grid of samples. Companion script `ablate.py`
does the DDPM-vs-DDIM step-count ablation from the same checkpoint.
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

    import torch
    from ddpm import DiffusionSchedule, SmallUNet, ddpm_loss, ddpm_sample
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import FashionMNIST

    from mlcourse.utils import detect_device, seed_everything

    seed_everything(0)
    device = detect_device()
    print(f"device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train = FashionMNIST(args.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    schedule = DiffusionSchedule.linear(args.T)
    model = SmallUNet(in_ch=1, base=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0
        for x, _ in loader:
            x = x.to(device)
            loss = ddpm_loss(model, x, schedule)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss) * x.size(0)
            count += x.size(0)
            if args.quick and count > 256:
                break
        print(f"epoch {epoch}: train loss = {running / max(count, 1):.4f}")

    torch.save({"model_state": model.state_dict(), "T": args.T, "schedule": "linear"}, args.out)
    print("saved:", args.out)

    # Quick sample grid for eyeballing.
    samples = ddpm_sample(model, (16, 1, 28, 28), schedule, device=device, seed=0)
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
