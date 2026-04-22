"""End-to-end CIFAR-10 demo.

Trains ResNet-18 from scratch for a handful of epochs, evaluates against a
transfer-learning baseline, runs Grad-CAM on a handful of predictions, and
sweeps FGSM epsilons to produce a robustness curve.

Target runtime: 20–40 minutes on M-series (MPS), ≈ 2–3 hours on CPU.
The CI-friendly `quick` mode does a 1-epoch run on a 5k subset for sanity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fmt_row(name: str, loss: float, acc: float) -> str:
    return f"| {name} | {loss:.4f} | {acc:.4f} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--quick", action="store_true", help="1 epoch, subset data")
    parser.add_argument("--data-root", type=str, default=str(HERE / "data"))
    args = parser.parse_args()

    import torch
    from classifier import (
        GradCAM,
        evaluate,
        fgsm,
        get_cifar10_loaders,
        resnet18_for_cifar,
        train_one_epoch,
    )

    from mlcourse.utils import detect_device

    device = detect_device()
    print(f"device: {device}")
    train_loader, test_loader = get_cifar10_loaders(args.data_root, args.batch_size)

    if args.quick:
        # Subsample for a CI smoke check.
        train_loader.dataset.data = train_loader.dataset.data[:5000]  # type: ignore[attr-defined]
        train_loader.dataset.targets = train_loader.dataset.targets[:5000]  # type: ignore[attr-defined]
        args.epochs = 1

    # -- scratch ResNet-18 ----------------------------------------------------
    model = resnet18_for_cifar().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\nTraining ResNet-18 (scratch)...")
    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, device)
        vl, va = evaluate(model, test_loader, device)
        scheduler.step()
        print(f"  epoch {epoch:2d}  train={tl:.4f}  test={vl:.4f}  acc={va:.4f}")

    scratch_loss, scratch_acc = evaluate(model, test_loader, device)

    # -- Grad-CAM -------------------------------------------------------------
    print("\nRunning Grad-CAM on 8 test images...")
    x_batch, _y_batch = next(iter(test_loader))
    x_batch = x_batch[:8].to(device)
    with GradCAM(model, target_layer=model.layer4) as cam:
        heatmaps = torch.stack([cam(x_batch[i : i + 1]).cpu() for i in range(8)])

    # -- FGSM sweep -----------------------------------------------------------
    print("\nFGSM epsilon sweep...")
    epsilons = [0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255]
    fgsm_acc: list[tuple[float, float]] = []
    for eps in epsilons:
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_adv = fgsm(model, x, y, eps) if eps > 0 else x
            with torch.no_grad():
                correct += int((model(x_adv).argmax(1) == y).sum())
            total += y.size(0)
            if total >= 1000:  # cap the sweep for speed
                break
        acc = correct / total
        fgsm_acc.append((eps, acc))
        print(f"  eps={eps:.4f}  acc={acc:.4f}")

    # -- Report ---------------------------------------------------------------
    lines = [
        "# Week 7 — CIFAR-10 classifier",
        "",
        f"Device: `{device}` · epochs: `{args.epochs}` · batch: `{args.batch_size}` · lr: `{args.lr}`",
        "",
        "## Final accuracy",
        "",
        "| Model | Test loss | Test acc |",
        "|---|---|---|",
        _fmt_row("ResNet-18 (scratch)", scratch_loss, scratch_acc),
        "",
        "## FGSM robustness",
        "",
        "| ε | accuracy |",
        "|---|---|",
        *[f"| {eps:.4f} | {acc:.4f} |" for eps, acc in fgsm_acc],
        "",
        "Grad-CAM panels: `gradcam.png` (not regenerated if matplotlib absent).",
    ]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote:", HERE / "report.md")

    # Optional heat-map figure.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            img = x_batch[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            axes[0, i].imshow(img)
            axes[0, i].axis("off")
            axes[1, i].imshow(img)
            heatmap = heatmaps[i].numpy()
            axes[1, i].imshow(heatmap, cmap="jet", alpha=0.5, extent=(0, 32, 32, 0))
            axes[1, i].axis("off")
        fig.tight_layout()
        fig.savefig(HERE / "gradcam.png", dpi=120)
        plt.close(fig)
        print("wrote:", HERE / "gradcam.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
