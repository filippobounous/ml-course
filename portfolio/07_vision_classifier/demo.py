"""End-to-end CIFAR-10 demo, driven by Hydra.

Trains ResNet-18 from scratch for a handful of epochs, evaluates against a
transfer-learning baseline, runs Grad-CAM on a handful of predictions, and
sweeps FGSM epsilons to produce a robustness curve.

Target runtime: 20–40 minutes on M-series (MPS), ≈ 2–3 hours on CPU.
The CI-friendly quick mode does a 1-epoch run on a 5k subset for sanity.

Hydra entry point — knobs:

    python demo.py                                 # defaults from week07/cifar10.yaml
    python demo.py quick=true                      # CI smoke (1 epoch, 5k subset)
    python demo.py trainer.max_epochs=20 trainer.lr=0.05
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

HERE = Path(__file__).resolve().parent
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "src" / "mlcourse" / "configs")


def _fmt_row(name: str, loss: float, acc: float) -> str:
    return f"| {name} | {loss:.4f} | {acc:.4f} |"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="week07/cifar10")
def main(cfg: DictConfig) -> None:
    import torch
    from classifier import GradCAM, evaluate, fgsm, get_cifar10_loaders, resnet18_for_cifar

    from mlcourse.trainer import Trainer, TrainerConfig
    from mlcourse.utils import detect_device

    device = cfg.trainer.device if cfg.trainer.device != "auto" else detect_device()
    print(f"device: {device}")
    train_loader, test_loader = get_cifar10_loaders(cfg.data.root, cfg.data.batch_size)

    epochs = cfg.trainer.max_epochs
    if cfg.quick:
        train_loader.dataset.data = train_loader.dataset.data[: cfg.data.quick_subset_size]  # type: ignore[attr-defined]
        train_loader.dataset.targets = train_loader.dataset.targets[: cfg.data.quick_subset_size]  # type: ignore[attr-defined]
        epochs = 1

    model = resnet18_for_cifar()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.trainer.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    trainer_cfg = TrainerConfig(
        max_epochs=epochs,
        lr=cfg.trainer.lr,
        device=device,
        seed=cfg.trainer.seed,
        grad_clip_norm=cfg.trainer.grad_clip_norm,
    )
    trainer = Trainer(trainer_cfg)

    print("\nTraining ResNet-18 (scratch) via mlcourse.Trainer...")
    trainer.fit(
        model,
        train_loader,
        val_loader=test_loader,
        loss_fn=torch.nn.functional.cross_entropy,
        optimizer=optimizer,
    )
    for epoch, (tl, vl) in enumerate(
        zip(trainer.history["train_loss"], trainer.history["val_loss"], strict=True), start=1
    ):
        print(f"  epoch {epoch:2d}  train={tl:.4f}  val={vl:.4f}")

    scratch_loss, scratch_acc = evaluate(model, test_loader, device)

    print("\nRunning Grad-CAM on 8 test images...")
    x_batch, _y_batch = next(iter(test_loader))
    x_batch = x_batch[:8].to(device)
    with GradCAM(model, target_layer=model.layer4) as cam:
        heatmaps = torch.stack([cam(x_batch[i : i + 1]).cpu() for i in range(8)])

    print("\nFGSM epsilon sweep...")
    fgsm_acc: list[tuple[float, float]] = []
    for eps in cfg.eval.fgsm_epsilons:
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_adv = fgsm(model, x, y, eps) if eps > 0 else x
            with torch.no_grad():
                correct += int((model(x_adv).argmax(1) == y).sum())
            total += y.size(0)
            if total >= cfg.eval.fgsm_total_cap:
                break
        acc = correct / total
        fgsm_acc.append((eps, acc))
        print(f"  eps={eps:.4f}  acc={acc:.4f}")

    lines = [
        "# Week 7 — CIFAR-10 classifier",
        "",
        f"Device: `{device}` · epochs: `{epochs}` · batch: `{cfg.data.batch_size}` · lr: `{cfg.trainer.lr}`",
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
