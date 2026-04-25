"""DDPM vs DDIM step-count ablation on a trained checkpoint.

For each sampler and step count:
  * draw 256 samples,
  * compute a simple proxy quality metric (pixel-statistics distance to the
    test set) — a stand-in for FID without the InceptionV3 dependency.

Writes `ablation.md` and `ablation_samples.png`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _stat_distance(samples, real):
    """Distance between pixel-statistics vectors (mean + std per pixel)."""
    import numpy as np

    m1 = samples.mean(axis=0).flatten()
    m2 = real.mean(axis=0).flatten()
    s1 = samples.std(axis=0).flatten()
    s2 = real.std(axis=0).flatten()
    return float(np.linalg.norm(m1 - m2) + np.linalg.norm(s1 - s2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(HERE / "checkpoint.pt"))
    parser.add_argument("--data-root", default=str(HERE / "data"))
    parser.add_argument("--n-samples", type=int, default=256)
    args = parser.parse_args()

    import torch
    from ddpm import DiffusionSchedule, SmallUNet, ddim_sample, ddpm_sample
    from torchvision import transforms
    from torchvision.datasets import FashionMNIST

    from mlcourse.utils import detect_device

    device = detect_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    schedule = DiffusionSchedule.linear(ckpt["T"])
    model = SmallUNet(in_ch=1, base=64).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Real-data stats for the proxy metric.
    test = FashionMNIST(
        args.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    )
    real = torch.stack([test[i][0] for i in range(args.n_samples)]).numpy()

    results = []
    step_counts = [1000, 100, 50, 20, 10]
    for n_steps in step_counts:
        if n_steps == 1000:
            samples = ddpm_sample(
                model, (args.n_samples, 1, 28, 28), schedule, device=device, seed=0
            )
            name = "DDPM (1000)"
        else:
            samples = ddim_sample(
                model,
                (args.n_samples, 1, 28, 28),
                schedule,
                n_steps=n_steps,
                device=device,
                eta=0.0,
                seed=0,
            )
            name = f"DDIM η=0 ({n_steps})"
        arr = samples.clamp(-1, 1).cpu().numpy()
        d = _stat_distance(arr, real)
        results.append((name, n_steps, d))
        print(f"  {name:>20s}: pixel-stat distance = {d:.4f}")

    lines = [
        "# Week 10 — DDPM vs DDIM ablation",
        "",
        "Proxy quality metric: pixel-statistics distance (mean + std per pixel)",
        "between generated samples and FashionMNIST test images. Lower is better.",
        "",
        "| Sampler | Steps | Pixel-stat distance |",
        "|---|---|---|",
        *[f"| {n} | {s} | {d:.4f} |" for n, s, d in results],
    ]
    (HERE / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote:", HERE / "ablation.md")

    # Sample grid comparison.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(step_counts), 8, figsize=(12, 1.5 * len(step_counts)))
        for i, (name, steps, _d) in enumerate(results):
            if steps == 1000:
                samples = ddpm_sample(model, (8, 1, 28, 28), schedule, device=device, seed=1)
            else:
                samples = ddim_sample(
                    model, (8, 1, 28, 28), schedule, n_steps=steps, device=device, seed=1
                )
            grid = samples.clamp(-1, 1).cpu().numpy()
            for j in range(8):
                axes[i, j].imshow(grid[j, 0] * 0.5 + 0.5, cmap="gray")
                axes[i, j].axis("off")
            axes[i, 0].set_ylabel(name, rotation=0, labelpad=40, va="center")
        fig.tight_layout()
        fig.savefig(HERE / "ablation_samples.png", dpi=120)
        plt.close(fig)
        print("wrote:", HERE / "ablation_samples.png")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
