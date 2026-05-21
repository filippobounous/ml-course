"""DDPM vs DDIM step-count ablation on a trained checkpoint.

For each sampler and step count:
  * draw N samples,
  * compute the canonical **Fréchet Inception Distance** (via `fid.py`),
  * also compute the original pixel-statistics proxy for cross-reference.

Writes `ablation.md` and `ablation_samples.png`.

Note: FID download (~100 MB InceptionV3 weights) happens on first use.
Pass `--no-fid` to skip and fall back to the pixel-stat proxy only —
useful for the CI smoke path.
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
    parser.add_argument(
        "--no-fid",
        action="store_true",
        help="Skip FID (no InceptionV3 download); use pixel-stat proxy only.",
    )
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

    # Real-data tensor for metric comparison.
    test = FashionMNIST(
        args.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    )
    real_tensor = torch.stack([test[i][0] for i in range(args.n_samples)])
    real = real_tensor.numpy()

    # FID setup — load InceptionV3 once, extract real-data features once.
    fid_model = None
    real_feats = None
    if not args.no_fid:
        from fid import extract_features, load_inception

        print("loading InceptionV3 (first run downloads ~100 MB)...")
        fid_model = load_inception(device=device)
        print("extracting real-data features...")
        real_feats = extract_features(real_tensor, fid_model, device=device)

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
        clamped = samples.clamp(-1, 1)
        arr = clamped.cpu().numpy()
        pixel_d = _stat_distance(arr, real)
        fid = None
        if fid_model is not None and real_feats is not None:
            from fid import extract_features, frechet_distance, statistics

            fake_feats = extract_features(clamped, fid_model, device=device)
            mu_r, sigma_r = statistics(real_feats)
            mu_f, sigma_f = statistics(fake_feats)
            fid = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
        results.append((name, n_steps, pixel_d, fid))
        fid_str = f"FID = {fid:.3f}" if fid is not None else "FID = (skipped)"
        print(f"  {name:>20s}: pixel-stat = {pixel_d:.4f}, {fid_str}")

    metric_header = ["Pixel-stat distance"] + (["FID"] if not args.no_fid else [])
    lines = [
        "# Week 10 — DDPM vs DDIM ablation",
        "",
        "Canonical **FID** (InceptionV3 features → Fréchet distance between fitted",
        "Gaussians; Heusel et al. 2017) plus the pixel-statistics proxy for",
        "cross-reference. Lower is better for both. FashionMNIST is greyscale-",
        "repeated to RGB and bilinear-upsampled to 299×299 before InceptionV3.",
        "",
        "| Sampler | Steps | " + " | ".join(metric_header) + " |",
        "|---|---|" + "|".join("---" for _ in metric_header) + "|",
    ]
    for name, n_steps, pixel_d, fid in results:
        cells = [f"{pixel_d:.4f}"]
        if not args.no_fid:
            cells.append(f"{fid:.3f}" if fid is not None else "n/a")
        lines.append(f"| {name} | {n_steps} | " + " | ".join(cells) + " |")
    (HERE / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote:", HERE / "ablation.md")

    # Sample grid comparison.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(step_counts), 8, figsize=(12, 1.5 * len(step_counts)))
        for i, (name, steps, _pixel, _fid) in enumerate(results):
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
