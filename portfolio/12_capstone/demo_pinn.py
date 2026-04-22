"""Train the PINN on Burgers' and compare to the analytical Cole-Hopf solution.

Outputs:
  * checkpoint.pt
  * pinn_vs_exact.png — side-by-side u_θ vs u_exact heatmaps with error panel.
  * results.md — scalar L2 and sup-norm error.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOLUTIONS_PATH = (
    HERE.parent.parent / "modules" / "12_applied_capstone" / "problems" / "solutions.py"
)


def _load_solutions():
    spec = importlib.util.spec_from_file_location("w12_solutions", SOLUTIONS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=4000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.max_iters = 200

    try:
        import torch
    except ImportError:
        print("torch not installed — skipping PINN demo.")
        return 0

    import numpy as np
    from pinn_burgers import PINN, PINNConfig, train

    from mlcourse.utils import detect_device

    device = detect_device()
    print(f"device: {device}")

    cfg = PINNConfig(max_iters=args.max_iters)
    result = train(cfg, device=device)
    model = PINN(cfg.hidden, cfg.depth).to(device)
    model.load_state_dict(result["model_state"])
    model.eval()

    # Dense evaluation grid.
    sols = _load_solutions()
    x = np.linspace(-1.0, 1.0, 101)
    t = np.linspace(0.01, 1.0, 50)
    u_exact = sols.burgers_cole_hopf(x, t, nu=cfg.nu)

    X, T = np.meshgrid(x, t)
    with torch.no_grad():
        xt = torch.tensor(
            np.column_stack([X.ravel(), T.ravel()]), dtype=torch.float32, device=device
        )
        u_pred = model(xt[:, :1], xt[:, 1:]).cpu().numpy().reshape(t.size, x.size)

    l2 = float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))
    sup = float(np.abs(u_pred - u_exact).max())

    report = [
        "# Week 12 — PINN for Burgers' (reproduction of Raissi 2019 Fig. 2)",
        "",
        f"Device: `{device}` · iters: `{cfg.max_iters}` · ν: `{cfg.nu:.6f}`",
        "",
        "## Errors vs Cole-Hopf analytical solution",
        "",
        f"- Relative L2 error:   `{l2:.4e}`",
        f"- Sup-norm error:      `{sup:.4e}`",
        "",
        "See `pinn_vs_exact.png` for side-by-side heatmaps.",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")
    print(f"L2 error: {l2:.4e}")
    print(f"sup error: {sup:.4e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, data, title in zip(
            axes,
            (u_exact, u_pred, u_pred - u_exact),
            ("exact", "PINN", "PINN − exact"),
            strict=True,
        ):
            im = ax.imshow(data, extent=(x[0], x[-1], t[-1], t[0]), aspect="auto", cmap="RdBu_r")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(HERE / "pinn_vs_exact.png", dpi=120)
        plt.close(fig)
        print("wrote:", HERE / "pinn_vs_exact.png")
    except ImportError:
        pass

    torch.save(result, HERE / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
