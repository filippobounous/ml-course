"""Train the micrograd-style MLP on two-moons using our scalar autograd.

Outputs in this directory:
  * training_curve.png
  * decision_boundary.png
  * report.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlcourse.autograd import MLP, Adam, Value

HERE = Path(__file__).resolve().parent


def two_moons(n: int = 200, noise: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, np.pi, n1)
    theta2 = rng.uniform(0, np.pi, n2)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    X2 = np.column_stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)])
    X = np.vstack([X1, X2]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([np.zeros(n1, dtype=np.float64), np.ones(n2, dtype=np.float64)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _bce_loss(logit: Value, target: float) -> Value:
    # Stable BCE: max(x, 0) - x*y + log(1 + exp(-|x|)).
    # Compute via our sigmoid primitive for clarity.
    p = logit.sigmoid()
    eps = 1e-7
    if target > 0.5:
        return -(p + eps).log()
    return -((1 - p) + eps).log()


def train(X, y, *, epochs: int = 200, lr: float = 5e-2, seed: int = 0):
    mlp = MLP(n_in=2, n_outs=[8, 8, 1], hidden_activation="tanh", seed=seed)
    opt = Adam(mlp.parameters(), lr=lr)
    losses: list[float] = []
    accs: list[float] = []
    for _ in range(epochs):
        # Single-example SGD is fastest for scalar micrograd on ~200 points.
        indices = np.random.default_rng(seed).permutation(len(X))
        total_loss = 0.0
        correct = 0
        for i in indices:
            logit = mlp([float(X[i, 0]), float(X[i, 1])])
            assert isinstance(logit, Value)
            loss = _bce_loss(logit, float(y[i]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.data)
            pred = 1.0 if logit.data > 0.0 else 0.0
            correct += int(pred == y[i])
        losses.append(total_loss / len(X))
        accs.append(correct / len(X))
        # Decrease seed usage overhead; indices is identical each epoch but we
        # reshuffle with a fresh RNG at the top of the loop for a real SGD.
        seed += 1
    return mlp, losses, accs


def _plot_training(losses, accs, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses)
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="BCE")
    axes[1].plot(accs)
    axes[1].set(title="Training accuracy", xlabel="epoch", ylabel="acc", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_boundary(mlp, X, y, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    grid_res = 80
    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    Z = np.zeros((grid_res, grid_res))
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            out_val = mlp([float(xv), float(yv)])
            Z[j, i] = float(out_val.data)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xs, ys, Z, levels=50, cmap="RdBu_r", alpha=0.7)
    ax.contour(xs, ys, Z, levels=[0.0], colors="k", linewidths=1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="white", edgecolors="k", s=20)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="black", edgecolors="k", s=20)
    ax.set(title="Decision boundary", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    X, y = two_moons(n=200, noise=0.18, seed=0)
    mlp, losses, accs = train(X, y, epochs=80, lr=3e-2, seed=0)

    _plot_training(losses, accs, HERE / "training_curve.png")
    _plot_boundary(mlp, X, y, HERE / "decision_boundary.png")

    lines = [
        "# Week 5 — micrograd MLP on two-moons",
        "",
        f"- final train loss: `{losses[-1]:.4f}`",
        f"- final train accuracy: `{accs[-1]:.4f}`",
        f"- parameters: `{len(mlp.parameters())}`",
        "",
        "See `training_curve.png` and `decision_boundary.png`.",
    ]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("final loss:", losses[-1])
    print("final acc:", accs[-1])
    print("wrote:", HERE / "training_curve.png")
    print("wrote:", HERE / "decision_boundary.png")
    print("wrote:", HERE / "report.md")


if __name__ == "__main__":
    main()
