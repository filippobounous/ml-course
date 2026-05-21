"""Reproduce Schulman 2017 Figure 6 at tiny scale: PPO performance vs $\\varepsilon$.

Sweeps clip $\\varepsilon \\in \\{0.1, 0.2, 0.3, \\infty\\}$ on CartPole-v1 with
4 seeds per config. Plots learning curves and writes a `findings.md` with
the ablation table.

Uses the `mlcourse` PPO from `portfolio/11_rl_agent/ppo.py` — same
implementation the learner builds in W11.

Usage:
    python portfolio/12_capstone/paper_reproduction/ppo_clip_ablation.py
    python portfolio/12_capstone/paper_reproduction/ppo_clip_ablation.py --quick

`--quick` reduces steps + seeds to ≤ 30 s total for the CI smoke path.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W11_PPO = Path(__file__).resolve().parents[2] / "11_rl_agent" / "ppo.py"

# No-clip sentinel — picks a clip ε so large the min(rA, clip(r) A) is always rA.
NO_CLIP_EPSILON = 1e6


def _load_ppo():
    """Side-load the W11 PPO module without requiring a package install."""
    spec = importlib.util.spec_from_file_location("ppo_w11", W11_PPO)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ppo_w11"] = module
    spec.loader.exec_module(module)
    return module


def _make_cartpole():
    """Build a fresh `CartPole-v1` env. Imported lazily so the module loads
    cleanly even without gymnasium installed."""
    import gymnasium as gym

    return gym.make("CartPole-v1")


def run_one(eps: float, seed: int, total_steps: int) -> dict:
    """Train PPO with clip ε at the given seed; return history + final-return."""
    ppo = _load_ppo()
    cfg = ppo.PPOConfig(
        total_steps=total_steps,
        steps_per_rollout=1024,
        clip_eps=eps,
        seed=seed,
        # Huang 2022 "37 details" subset already lives in PPOConfig defaults.
    )
    out = ppo.train(_make_cartpole, cfg=cfg, device="cpu")
    history = out["history"]
    # Mean of the last 5 logged points = final-100k-steps average for tiny runs.
    finals = [h["mean_return"] for h in history[-5:]]
    return {
        "eps": eps,
        "seed": seed,
        "history": history,
        "final_mean": statistics.fmean(finals) if finals else 0.0,
        "final_std": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce steps + seeds for a CI smoke run (~30 s total).",
    )
    parser.add_argument("--seeds", type=int, default=4, help="seeds per config")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=50_000,
        help="PPO total environment steps per run",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.total_steps = 4_096  # ~4 rollouts -> ~10s on CPU

    eps_configs: list[tuple[str, float]] = [
        ("eps=0.1", 0.1),
        ("eps=0.2", 0.2),
        ("eps=0.3", 0.3),
        ("no-clip", NO_CLIP_EPSILON),
    ]

    # Try the actual training run. If gymnasium / torch unavailable, fall
    # back to a stub message so the smoke path doesn't fail in CI envs
    # without the rl extra installed.
    try:
        import gymnasium  # noqa: F401  - presence check
        import torch  # noqa: F401  - presence check
    except ImportError:
        print("gymnasium + torch required; install with `pip install -e '.[dl,rl]'`")
        print("skipping reproduction.")
        return 0

    table: list[dict] = []
    print(f"\nRunning {len(eps_configs)} configs × {args.seeds} seeds")
    print(f"= {len(eps_configs) * args.seeds} PPO runs at {args.total_steps:,} steps each\n")
    for label, eps in eps_configs:
        seed_finals: list[float] = []
        all_histories: list[list[dict]] = []
        for seed in range(args.seeds):
            result = run_one(eps, seed, args.total_steps)
            seed_finals.append(result["final_mean"])
            all_histories.append(result["history"])
            print(f"  {label}, seed {seed}: final = {result['final_mean']:6.1f}")
        mean = statistics.fmean(seed_finals)
        std = statistics.pstdev(seed_finals) if len(seed_finals) > 1 else 0.0
        table.append(
            {
                "label": label,
                "eps": eps,
                "mean": mean,
                "std": std,
                "seeds": args.seeds,
                "histories": all_histories,
            }
        )

    _write_findings(table, args)
    _maybe_plot(table)
    return 0


def _write_findings(table: list[dict], args) -> None:
    """Write `findings.md` with the ablation table + a one-paragraph note."""
    lines = [
        "# PPO clip-ε ablation — findings",
        "",
        "Reproducing Schulman 2017 Figure 6 at tiny scale on `CartPole-v1` using",
        f"the W11 `mlcourse` PPO implementation ({args.total_steps:,} steps × ",
        f"{args.seeds} seeds per config).",
        "",
        "## Ablation table",
        "",
        "| Config | ε | Final mean reward (over last 5 rollouts) | Std (across seeds) |",
        "|---|---|---|---|",
    ]
    for row in table:
        eps_str = "10⁶ (no-clip)" if row["eps"] >= NO_CLIP_EPSILON else f"{row['eps']:.1f}"
        lines.append(f"| {row['label']} | {eps_str} | {row['mean']:.1f} | {row['std']:.1f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- CartPole-v1 max return = 500. Reaching ~400+ is the convergence threshold.",
        "- 4 seeds is not enough to separate $\\varepsilon = 0.1$ from $\\varepsilon = 0.2$ on this",
        "  toy env; the paper's > 8-seed runs on MuJoCo show a clearer ranking.",
        "- No-clip ($\\varepsilon = 10^6$) is the failure case the paper studies: even on",
        "  the forgiving CartPole, it should be noisier and sometimes diverge.",
        "",
        "## What I saw that surprised me",
        "",
        "*Fill in after running.*",
    ]
    (HERE / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote: {HERE / 'findings.md'}")


def _maybe_plot(table: list[dict]) -> None:
    """Plot learning curves (one per config, all seeds overlaid)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["C0", "C1", "C2", "C3"]
    for color, row in zip(colors, table, strict=True):
        for hist in row["histories"]:
            steps = [h["steps"] for h in hist]
            returns = [h["mean_return"] for h in hist]
            ax.plot(steps, returns, color=color, alpha=0.35)
        # Bold line: mean across seeds (assume aligned step grids).
        if row["histories"]:
            n = min(len(h) for h in row["histories"])
            mean_curve = [
                statistics.fmean(h[i]["mean_return"] for h in row["histories"]) for i in range(n)
            ]
            steps = [row["histories"][0][i]["steps"] for i in range(n)]
            ax.plot(steps, mean_curve, color=color, label=row["label"], linewidth=2)

    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean episode return (last 10 eps)")
    ax.set_title("PPO clip-ε ablation on CartPole-v1 (reproducing Schulman 2017 Fig 6)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE / "figure_clip_ablation.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote: {out}")


if __name__ == "__main__":
    sys.exit(main())
