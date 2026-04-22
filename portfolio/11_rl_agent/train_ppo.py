"""Train PPO on the SimpleMarketMakerEnv and save a learning curve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default=str(HERE / "ppo_checkpoint.pt"))
    args = parser.parse_args()

    if args.quick:
        args.total_steps = 4096

    try:
        import torch
    except ImportError:
        print("torch not installed — skipping PPO training demo.")
        return 0

    from market_env import MarketConfig, SimpleMarketMakerEnv
    from ppo import PPOConfig, train

    from mlcourse.utils import detect_device

    device = detect_device()
    print(f"device: {device}")
    cfg = PPOConfig(total_steps=args.total_steps, seed=args.seed)

    def env_fn():
        return SimpleMarketMakerEnv(MarketConfig())

    out = train(env_fn, cfg, device=device)
    import torch

    torch.save(out, args.out)
    print("saved:", args.out)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        steps = [h["steps"] for h in out["history"]]
        returns = [h["mean_return"] for h in out["history"]]
        ax.plot(steps, returns)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("mean episode return (last 10)")
        ax.set_title("PPO on SimpleMarketMakerEnv")
        fig.tight_layout()
        fig.savefig(HERE / "ppo_curve.png", dpi=120)
        plt.close(fig)
        print("wrote:", HERE / "ppo_curve.png")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
