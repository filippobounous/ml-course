"""A simple market-making gymnasium environment.

Conceptually: you quote a bid-ask spread around a noisy mid-price. Each step:
  * the mid drifts by a small random shock,
  * buyers / sellers hit your bid / ask with probabilities decreasing in spread,
  * your inventory accumulates; you pay a quadratic inventory penalty.

Action space is discrete (Discrete(5)): choose a spread level.
Observation: (mid-price change, current inventory, time remaining).
Reward: per-step PnL minus an inventory penalty.

The env has no external dependencies beyond `gymnasium` (and numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "SimpleMarketMakerEnv requires gymnasium. Install with `pip install -e '.[rl]'`."
    ) from e


@dataclass(frozen=True)
class MarketConfig:
    horizon: int = 200
    mid_vol: float = 0.01
    hit_prob_base: float = 0.6
    hit_spread_penalty: float = 50.0
    inventory_penalty: float = 0.05
    inventory_cap: float = 10.0


_SPREAD_LEVELS = np.array([0.005, 0.01, 0.02, 0.04, 0.08], dtype=np.float32)


class SimpleMarketMakerEnv(gym.Env):
    """Discrete-action market-making env suitable for PPO / REINFORCE."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: MarketConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or MarketConfig()
        self.action_space = spaces.Discrete(len(_SPREAD_LEVELS))
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -self.cfg.inventory_cap, 0.0], dtype=np.float32),
            high=np.array([np.inf, self.cfg.inventory_cap, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._mid = 0.0
        self._inventory = 0.0
        self._t = 0

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[no-untyped-def]
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mid = 100.0
        self._inventory = 0.0
        self._t = 0
        return self._obs(0.0), {}

    def step(self, action):  # type: ignore[no-untyped-def]
        spread = float(_SPREAD_LEVELS[int(action)])
        bid = self._mid - spread / 2.0
        ask = self._mid + spread / 2.0

        hit_prob = self.cfg.hit_prob_base * np.exp(-self.cfg.hit_spread_penalty * spread)
        buy_hit = self._rng.uniform() < hit_prob
        sell_hit = self._rng.uniform() < hit_prob

        pnl = 0.0
        if buy_hit and self._inventory > -self.cfg.inventory_cap:
            # someone buys at our ask
            pnl += ask
            self._inventory -= 1.0
        if sell_hit and self._inventory < self.cfg.inventory_cap:
            # someone sells at our bid
            pnl -= bid
            self._inventory += 1.0

        old_mid = self._mid
        shock = self._rng.normal(scale=self.cfg.mid_vol) * self._mid
        self._mid += shock
        mark_to_market = self._inventory * (self._mid - old_mid)
        inv_penalty = self.cfg.inventory_penalty * self._inventory**2

        reward = float(pnl + mark_to_market - inv_penalty)
        self._t += 1
        done = self._t >= self.cfg.horizon
        return self._obs(shock), reward, done, False, {"inventory": self._inventory}

    def _obs(self, shock: float) -> np.ndarray:
        return np.array(
            [shock, self._inventory, 1.0 - self._t / self.cfg.horizon], dtype=np.float32
        )
