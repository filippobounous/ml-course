"""Single-file PPO, CleanRL-style.

Actor-critic over a 3-layer MLP, GAE advantages, clipped surrogate. Trained
here on the `SimpleMarketMakerEnv` but works on any gymnasium discrete-action
env with a flat observation space.

Torch is required (`pip install -e '.[dl,rl]'`).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as e:  # pragma: no cover
    raise ImportError("PPO requires PyTorch. Install with `pip install -e '.[dl,rl]'`.") from e


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    steps_per_rollout: int = 2048
    mini_batch: int = 64
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 0
    # --- Huang et al. 2022 "37 details" subset --------------------------------
    # Normalise observations with a Welford running mean/std.
    normalize_obs: bool = True
    # Clip the value-function loss symmetrically around the old value (CleanRL).
    vf_clip_eps: float | None = 0.2
    # Linearly anneal the learning rate from `lr` → 0 over `total_steps`.
    anneal_lr: bool = True
    # Anneal the clip range alongside the learning rate (optional — default off).
    anneal_clip: bool = False


class RunningNormalizer:
    """Welford-style running mean / variance estimator for observation vectors.

    Numerically stable in an online streaming setting; standard in PPO
    reference implementations (CleanRL, stable-baselines3).
    """

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4) -> None:
        self.count = eps
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.M2 = torch.zeros(shape, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        # x: (batch, *shape). Accumulate mean + M2 in fp32 on CPU for determinism.
        x_cpu = x.detach().to("cpu", dtype=torch.float32)
        batch_mean = x_cpu.mean(dim=0)
        batch_var = x_cpu.var(dim=0, unbiased=False)
        batch_count = x_cpu.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        self.M2 = (
            self.M2
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / tot_count
        )
        self.mean = new_mean
        self.count = tot_count

    @property
    def variance(self) -> torch.Tensor:
        return self.M2 / max(self.count, 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.variance + 1e-8).to(x.device)
        return (x - self.mean.to(x.device)) / std


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def compute_gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    last_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device)
    next_value = last_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        running = delta + gamma * lam * mask * running
        adv[t] = running
        next_value = values[t]
    returns = adv + values
    return adv, returns


def train(env_fn, cfg: PPOConfig | None = None, device: str = "cpu") -> dict:
    """Train PPO end-to-end. `env_fn()` returns a fresh gymnasium env.

    Implements the Huang-2022 "37 details" subset listed on `PPOConfig`:
    observation normalisation, advantage normalisation (already present),
    linear LR annealing, and value-loss clipping.
    """
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    obs_norm = RunningNormalizer(shape=(obs_dim,)) if cfg.normalize_obs else None

    obs_np, _ = env.reset(seed=cfg.seed)
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    history: list[dict[str, float]] = []
    total_steps_done = 0
    episode_return = 0.0
    episode_returns: list[float] = []

    while total_steps_done < cfg.total_steps:
        # -- Rollout ---------------------------------------------------------
        obs_buf = torch.zeros((cfg.steps_per_rollout, obs_dim), device=device)
        act_buf = torch.zeros(cfg.steps_per_rollout, dtype=torch.long, device=device)
        logp_buf = torch.zeros(cfg.steps_per_rollout, device=device)
        rew_buf = torch.zeros(cfg.steps_per_rollout, device=device)
        val_buf = torch.zeros(cfg.steps_per_rollout, device=device)
        done_buf = torch.zeros(cfg.steps_per_rollout, device=device)

        raw_obs_buf = torch.zeros((cfg.steps_per_rollout, obs_dim), device=device)
        for step in range(cfg.steps_per_rollout):
            # Update the normalizer online with the raw obs, then normalise.
            if obs_norm is not None:
                obs_norm.update(obs.unsqueeze(0))
                obs_input = obs_norm.normalize(obs)
            else:
                obs_input = obs
            with torch.no_grad():
                logits, value = model(obs_input)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs_np, reward, terminated, truncated, _info = env.step(int(action))
            done = bool(terminated or truncated)
            raw_obs_buf[step] = obs
            obs_buf[step] = obs_input  # store the *normalised* observation for training
            act_buf[step] = action
            logp_buf[step] = log_prob
            rew_buf[step] = reward
            val_buf[step] = value
            done_buf[step] = float(done)
            episode_return += float(reward)

            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                next_obs_np, _ = env.reset()
            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            total_steps_done += 1

        # Last value for GAE bootstrap (also normalised).
        with torch.no_grad():
            last_input = obs_norm.normalize(obs) if obs_norm is not None else obs
            _, last_value = model(last_input)
        adv, returns = compute_gae_torch(
            rew_buf, val_buf, done_buf, gamma=cfg.gamma, lam=cfg.gae_lambda, last_value=last_value
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Linear LR anneal → 0 across total_steps.
        frac = max(1.0 - total_steps_done / cfg.total_steps, 0.0)
        if cfg.anneal_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac
        clip_eps = cfg.clip_eps * (frac if cfg.anneal_clip else 1.0)

        # -- Update ----------------------------------------------------------
        idx = torch.arange(cfg.steps_per_rollout, device=device)
        for _ in range(cfg.update_epochs):
            idx = idx[torch.randperm(cfg.steps_per_rollout, device=device)]
            for start in range(0, cfg.steps_per_rollout, cfg.mini_batch):
                batch = idx[start : start + cfg.mini_batch]
                logits, values = model(obs_buf[batch])
                dist = Categorical(logits=logits)
                new_log_prob = dist.log_prob(act_buf[batch])
                ratio = torch.exp(new_log_prob - logp_buf[batch])
                unclipped = ratio * adv[batch]
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                # Value-function clipping (Huang 2022 §5): min of two losses,
                # one on the new value, one on the old value clipped by ε.
                if cfg.vf_clip_eps is not None:
                    v_old = val_buf[batch]
                    v_clipped = v_old + torch.clamp(
                        values - v_old, -cfg.vf_clip_eps, cfg.vf_clip_eps
                    )
                    vf_losses = torch.stack(
                        [(values - returns[batch]) ** 2, (v_clipped - returns[batch]) ** 2]
                    )
                    value_loss = vf_losses.max(dim=0).values.mean()
                else:
                    value_loss = nn.functional.mse_loss(values, returns[batch])
                entropy = dist.entropy().mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        mean_return = sum(episode_returns[-10:]) / max(len(episode_returns[-10:]), 1)
        history.append({"steps": total_steps_done, "mean_return": mean_return})
        if total_steps_done // cfg.steps_per_rollout % 5 == 0:
            print(f"  steps={total_steps_done}  mean_return(last 10 eps)={mean_return:.3f}")

    return {"model_state": model.state_dict(), "history": history, "config": cfg.__dict__}
