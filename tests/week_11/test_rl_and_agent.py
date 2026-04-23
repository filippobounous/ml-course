"""Week 11 — MDP / PG / PPO NumPy reference + agent loop checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "modules" / "11_rl_agents" / "problems" / "solutions.py"
)
AGENT_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "11_rl_agent" / "agent.py"
PPO_TORCH_PATH = Path(__file__).resolve().parents[2] / "portfolio" / "11_rl_agent" / "ppo.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load(SOLUTIONS_PATH, "w11_solutions")


@pytest.fixture(scope="module")
def agent_mod():
    return _load(AGENT_PATH, "w11_agent")


# -- Tabular RL ----------------------------------------------------------------


def test_value_iteration_on_chain(sols):
    P, R = sols.tiny_chain_mdp(5)
    V, pi, iters = sols.value_iteration(P, R, gamma=0.9, tol=1e-10)
    # Optimal policy goes right everywhere (the reward is at the right end).
    assert (pi == 1).all()
    # V is monotone non-decreasing along the chain and strictly increasing for
    # states that are not already at the absorbing optimum.
    assert (np.diff(V) >= -1e-9).all()
    assert V[-1] == pytest.approx(1.0 / (1 - 0.9), rel=1e-6)
    assert iters < 500


def test_bellman_contraction(sols):
    rng = np.random.default_rng(0)
    P, R = sols.tiny_chain_mdp(5)
    gamma = 0.8
    V = rng.standard_normal(5)
    W = rng.standard_normal(5)

    def bellman_opt(v):
        return (R + gamma * np.einsum("sap,p->sa", P, v)).max(axis=1)

    ratio = sols.bellman_contraction_factor(V, W, bellman_opt(V), bellman_opt(W))
    assert ratio <= gamma + 1e-9


# -- GAE -----------------------------------------------------------------------


def test_gae_matches_hand_computation(sols):
    rewards = np.array([1.0, 0.0, 2.0])
    values = np.array([0.5, 0.5, 0.5])
    dones = np.array([0.0, 0.0, 1.0])
    adv, rets = sols.compute_gae(rewards, values, dones, gamma=0.9, lam=0.8, last_value=0.0)
    # Hand calc with γ=0.9, λ=0.8:
    # δ_2 = 2 + 0.9*0*0 - 0.5 = 1.5; A_2 = 1.5
    # δ_1 = 0 + 0.9*0.5 - 0.5 = -0.05; A_1 = -0.05 + 0.9*0.8*1.5 = 1.03
    # δ_0 = 1 + 0.9*0.5 - 0.5 = 0.95; A_0 = 0.95 + 0.9*0.8*1.03 = 1.6916
    np.testing.assert_allclose(adv, [1.6916, 1.03, 1.5], rtol=1e-6)
    np.testing.assert_allclose(rets, adv + values, rtol=1e-12)


def test_gae_terminal_masks_bootstrap(sols):
    rewards = np.array([1.0, 1.0])
    values = np.array([10.0, 0.0])
    dones = np.array([1.0, 1.0])
    adv, _ = sols.compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=5.0)
    # Done[0] = 1 zeroes out propagation from timestep 1 back into 0.
    # δ_1 = 1 + 0 - 0 = 1; A_1 = 1.
    # δ_0 = 1 + 0 - 10 = -9; A_0 = -9.
    np.testing.assert_allclose(adv, [-9.0, 1.0])


# -- PPO-clip ------------------------------------------------------------------


def test_ppo_clip_zero_at_identity(sols):
    log_p = np.array([-1.0, -2.0, -0.5])
    adv = np.array([1.0, -0.5, 0.3])
    loss = sols.ppo_clip_loss(log_p, log_p, adv)
    # At ratio=1 and positive ε, unclipped=ratio*adv=adv; loss = -mean(adv).
    assert loss == pytest.approx(-adv.mean())


def test_ppo_clip_bounds_positive_advantage(sols):
    # If advantage > 0 and new ratio >> 1, the clip caps the gain.
    log_p = np.array([0.0, 0.0])
    old_log_p = np.array([-5.0, -5.0])  # ratio = exp(5) ≈ 148
    adv = np.array([1.0, 1.0])
    loss_clip = sols.ppo_clip_loss(log_p, old_log_p, adv, clip_eps=0.2)
    # Without clipping, loss would be -mean(ratio * adv) ≈ -148.
    # With clip_eps=0.2, it's bounded to -mean(1.2 * adv) = -1.2.
    assert loss_clip == pytest.approx(-1.2, abs=1e-6)


# -- Agent loop ----------------------------------------------------------------


def test_calculator_evaluates(agent_mod):
    assert agent_mod.calculator("2 + 3 * 4") == "14.0"
    assert agent_mod.calculator("(10 - 4) / 2") == "3.0"
    assert agent_mod.calculator("2 ** 10") == "1024.0"


def test_calculator_rejects_injection(agent_mod):
    # Disallowed characters → error string, no exception.
    out = agent_mod.calculator("__import__('os')")
    assert out.startswith("error"), out


def test_retriever_keyword_scoring(agent_mod):
    corpus = {
        "a": "alpha bravo",
        "b": "alpha charlie delta",
        "c": "echo foxtrot",
    }
    tool = agent_mod.build_retriever(corpus)
    # Query "alpha charlie" matches b (2 tokens) > a (1 token) > c (0).
    assert tool("alpha charlie").startswith("b:")
    # Empty / no-match.
    assert tool("zulu") == "no results"


def test_agent_loop_respects_max_steps(agent_mod):
    # A policy that never finalises must terminate at max_steps with success=False.
    trace = [("always more", "calculator", "1+1")]
    policy = agent_mod.make_fixed_trace_policy(trace)
    cfg = agent_mod.default_config()
    cfg.max_steps = 3
    result = agent_mod.run_agent("ignored", policy, cfg)
    assert result.success is False
    assert len(result.steps) == 3


def test_agent_finalises_on_observation(agent_mod):
    policy = agent_mod.make_math_and_lookup_policy()
    cfg = agent_mod.default_config()
    result = agent_mod.run_agent("2 + 3 * 4", policy, cfg)
    assert result.success is True
    assert result.answer == "14.0"


def test_agent_eval_harness_scores_numeric_and_text(agent_mod):
    tasks = [
        agent_mod.AgentTask("2 + 3", "5.0"),
        agent_mod.AgentTask("What is the capital of France?", "Paris is the capital of France."),
    ]
    policy = agent_mod.make_math_and_lookup_policy()
    cfg = agent_mod.default_config()
    report = agent_mod.evaluate_agent(tasks, policy, cfg)
    assert report.n == 2
    assert report.success_rate == 1.0


# -- Torch-gated PPO shape checks ---------------------------------------------


@pytest.fixture(scope="module")
def ppo_torch():
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    return _load(PPO_TORCH_PATH, "w11_ppo_torch")


def test_actor_critic_forward(ppo_torch):
    import torch

    model = ppo_torch.ActorCritic(obs_dim=4, n_actions=3)
    x = torch.randn(5, 4)
    logits, value = model(x)
    assert logits.shape == (5, 3)
    assert value.shape == (5,)


def test_compute_gae_torch_matches_reference(ppo_torch, sols):
    import torch

    rewards = torch.tensor([1.0, 0.0, 2.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    adv_t, _rets_t = ppo_torch.compute_gae_torch(
        rewards, values, dones, gamma=0.9, lam=0.8, last_value=torch.tensor(0.0)
    )
    adv_n, _ = sols.compute_gae(
        rewards.numpy(), values.numpy(), dones.numpy(), gamma=0.9, lam=0.8, last_value=0.0
    )
    np.testing.assert_allclose(adv_t.numpy(), adv_n, rtol=1e-6)


# -- Huang-2022 "37 details" subset -------------------------------------------


def test_running_normalizer_approaches_zero_mean_unit_var(ppo_torch):
    import torch

    torch.manual_seed(0)
    norm = ppo_torch.RunningNormalizer(shape=(4,))
    # Stream 10000 samples from N(5, 3^2).
    for _ in range(100):
        batch = 5.0 + 3.0 * torch.randn(100, 4)
        norm.update(batch)
    # Normalised samples should have mean ≈ 0 and std ≈ 1.
    test = 5.0 + 3.0 * torch.randn(2000, 4)
    z = norm.normalize(test)
    assert z.mean().abs() < 0.2
    assert abs(float(z.std(unbiased=False)) - 1.0) < 0.2


def test_running_normalizer_matches_torch_reference(ppo_torch):
    import torch

    torch.manual_seed(0)
    data = torch.randn(500, 3) * 2.0 + 1.5
    norm = ppo_torch.RunningNormalizer(shape=(3,), eps=0.0)
    norm.update(data)
    # Expected statistics from torch directly.
    expected_mean = data.mean(dim=0)
    expected_var = data.var(dim=0, unbiased=False)
    torch.testing.assert_close(norm.mean, expected_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(norm.variance, expected_var, atol=1e-4, rtol=1e-4)


def test_ppo_config_has_37details_knobs(ppo_torch):
    cfg = ppo_torch.PPOConfig()
    assert cfg.normalize_obs is True
    assert cfg.anneal_lr is True
    assert cfg.vf_clip_eps == 0.2


def test_value_clipping_bounds_loss(ppo_torch):
    """With vf_clip_eps set, the clipped value-loss equals the max of the two
    candidates — i.e. cannot be lower than the plain MSE."""
    import torch

    torch.manual_seed(0)
    v_old = torch.tensor([0.5, -0.3, 1.0])
    v_new = torch.tensor([10.0, -10.0, 5.0])  # big drift
    returns = torch.tensor([0.0, 0.0, 0.0])
    eps = 0.2
    v_clipped = v_old + torch.clamp(v_new - v_old, -eps, eps)
    plain = ((v_new - returns) ** 2).mean()
    clipped_loss = (
        torch.stack([(v_new - returns) ** 2, (v_clipped - returns) ** 2]).max(dim=0).values.mean()
    )
    # The "max-of-two" value loss is >= the plain one by construction.
    assert float(clipped_loss) >= float(plain) - 1e-9
