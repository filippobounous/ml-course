"""Week 13 — LLM-judge wrapper, cost model, MCP demo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = ROOT / "portfolio" / "13_dev_surface" / "llm_judge.py"
COST_PATH = ROOT / "portfolio" / "13_dev_surface" / "cost_model.py"
MCP_PATH = ROOT / "portfolio" / "13_dev_surface" / "mcp_demo" / "server.py"
WEEK9_HARNESS = ROOT / "portfolio" / "09_dpo_tinyllama" / "eval_harness.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def judge():
    return _load(JUDGE_PATH, "w13_llm_judge")


@pytest.fixture(scope="module")
def cost():
    return _load(COST_PATH, "w13_cost_model")


@pytest.fixture(scope="module")
def harness():
    return _load(WEEK9_HARNESS, "w13_w9_eval_harness")


# -- compose_prompt / parse_verdict ---------------------------------------------


def test_compose_prompt_contains_both_answers(judge):
    full = judge.compose_prompt("What's 2+2?", "four", "5")
    assert "Answer A:\nfour" in full
    assert "Answer B:\n5" in full
    assert "Verdict:" in full


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Answer A", "A"),
        ("Answer B", "B"),
        ("A", "A"),
        ("B", "B"),
        ("tie", "tie"),
        ("they are equal", "tie"),
        ("I don't know", "tie"),
        ("", "tie"),
    ],
)
def test_parse_verdict_handles_various_outputs(judge, raw, expected):
    assert judge.parse_verdict(raw).winner == expected


# -- make_llm_judge -------------------------------------------------------------


def test_judge_returns_margin_in_range_with_keyword_transport(judge):
    transport = judge.deterministic_keyword_transport(["rope", "kl", "softmax"])
    j = judge.make_llm_judge(transport, average_positions=False)
    # Candidate has the keywords, baseline does not → positive margin.
    a = "generic answer without technical terms"
    b = "mentions rope and softmax and KL directly"
    assert j("prompt", a, b) > 0
    assert j("prompt", b, a) < 0
    assert j("prompt", a, a) == 0.0


def test_position_averaging_filters_unstable_verdicts(judge):
    # A transport that flips its decision based on position → averaging
    # should yield a tie.
    call_state = {"n": 0}

    def flaky(_prompt_text: str) -> str:
        call_state["n"] += 1
        return "Answer A" if call_state["n"] % 2 == 1 else "Answer A"
        # Always says "A". Then when A is baseline once and candidate once,
        # the two orderings give opposite results → tie.

    j = judge.make_llm_judge(flaky, average_positions=True)
    assert j("p", "baseline", "candidate") == 0.0


def test_judge_compatible_with_week9_harness(judge, harness):
    """End-to-end: plug our judge into the W9 eval harness and run."""
    transport = judge.deterministic_keyword_transport(["bravo"])
    j = judge.make_llm_judge(transport, average_positions=False)
    prompts = ["q1", "q2", "q3"]

    def base(_: str) -> str:
        return "alpha charlie"

    def cand(_: str) -> str:
        return "alpha bravo charlie"

    results = harness.evaluate_pairwise(prompts, base, cand, judge=j)
    agg = harness.aggregate(results)
    assert agg.n == 3
    assert agg.win_rate == 1.0  # candidate always contains the keyword


# -- cost model -----------------------------------------------------------------


def test_cost_per_request_matches_hand_calc(cost):
    pricing = cost.ModelPricing(
        name="test",
        input_per_1m=3.0,
        output_per_1m=15.0,
        cache_read_per_1m=0.3,
        p50_latency_s=0.5,
        p95_latency_s=2.0,
    )
    profile = cost.RequestProfile(input_tokens=1000, output_tokens=200, cache_hit_rate=0.8)
    # Input: 0.2*1000*3e-6 + 0.8*1000*0.3e-6 = 6e-4 + 2.4e-4 = 8.4e-4
    # Output: 200 * 15e-6 = 3e-3
    # Total: 3.84e-3
    got = cost.cost_per_request(pricing, profile)
    assert got == pytest.approx(3.84e-3, rel=1e-6)


def test_monthly_cost_scales_with_dau(cost):
    pricing = cost.CLAUDE_SONNET_4
    profile = cost.RequestProfile(input_tokens=1500, output_tokens=400, cache_hit_rate=0.8)
    small = cost.monthly_cost(
        pricing,
        profile,
        daily_active_users=1_000,
        actions_per_user_per_day=5,
        calls_per_action=2,
    )
    big = cost.monthly_cost(
        pricing,
        profile,
        daily_active_users=10_000,
        actions_per_user_per_day=5,
        calls_per_action=2,
    )
    # 10× users → 10× monthly cost (modulo rounding).
    assert big.monthly_cost_usd == pytest.approx(10 * small.monthly_cost_usd, rel=1e-6)
    assert big.monthly_cost_usd > 0


def test_compare_models_returns_per_model_report(cost):
    reports = cost.compare_models(
        [cost.CLAUDE_HAIKU_4_5, cost.CLAUDE_SONNET_4, cost.CLAUDE_OPUS_4_7],
        cost.RequestProfile(input_tokens=1500, output_tokens=400, cache_hit_rate=0.8),
        daily_active_users=10_000,
        actions_per_user_per_day=5,
    )
    assert len(reports) == 3
    # Haiku should cost strictly less than Sonnet, Sonnet less than Opus.
    assert reports[0].monthly_cost_usd < reports[1].monthly_cost_usd < reports[2].monthly_cost_usd


# -- MCP demo -------------------------------------------------------------------


def test_mcp_demo_lists_all_tools():
    try:
        mcp_mod = _load(MCP_PATH, "w13_mcp_server")
    except ImportError as exc:
        pytest.skip(f"mcp SDK not installed: {exc}")
    tools = mcp_mod._list_tools()
    names = [name for name, _doc in tools]
    assert names == ["add", "multiply", "compound_interest"]
    for _name, doc in tools:
        assert doc  # every tool has a docstring


def test_mcp_demo_tools_compute_correctly():
    try:
        mcp_mod = _load(MCP_PATH, "w13_mcp_server_exec")
    except ImportError as exc:
        pytest.skip(f"mcp SDK not installed: {exc}")
    # The functions are still callable directly.
    assert mcp_mod.add(2, 3) == 5
    assert mcp_mod.multiply(4, 5) == 20
    import math

    assert mcp_mod.compound_interest(100.0, 0.1, 0.0) == pytest.approx(100.0)
    assert mcp_mod.compound_interest(100.0, 0.1, 1.0) == pytest.approx(100.0 * math.exp(0.1))
