"""Week 9 — DPO loss, Chinchilla helpers, LoRA sizing, eval harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SOLUTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "modules" / "09_llms_dpo" / "problems" / "solutions.py"
)
EVAL_PATH = (
    Path(__file__).resolve().parents[2] / "portfolio" / "09_dpo_tinyllama" / "eval_harness.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sols():
    return _load(SOLUTIONS_PATH, "w9_solutions")


@pytest.fixture(scope="module")
def harness():
    return _load(EVAL_PATH, "w9_eval_harness")


# -- DPO loss ------------------------------------------------------------------


def test_dpo_loss_at_identity_equals_ln2(sols):
    n = 5
    zeros = np.zeros(n)
    loss, acc = sols.dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert loss == pytest.approx(np.log(2), abs=1e-12)
    assert acc == 0.0  # strict > 0 comparison, no winners


def test_dpo_loss_monotone_in_margin(sols):
    """Larger chosen - rejected ratio should reduce loss."""
    base = np.zeros(4)
    weak, w_acc = sols.dpo_loss(base + 0.5, base - 0.5, base, base, beta=1.0)
    strong, s_acc = sols.dpo_loss(base + 5.0, base - 5.0, base, base, beta=1.0)
    assert strong < weak
    assert s_acc >= w_acc


def test_dpo_loss_matches_pytorch_when_available(sols):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 8
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
    beta = 0.2
    ours, _ = sols.dpo_loss(a, b, c, d, beta=beta)

    # Torch reference using the same formula.
    margins = torch.tensor(beta * ((a - c) - (b - d)))
    ref = -torch.nn.functional.logsigmoid(margins).mean().item()
    assert ours == pytest.approx(ref, rel=1e-10)


def test_reward_margin_shape(sols):
    n = 3
    zeros = np.zeros(n)
    margins = sols.dpo_reward_margin(zeros + 1, zeros, zeros, zeros, beta=0.5)
    assert margins.shape == (n,)
    assert np.allclose(margins, 0.5)


# -- Chinchilla ----------------------------------------------------------------


def test_chinchilla_optimal_tokens(sols):
    assert sols.chinchilla_optimal_tokens(1e9) == 2e10  # 20 tokens per param


def test_chinchilla_flops(sols):
    assert sols.chinchilla_flops(1e9, 2e10) == pytest.approx(6 * 1e9 * 2e10)


# -- LoRA counts ---------------------------------------------------------------


def test_lora_param_count(sols):
    assert sols.lora_param_count(4096, 4096, 16) == 16 * (4096 + 4096)


def test_lora_reduction_at_llm_scale(sols):
    reduction = sols.lora_param_reduction(4096, 16)
    # 2 * 16 / 4096 ≈ 0.78%
    assert 0.007 < reduction < 0.009


# -- Eval harness --------------------------------------------------------------


def test_length_preference_judge(harness):
    # Margin > 0 ⇒ candidate (b) wins. Judge prefers the shorter response.
    assert harness.length_preference_judge("p", "longlong", "short") > 0  # b shorter → win
    assert harness.length_preference_judge("p", "a", "bb") < 0  # b longer → lose
    assert harness.length_preference_judge("p", "same", "same") == 0


def test_evaluate_pairwise_and_aggregate(harness):
    prompts = ["p1", "p2", "p3"]

    def base(p: str) -> str:
        return "long baseline response"

    def cand(p: str) -> str:
        return "short"

    # With length_preference_judge the shorter one (cand) wins every time.
    results = harness.evaluate_pairwise(prompts, base, cand, harness.length_preference_judge)
    agg = harness.aggregate(results)
    assert agg.n == 3
    assert agg.win_rate == pytest.approx(1.0)
    assert agg.lose_rate == 0.0


def test_keyword_judge(harness):
    judge = harness.keyword_judge(["python", "torch"])
    # Candidate has 2 matches, baseline has 0 → positive margin.
    assert judge("p", "I like cats", "I use python and torch") > 0


def test_save_round_trip(harness, tmp_path):
    results = [harness.PairwiseResult("p", "a", "b", 0.5)]
    agg = harness.aggregate(results)
    out = tmp_path / "eval.json"
    harness.save(results, agg, out)
    data = json.loads(out.read_text())
    assert data["aggregate"]["n"] == 1
    assert data["per_prompt"][0]["prompt"] == "p"
