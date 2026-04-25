"""End-to-end smoke for the Week 13 artifact.

Runs the Week-9 pairwise evaluation harness with a mockable LLM judge,
then prints a cost projection for the same evaluation under Claude Sonnet 4.

Everything runs offline — no API key required. Swap
`deterministic_keyword_transport(...)` for `anthropic_transport(...)` to hit
a real model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

WEEK9_HARNESS = HERE.parent / "09_dpo_tinyllama" / "eval_harness.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    week9 = _load(WEEK9_HARNESS, "w9_eval_harness")
    from cost_model import CLAUDE_SONNET_4, RequestProfile, monthly_cost
    from llm_judge import deterministic_keyword_transport, make_llm_judge

    # Five synthetic prompts + two fake "generators" to exercise the judge.
    prompts = [
        "Write a haiku about neural nets.",
        "Explain diffusion models in one paragraph.",
        "What is RoPE and why does it generalise to longer contexts?",
        "Describe the DPO loss derivation.",
        "Explain the Bellman contraction property.",
    ]

    def baseline_gen(p: str) -> str:
        return "A verbose answer that rambles on without using technical terms."

    def candidate_gen(p: str) -> str:
        return (
            "A concise technical answer mentioning attention, RoPE, positional "
            "encoding, KL, softmax and contraction as needed."
        )

    judge_transport = deterministic_keyword_transport(
        keywords=["attention", "rope", "kl", "softmax", "contraction", "technical"]
    )
    judge = make_llm_judge(judge_transport, average_positions=True)

    results = week9.evaluate_pairwise(prompts, baseline_gen, candidate_gen, judge=judge)
    aggregate = week9.aggregate(results)

    print("=== Pairwise evaluation ===")
    print(f"n = {aggregate.n}  win={aggregate.win_rate:.2f}  tie={aggregate.tie_rate:.2f}")

    # Cost projection: the same eval, but running the judge via Sonnet 4.
    profile = RequestProfile(input_tokens=1500, output_tokens=32, cache_hit_rate=0.8)
    report = monthly_cost(
        CLAUDE_SONNET_4,
        profile,
        daily_active_users=10_000,
        actions_per_user_per_day=5,
        calls_per_action=2,
    )
    print("\n=== Cost model — 10k DAU × 5 actions × 2 calls on Sonnet 4 ===")
    print(f"model:              {report.model}")
    print(f"cost / request:     ${report.cost_per_request_usd:.5f}")
    print(f"requests / month:   {report.n_requests_per_month:,}")
    print(f"monthly cost:       ${report.monthly_cost_usd:,.0f}")
    print(f"p50 latency:        {report.p50_latency_s:.2f} s")
    print(f"p95 latency:        {report.p95_latency_s:.2f} s")


if __name__ == "__main__":
    main()
