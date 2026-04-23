"""LLM-as-judge wrapper with a pluggable transport.

The transport is separated from the prompt logic so the harness can be
unit-tested with a fake transport (no API key, deterministic). Plug in the
real Anthropic SDK as the transport in production.

Compatible with `portfolio/09_dpo_tinyllama/eval_harness.py`: the
`make_llm_judge(...)` factory returns a `Judge` callable of signature
`(prompt, a, b) -> margin in [-1, 1]`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    winner: str  # "A" | "B" | "tie"
    raw_text: str


Transport = Callable[[str], str]
# A Transport takes the composed prompt and returns the raw model output.


DEFAULT_SYSTEM_PROMPT = """You are an impartial judge. Read the user's question and \
two candidate answers, then decide which answer is better. Reply with exactly one of: \
"Answer A", "Answer B", or "tie". Do not include reasoning."""


def compose_prompt(prompt: str, answer_a: str, answer_b: str) -> str:
    """Build the judge prompt (independent of transport for easy testing)."""
    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Question:\n{prompt}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Verdict:"
    )


def parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the first line of the model output for A / B / tie."""
    text = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    lower = text.lower()
    has_a = bool(re.search(r"\banswer\s+a\b", lower))
    has_b = bool(re.search(r"\banswer\s+b\b", lower))
    if has_a and not has_b:
        return JudgeVerdict(winner="A", raw_text=text)
    if has_b and not has_a:
        return JudgeVerdict(winner="B", raw_text=text)
    if "tie" in lower or "equal" in lower:
        return JudgeVerdict(winner="tie", raw_text=text)
    # Last resort: a lone single-letter token.
    if lower == "a" or lower.startswith("a "):
        return JudgeVerdict(winner="A", raw_text=text)
    if lower == "b" or lower.startswith("b "):
        return JudgeVerdict(winner="B", raw_text=text)
    return JudgeVerdict(winner="tie", raw_text=text)


def make_llm_judge(
    transport: Transport, *, average_positions: bool = True
) -> Callable[[str, str, str], float]:
    """Return a Judge callable compatible with the Week 9 eval harness.

    If `average_positions=True`, judge each pair twice with A/B swapped; count
    the verdict only when both orderings agree — the position-bias mitigation
    from Zheng 2024.
    """

    def judge(prompt: str, a: str, b: str) -> float:
        # Primary ordering: a = A, b = B.
        v1 = parse_verdict(transport(compose_prompt(prompt, a, b)))
        if not average_positions:
            if v1.winner == "A":
                return -1.0  # a (baseline) wins → margin is negative
            if v1.winner == "B":
                return 1.0
            return 0.0
        # Swap order: b = A, a = B.
        v2 = parse_verdict(transport(compose_prompt(prompt, b, a)))
        # Agreement needed for a non-tie verdict.
        if v1.winner == "A" and v2.winner == "B":
            return -1.0
        if v1.winner == "B" and v2.winner == "A":
            return 1.0
        return 0.0

    return judge


# -----------------------------------------------------------------------------
# Transports


def deterministic_keyword_transport(keywords: list[str]) -> Transport:
    """A deterministic fake transport for offline tests.

    Prefers whichever answer contains more of the target keywords; ties go
    to "tie". Keeps the harness runnable without an API key.
    """

    def _transport(full_prompt: str) -> str:
        # Extract the two answers from the composed prompt; cheap but robust
        # for unit tests that control prompt shape.
        a_match = re.search(r"Answer A:\n(.*?)\n\nAnswer B:", full_prompt, re.DOTALL)
        b_match = re.search(r"Answer B:\n(.*?)\n\nVerdict:", full_prompt, re.DOTALL)
        a = a_match.group(1) if a_match else ""
        b = b_match.group(1) if b_match else ""
        sa = sum(k.lower() in a.lower() for k in keywords)
        sb = sum(k.lower() in b.lower() for k in keywords)
        if sa > sb:
            return "Answer A"
        if sb > sa:
            return "Answer B"
        return "tie"

    return _transport


def anthropic_transport(*, model: str = "claude-sonnet-4-6", max_tokens: int = 16) -> Transport:
    """Real Anthropic API transport. Requires `anthropic` SDK + `ANTHROPIC_API_KEY`."""

    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError("anthropic SDK not installed. Run `pip install anthropic`.") from e

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set; cannot call the real API.")

    client = anthropic.Anthropic()

    def _transport(prompt: str) -> str:  # pragma: no cover - hits the network
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return _transport
