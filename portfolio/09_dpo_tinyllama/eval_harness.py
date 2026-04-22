"""Deterministic LLM-as-judge style eval harness (pairwise win rate).

Given two generators (baseline, candidate) and a list of prompts, produces a
score table. The judge can be:
  * an external LLM via a user-supplied callable (e.g. Claude / GPT API), or
  * a deterministic rubric function for offline / unit-testable behaviour.

This file is torch-free — only the generators (supplied by the caller) depend
on torch/transformers/MLX.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

Generator = Callable[[str], str]
Judge = Callable[[str, str, str], float]  # (prompt, a, b) -> margin in [-1, 1]


@dataclass
class PairwiseResult:
    prompt: str
    baseline: str
    candidate: str
    margin: float  # > 0 means candidate wins

    def winner(self) -> str:
        if self.margin > 0:
            return "candidate"
        if self.margin < 0:
            return "baseline"
        return "tie"


@dataclass
class AggregateResult:
    win_rate: float
    tie_rate: float
    lose_rate: float
    mean_margin: float
    n: int


def evaluate_pairwise(
    prompts: list[str],
    baseline: Generator,
    candidate: Generator,
    judge: Judge,
) -> list[PairwiseResult]:
    results: list[PairwiseResult] = []
    for prompt in prompts:
        a = baseline(prompt)
        b = candidate(prompt)
        margin = judge(prompt, a, b)
        results.append(PairwiseResult(prompt, a, b, margin))
    return results


def aggregate(results: list[PairwiseResult]) -> AggregateResult:
    if not results:
        return AggregateResult(0.0, 0.0, 0.0, 0.0, 0)
    margins = [r.margin for r in results]
    wins = sum(1 for m in margins if m > 0)
    ties = sum(1 for m in margins if m == 0)
    losses = sum(1 for m in margins if m < 0)
    n = len(results)
    return AggregateResult(
        win_rate=wins / n,
        tie_rate=ties / n,
        lose_rate=losses / n,
        mean_margin=sum(margins) / n,
        n=n,
    )


def save(results: list[PairwiseResult], aggregate_result: AggregateResult, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": asdict(aggregate_result),
        "per_prompt": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# A tiny deterministic judge for unit tests and offline smoke checks.


def length_preference_judge(_prompt: str, a: str, b: str) -> float:
    """Trivial deterministic judge: prefer the shorter non-empty answer.

    Lets us unit-test the harness with no external calls. Margin is in [-1, 1].
    """
    if not a and not b:
        return 0.0
    if not a:
        return 1.0
    if not b:
        return -1.0
    len_a, len_b = len(a), len(b)
    return (len_a - len_b) / max(len_a, len_b)


def keyword_judge(keywords: list[str]) -> Judge:
    """Score based on how many target keywords each completion contains."""

    def _judge(_prompt: str, a: str, b: str) -> float:
        sa = sum(k.lower() in a.lower() for k in keywords)
        sb = sum(k.lower() in b.lower() for k in keywords)
        total = max(sa + sb, 1)
        return (sb - sa) / total

    return _judge
