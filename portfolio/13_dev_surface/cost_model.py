"""Token-cost + latency model for API-backed LLM features.

Not a real benchmark — a deliberately simple first-order model that gives
learners a defensible estimate before shipping anything. Plug in your own
pricing by constructing a `ModelPricing` instance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """USD price per 1M tokens, plus rough p50/p95 latency in seconds."""

    name: str
    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float  # cached input tokens
    p50_latency_s: float
    p95_latency_s: float


# Rough 2025 published prices. Verify against current docs before citing.
CLAUDE_SONNET_4 = ModelPricing(
    name="claude-sonnet-4",
    input_per_1m=3.0,
    output_per_1m=15.0,
    cache_read_per_1m=0.3,
    p50_latency_s=0.5,
    p95_latency_s=2.0,
)
CLAUDE_HAIKU_4_5 = ModelPricing(
    name="claude-haiku-4.5",
    input_per_1m=1.0,
    output_per_1m=5.0,
    cache_read_per_1m=0.1,
    p50_latency_s=0.3,
    p95_latency_s=1.2,
)
CLAUDE_OPUS_4_7 = ModelPricing(
    name="claude-opus-4.7",
    input_per_1m=15.0,
    output_per_1m=75.0,
    cache_read_per_1m=1.5,
    p50_latency_s=1.0,
    p95_latency_s=4.0,
)


@dataclass(frozen=True)
class RequestProfile:
    input_tokens: int
    output_tokens: int
    cache_hit_rate: float  # 0..1, fraction of input tokens served from cache


@dataclass
class CostReport:
    model: str
    cost_per_request_usd: float
    monthly_cost_usd: float
    n_requests_per_month: int
    p50_latency_s: float
    p95_latency_s: float


def cost_per_request(pricing: ModelPricing, profile: RequestProfile) -> float:
    """USD per request under the given pricing and request profile."""
    cached = profile.cache_hit_rate * profile.input_tokens
    uncached = (1 - profile.cache_hit_rate) * profile.input_tokens
    input_cost = (
        uncached * pricing.input_per_1m / 1_000_000 + cached * pricing.cache_read_per_1m / 1_000_000
    )
    output_cost = profile.output_tokens * pricing.output_per_1m / 1_000_000
    return input_cost + output_cost


def monthly_cost(
    pricing: ModelPricing,
    profile: RequestProfile,
    *,
    daily_active_users: int,
    actions_per_user_per_day: int,
    calls_per_action: int = 1,
    days_per_month: int = 30,
) -> CostReport:
    """Aggregate cost report for a simple DAU × actions workload."""
    per_req = cost_per_request(pricing, profile)
    n_requests = daily_active_users * actions_per_user_per_day * calls_per_action * days_per_month
    return CostReport(
        model=pricing.name,
        cost_per_request_usd=per_req,
        monthly_cost_usd=per_req * n_requests,
        n_requests_per_month=n_requests,
        p50_latency_s=pricing.p50_latency_s,
        p95_latency_s=pricing.p95_latency_s,
    )


def compare_models(
    pricings: list[ModelPricing],
    profile: RequestProfile,
    *,
    daily_active_users: int,
    actions_per_user_per_day: int,
    calls_per_action: int = 1,
) -> list[CostReport]:
    """Side-by-side cost reports for a list of models."""
    return [
        monthly_cost(
            p,
            profile,
            daily_active_users=daily_active_users,
            actions_per_user_per_day=actions_per_user_per_day,
            calls_per_action=calls_per_action,
        )
        for p in pricings
    ]
