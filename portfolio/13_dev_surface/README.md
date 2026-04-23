# 13 — LLMs as a development surface

The piece of the original course brief that Weeks 1–12 skipped: using LLMs
as an **external tool** — Claude Code, MCP servers, LLM-as-judge, cost/latency
modelling — rather than building them from scratch.

## What's here

- `llm_judge.py` — pairwise LLM-as-judge with a pluggable transport.
  - `compose_prompt` / `parse_verdict` — pure functions, trivially tested.
  - `make_llm_judge(transport, average_positions=True)` — returns a `Judge`
    callable that plugs into the Week 9 eval harness (same protocol as
    `length_preference_judge` / `keyword_judge`).
  - `deterministic_keyword_transport` — offline, reproducible transport for
    unit tests and CI.
  - `anthropic_transport` — calls the real Anthropic API; needs
    `ANTHROPIC_API_KEY` and `pip install anthropic`.
- `cost_model.py` — `ModelPricing`, `RequestProfile`, `monthly_cost`,
  `compare_models`. Simple first-order cost + latency model for API-backed
  features; three published pricings (Haiku 4.5, Sonnet 4, Opus 4.7) are
  included as constants.
- `mcp_demo/server.py` — minimal Model Context Protocol server exposing
  `add` / `multiply` / `compound_interest` tools. See the sub-README for
  how to wire it into Claude Code / Claude Desktop.
- `demo.py` — runs the Week-9 harness with a mock judge and prints a cost
  projection for running the same eval via Claude Sonnet 4 at scale.

## Reproduce

```bash
# Zero dependencies beyond the course base env:
python portfolio/13_dev_surface/demo.py

# With real API calls:
pip install anthropic
export ANTHROPIC_API_KEY=...
python -c "from llm_judge import make_llm_judge, anthropic_transport; ..."
```

## Why this exists

A 2026 "ML engineer" role is roughly 50% the stuff in Weeks 1–12 (training,
architectures, alignment) and 50% the stuff here (prompt engineering,
evaluation rigour, cost modelling, agentic-coding fluency). Shipping a course
that only covers the first half is dishonest.

## Tests

`tests/week_13/test_llm_judge_and_mcp.py` verifies:

- `compose_prompt` contains both answers in the expected shape.
- `parse_verdict` correctly handles "Answer A", "Answer B", "tie", and
  ambiguous outputs (falls back to tie).
- `make_llm_judge` returns `-1 / 0 / +1` margins consistent with the keyword
  transport.
- Position-averaging rejects unstable verdicts when the two orderings
  disagree.
- `cost_model.cost_per_request` matches a hand-computed number.
- `mcp_demo.server._list_tools` returns all three expected tools.

## Relationship to other weeks

- Takes **W9's `eval_harness.py`** as its eval substrate — the LLM judge is a
  drop-in for `length_preference_judge` / `keyword_judge`.
- Generalises the **W11 tool-use agent** from a hand-written policy to a
  real LLM + MCP tool server.
- Surfaces the **honest-metrics** thread from W9/W10 by explicitly measuring
  judge variance and position bias in the problem set.

## What I learned

*To be filled after running Claude Code on the W13 refactor exercise.*
