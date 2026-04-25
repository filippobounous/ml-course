# Week 13 — LLMs as a development surface

## Learning objectives

1. Use **agentic-coding tools** (Claude Code, Cursor, Aider) as a power-user: scaffolding, refactoring, test-writing, debugging.
2. Understand what **MCP (Model Context Protocol)** is and build a minimal MCP server that exposes a local resource as a tool.
3. Design **reliable prompts for tool-use**: structured outputs, retries, failure modes.
4. Run an honest **LLM-as-judge** evaluation against a tuned model (from Week 9) and quantify judge variance.
5. Estimate **cost / latency / reliability** for an API-backed LLM feature before shipping it.

This module answers the piece of the course's original brief that earlier
weeks skipped: "using it like one would use Claude Code". Weeks 1–12 built
LLMs from the inside; this week uses them as an external tool.

## Topics

- Agentic-coding workflows (Claude Code, Cursor, Aider) — when each one wins.
- Prompt-engineering for reliability: few-shot, structured outputs, JSON mode, tool-use, retries, partial-output handling.
- **MCP servers.** Spec at a glance, anatomy of a server, tool definitions, resources vs tools vs prompts.
- **LLM-as-judge.** Ordinal vs cardinal interpretation, pairwise vs pointwise, judge variance, position bias, length bias. Anchoring to human agreement via pilot studies.
- **Cost / latency modelling.** Token cost per request, variance across requests, p50/p95/p99 latency. Budget-aware routing (cheap-first, fallback to strong).
- **Safety.** Jailbreaks in the wild, prompt injection, content filters, rate limits.

## Deliverables

- Portfolio artifact: `portfolio/13_dev_surface/` — three pieces, plus a demo:
  - `llm_judge.py`: a real LLM-as-judge wrapper (Anthropic SDK) with a mockable transport for tests.
  - `mcp_demo/`: a minimal MCP server exposing one tool (a calculator) + README on wiring it into Claude Code.
  - `cost_model.py`: token-cost + p50/p95 latency model and a worked example.
  - `demo.py`: end-to-end smoke — creates a mock judge, runs `portfolio/09_dpo_tinyllama/eval_harness.py` with it, and prints cost + latency projections.

## Reading plan

See `readings.md`.
