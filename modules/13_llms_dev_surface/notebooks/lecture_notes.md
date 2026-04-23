# Week 13 — LLMs as a development surface (lecture notes)

*Reading pair: Anthropic Claude Code docs · Model Context Protocol spec · Zheng 2024 MT-Bench · Dubois 2024 AlpacaEval 2.*

---

## 1. Two worlds: building LLMs vs using them

Weeks 1–12 built the machinery — transformers from scratch, DPO alignment, diffusion samplers, PPO agents. This week switches perspective: we use a frontier LLM (Claude, GPT-4, Gemini) as a black box and ask what changes when it's your **development surface**, not your implementation target.

Two distinct modes matter:

- **Agentic coding** — the LLM writes / edits / tests code in your repo. Tools: Claude Code, Cursor, Aider. Here you mostly want **reliability, speed, and good blast-radius intuition** (what files are touched, are edits atomic, what to revert).
- **LLM-backed features** — your app calls an API at inference time. Here you need **cost modelling, latency budgeting, evaluation rigour, and prompt-engineering discipline**.

This note focuses on the second mode; the first is best learned by using Claude Code for a week.

## 2. Prompt-engineering for reliability

The biggest shift from "LLM chatbot user" to "LLM engineer" is **treating prompts as interfaces with contracts**, not text. Concretely:

- **Structured outputs.** Tell the model exactly what schema to return (JSON, XML tags). Validate on the way out; retry with a "your last output was invalid JSON, please fix" turn. The Anthropic SDK has `tools`; OpenAI has `response_format=json_object`.
- **Few-shot examples.** 2–5 worked examples in the system prompt beat clever instructions. Pick diverse examples; include edge cases.
- **Failure modes to anticipate.** (a) truncation at max_tokens, (b) refusal, (c) hallucinated tool names, (d) tool-call loops, (e) subtle numeric errors. Build a retry harness for each.
- **Temperature.** 0 for extraction / classification; 0.7 for "help me write an email"; never just "default".

## 3. Model Context Protocol (MCP) in one page

MCP is a JSON-RPC protocol (Anthropic 2024) that lets any LLM client (Claude Desktop, Claude Code, Cline, your bespoke agent) connect to any external tool server without custom glue. A server exposes:

- **Tools** — function calls the model can invoke (`calculator`, `run_sql`, `send_email`).
- **Resources** — read-only state the model can reference by URI (`file:///path/to/doc`).
- **Prompts** — named, parameterised prompt templates the client can fetch.

Transport is usually stdio (Claude Desktop, Claude Code) or HTTP/SSE (remote servers). The Python SDK (`mcp` package) wraps transport + schema + retries.

A minimal server (see `portfolio/13_dev_surface/mcp_demo/`):

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

Wire into Claude Code by editing `~/.config/claude-code/mcp.json` (or the CLI's config path) to list the server command. Claude Code auto-discovers the tool and makes it available to the model.

## 4. LLM-as-judge — what to measure honestly

**Pointwise** vs **pairwise** scoring. Pairwise (A vs B on the same prompt) is the standard. It mostly removes calibration drift between runs.

**Known biases** (Zheng 2024):
- **Position bias** — the judge prefers the first completion more often than chance. Fix: randomise order; always score both orderings and average.
- **Length bias** — the judge prefers longer answers. Fix: normalise by length, or use rubric-anchored prompts.
- **Self-preference** — the judge prefers completions from its own family. Fix: use a different family as the judge when possible.

**Variance estimation.** Run the judge on the same pair 5–10 times at `temperature=0.7`. If the verdict flips > 30% of the time, your judge signal is too noisy for the metric you're reporting. Common in 2024; expect ~10–20% flip rate for a well-designed rubric on clear comparisons.

**Agreement with humans.** Anchor via a pilot study: collect 50 human-scored pairs, compute the judge's agreement rate. If agreement is < 70%, the automated metric is not reliable enough for anything but ranking; if > 85%, you can cite single numbers with confidence intervals.

## 5. Cost / latency / reliability model

Simple unit economics per request (Anthropic 2025 pricing as a worked example, check current docs):

| Quantity | Claude Sonnet 4 | Claude Opus 4 |
|---|---|---|
| Input tokens / $1 | ~300k | ~66k |
| Output tokens / $1 | ~66k | ~15k |
| p50 latency (streaming start) | ~0.5 s | ~1.0 s |
| p95 latency (streaming start) | ~2 s | ~4 s |

For a product that calls the API twice per user action at 1500 input / 400 output tokens (after prompt caching), typical monthly cost for 10k DAU × 5 actions/user/day is on the order of $50–500. Model this up-front; learners consistently under-budget by 5–10×.

**Reliability.** Rate-limit errors and transient 5xx happen. Use exponential backoff (factor 2, max 5 retries). Budget for 1–2% unrecoverable failures per day; display them as fallbacks, not crashes.

## 6. Agentic coding in practice (Claude Code notes)

- Spawn **subagents** (`/agents ...`) for searches and plans; reserve the main context for the delta you're implementing.
- **Hooks** (`settings.json`) can wrap every tool call — useful for running linters after edits, or injecting "remember X" context at session start.
- **Skills** are reusable lightweight tools loaded on demand (e.g. "claude-api" skill). Add a skill when you want Claude Code to remember a domain-specific workflow.
- **Permissions.** Decline network / secret access by default; grant narrowly and temporarily.
- **Context economy.** Treat your context window as a scarce resource. Long read-only context → dispatch to a subagent; don't inline.

The course exercise in `problems/README.md` turns this into a rubric: you'll run Claude Code on a small refactor, review its diff, and document which of the 37-details subset of prompt-engineering rules you consciously relied on.

## What to do with these notes

Work the problem set in `../problems/README.md`. Ship the portfolio artifact in `../../../portfolio/13_dev_surface/`: a runnable MCP server, a mockable LLM-as-judge wrapper, and a cost model demo that ties back into the Week 9 eval harness.
