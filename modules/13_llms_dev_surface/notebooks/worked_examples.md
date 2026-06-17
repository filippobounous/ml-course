# Worked examples — Week 13 (LLMs as a development surface)

Three REPL-doable exercises over the artifact in `portfolio/13_dev_surface/`. All run
without an API key (deterministic fakes); only the real `anthropic_transport` needs one.

## Example 1 — LLM-as-judge: parsing verdicts and killing position bias

`parse_verdict` reads the *first line* of the model's reply and maps it to `A` / `B` / `tie`:

| Raw model output | `parse_verdict(...).winner` |
|---|---|
| `Answer B` | `B` |
| `Answer A is better` | `A` |
| `tie` | `tie` |
| `They are equal.` | `tie` (matches "equal") |
| `B` | `B` (lone-letter fallback) |

The contract — one of three tokens, no reasoning — is pinned in `DEFAULT_SYSTEM_PROMPT`:
that's "prompt as interface" (§2) in practice.

**Position-bias mitigation.** `make_llm_judge(transport, average_positions=True)` judges each
pair *twice* with the answers swapped, and returns a non-tie verdict only when both orderings
agree (Zheng 2024). Trace it with the offline keyword transport:

```python
from llm_judge import deterministic_keyword_transport, make_llm_judge

transport = deterministic_keyword_transport(["chain", "rule", "gradient"])
judge = make_llm_judge(transport, average_positions=True)
judge("explain backprop",
      "It goes downhill.",                                      # answer a (baseline)
      "Backprop applies the chain rule to get the gradient.")   # answer b
# -> +1.0   (b wins as "A" in ordering 1 and as "B" in ordering 2 -> they agree)
```

Margin convention (matches the Week-9 harness): `-1` = baseline `a` wins, `+1` = candidate
`b` wins, `0` = tie **or disagreement**. **Lesson:** a judge that flips when you swap the
order isn't measuring quality — averaging both orderings turns that bias into an honest `0`
instead of a coin-flip "win".

## Example 2 — Cost model: never ship an LLM feature un-budgeted

A feature calls the API twice per user action, 1500 input / 400 output tokens, 50% of input
served from cache:

```python
from cost_model import CLAUDE_SONNET_4, CLAUDE_HAIKU_4_5, RequestProfile, monthly_cost

prof = RequestProfile(input_tokens=1500, output_tokens=400, cache_hit_rate=0.5)
rep = monthly_cost(CLAUDE_SONNET_4, prof,
                   daily_active_users=10_000, actions_per_user_per_day=5, calls_per_action=2)
```

Per request (Sonnet-4 at \$3 / \$15 / \$0.30 per 1M input / output / cached-input):

$$
\underbrace{(750 \cdot 3 + 750 \cdot 0.3)}_{\text{input, half cached}}/10^6
\;+\; \underbrace{400 \cdot 15}_{\text{output}}/10^6
= 0.002475 + 0.006 = \$0.008475 .
$$

Scale: $10\text{k} \times 5 \times 2 \times 30 = 3{,}000{,}000$ requests/month →
**\$25,425/month** on Sonnet, **\$8,475** on Haiku-4.5 for the same profile. Output tokens are
~70% of the per-request cost (\$0.006 of \$0.0085), which is why "make the model terse" and
"cache the system prompt" are the two highest-leverage levers, and why routing easy calls to
Haiku is worth exactly 3× here. **Lesson:** at 10k DAU this is a five-figure line item, not
the "a few dollars" people guess — model it before you commit.

## Example 3 — A minimal MCP server, end to end

`portfolio/13_dev_surface/mcp_demo/server.py` registers three tools with `FastMCP`. The
protocol exchange a client (e.g. Claude Code) drives:

1. **`tools/list`** → the server returns tool schemas derived from the type hints + docstrings:
   `add(a, b)`, `multiply(a, b)`, `compound_interest(principal, rate, years)`.
2. The model picks one and sends **`tools/call`**:

   ```json
   {"method": "tools/call",
    "params": {"name": "compound_interest",
               "arguments": {"principal": 1000, "rate": 0.05, "years": 10}}}
   ```
3. The server runs the Python function and returns the result:
   $1000 \cdot e^{0.05 \cdot 10} = 1000\,e^{0.5} \approx \$1648.72$.

The decorator `@mcp.tool()` is the whole trick: it turns a typed Python function into a
JSON-schema'd tool the model can discover and call, with transport (stdio) handled by the SDK.
`_list_tools()` exists so unit tests can assert the registry without standing up a live server.
**Lesson:** an MCP tool is just a typed function + a docstring; the protocol is the boring,
reusable part — which is exactly why it beats bespoke per-client function-calling glue.
