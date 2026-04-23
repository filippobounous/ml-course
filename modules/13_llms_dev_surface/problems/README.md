# Problem set — Week 13

## Theory / reflection

1. **Judge variance.** Pick 10 prompts from your Week-9 DPO eval set. Get pairwise LLM-judge verdicts (SFT vs DPO) at temperature 0 and at temperature 0.7, 5 trials each. Report the flip-rate (fraction of (prompt, trial) pairs where the two temperatures disagree). Hypothesise an acceptable flip rate for publishing a headline number.
2. **Position-bias measurement.** Re-judge the same 10 prompts with SFT and DPO in the opposite order. Compute the fraction of verdicts that flip purely due to ordering. If > 5%, adopt order-averaging for the main eval.
3. **Cost estimate.** For an app that calls Claude Sonnet 4 twice per user action (1500 input / 400 output tokens, 80% prompt cache hit), at 10k DAU × 5 actions/user/day, compute the monthly API bill.

## Implementation

4. **MCP server.** Write a minimal MCP server that exposes a single tool: `run_sqlite(query: str, db_path: str) -> str`. Use `fastmcp` or the lower-level `mcp.server` API. Wire it into Claude Code (`~/.config/claude-code/mcp.json`) and verify Claude Code can query a bundled SQLite DB.
5. **LLM-as-judge with a mockable transport.** Build on `portfolio/13_dev_surface/llm_judge.py`. Separate "compose prompt + parse verdict" from "call the API". Inject a fake transport in tests; verify the prompt contains both completions and the parser correctly extracts `+1 / -1 / 0`.
6. **Reproduce Week 9's win-rate table at two temperatures.** Using the harness from `portfolio/09_dpo_tinyllama/eval_harness.py` + your new judge, show a side-by-side at temperature 0 vs 0.7 and a bootstrap 95% CI on each.

## Applied

7. **Claude Code refactor exercise.** Pick one file from `portfolio/`. Open Claude Code and ask it to refactor the file for clarity without changing behaviour. Diff the result; write a short `findings.md` covering (a) which prompt-engineering choices mattered, (b) one thing Claude Code did well, (c) one thing it did poorly, (d) how you would structure the prompt next time.

## Grading

Tests in `tests/week_13/` check: (a) the judge wrapper handles a fake verdict stream correctly; (b) the cost model produces a plausible monthly total on the worked example; (c) the MCP demo server registers its tool schema.
