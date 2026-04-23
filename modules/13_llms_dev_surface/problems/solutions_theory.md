# Week 13 — Theory-problem solutions

## 1. Judge-variance flip rate

An acceptable threshold depends on the metric you're reporting.

- **Ranking only ("A > B on this eval set").** Up to ~20% flip rate is tolerable if the underlying win-rate gap is large (say > 60/40). Compute a bootstrap CI on the aggregate win-rate; if the CI excludes 50% at the 95% level, the ranking claim is defensible.
- **Headline numbers ("DPO wins 64.3% of pairs").** The flip-rate bounds how precise that number can be. If flip rate on individual (prompt, trial) pairs is $f$, the 95% CI half-width on the aggregate over $N$ prompts is roughly $1.96 \sqrt{f(1-f)/N}$. For $f = 0.15, N = 50$ that's ±10 percentage points — ugly.
- **Practical rule.** If you're citing a number beyond one significant figure, your judge should agree with itself at $> 90\%$ consistency on clear cases. Run the same pair 5× at temperature 0.7, drop prompts where the judge can't self-agree, and use the cleaned set.

## 2. Position bias

Zheng 2024 found ~4% position bias for GPT-4-0613 on MT-Bench. Modern Claude / GPT-4-turbo variants are often closer to 1–3%, but **always measure**. Mitigations:

- **Randomise per-pair.** Flip a coin on the order for each (prompt, SFT, DPO) triple before sending to the judge.
- **Average both orderings.** Run each pair twice, once in each order; count the verdict only if the two runs agree on the winner. This throws away ambiguous cases but gives bias-free ranking.
- **Rubric anchoring.** Ask the judge to score each completion against a written rubric on a 0–10 scale first, *then* ask for the preference. Position bias is much smaller for rubric-based scores than for preference-based ones.

## 3. Cost estimate (worked example)

Anthropic pricing as of mid-2025 (check the docs for current numbers):

- Claude Sonnet 4 input: ~$3 / 1M tokens → 333k tokens / $1.
- Claude Sonnet 4 output: ~$15 / 1M tokens → 66k tokens / $1.
- Prompt caching: cache reads ~$0.30 / 1M tokens (10% of base), writes same as base.

Assume 80% cache-hit rate. Effective input cost per request:
$1500 \text{ tok} \times (0.2 \cdot 3 \times 10^{-6} + 0.8 \cdot 0.3 \times 10^{-6}) = 1500 \times 0.84 \times 10^{-6} \approx \$1.26 \times 10^{-3}.$

Output cost per request: $400 \times 15 \times 10^{-6} = \$6 \times 10^{-3}$.

Total per request: ~$7.3 \times 10^{-3}$ = **0.73¢**. Two calls per action: ~1.5¢/action.

Scale: $10\text{k DAU} \times 5 \text{ actions/day} \times 30 \text{ days} \times 0.015 = \$22{,}500/\text{month}$.

Back-of-envelope error bars: ±2× because the 1500/400 token count is an educated guess before measuring, and real cache-hit rates fluctuate. Plan accordingly.

**Takeaway.** At 10k DAU, a Sonnet-4 feature is a ~$20k/month line item, about a senior engineer's loaded cost for a week. Haiku is ~5× cheaper, Opus is ~5× more expensive. Route cheap-first, fallback to strong. Evaluate each tier's quality separately.
