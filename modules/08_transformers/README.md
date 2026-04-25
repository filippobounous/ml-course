# Week 8 — Transformers from scratch — *capstone kickoff*

## Learning objectives

1. Derive **scaled dot-product attention** and **multi-head attention** from first principles.
2. Implement a **decoder-only transformer** (tiny GPT) in PyTorch, including BPE tokenization and causal masking.
3. Train it on **TinyStories** on MPS to produce coherent completions.
4. **Propose your capstone** (physics track or quant track) — runs in parallel with Weeks 9–12.

## Topics

- Attention: scaled dot-product, multi-head, why $\sqrt{d_k}$.
- Positional encodings: sinusoidal, learned, ALiBi, RoPE (brief).
- BPE tokenization (`tokenizers` library) and its idiosyncrasies.
- Decoder-only vs encoder-only vs encoder-decoder architectures.
- Training mechanics: teacher forcing, autoregressive sampling (greedy / top-k / top-p / temperature).
- Attention map visualisation and circuit interpretation (teaser).

## Deliverables

- Portfolio artifact: `portfolio/08_tinygpt/` — tiny GPT (~10M params) on TinyStories with attention-map visualisations and a short technical write-up.
- **Capstone proposal**: one-page doc in `capstone/proposal.md` with goal, success metric, dataset, compute budget.

## Reading plan

See `readings.md`.
