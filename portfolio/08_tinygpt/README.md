# 08 — Tiny GPT on TinyStories

> Populated in Week 8. See `modules/08_transformers/`.

## Problem
Train a decoder-only transformer (~10M parameters) from scratch on
**TinyStories** and show coherent generation — entirely on MPS / Apple Silicon.

## Method
- BPE tokenizer trained on TinyStories (`tokenizers` library).
- Decoder-only transformer: multi-head attention, pre-LN, GELU, RoPE,
  weight tying.
- Training via the Week-6 `Trainer` harness.

## Results
*Training curve, 5 sampled continuations, a few attention-map panels.*

## Reproduce
```bash
make -C portfolio/08_tinygpt reproduce
```

## Why this matters
End-to-end transformer pipeline (tokenizer → model → training → generation)
done by hand. High research-signal.
