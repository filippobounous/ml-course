# Problem set — Week 8

## Theory

1. **Attention gradient.** Derive $\partial L / \partial Q, \partial L / \partial K, \partial L / \partial V$ for scaled dot-product attention $A = \mathrm{softmax}(QK^\top / \sqrt{d_k}) V$.
2. **Softmax shift invariance.** Show softmax is invariant to adding a constant to all logits; explain why this matters for numerically stable implementations.
3. **Causal masking.** Prove the causal-masked attention output at position $t$ depends only on positions $\le t$.
4. **RoPE.** State rotary positional encoding mathematically and show the inner product of two RoPE-rotated vectors depends only on their relative position.

## Implementation (portfolio)

5. Implement **multi-head attention** from single-head; verify numerical equivalence to `torch.nn.functional.scaled_dot_product_attention` for the merged case.
6. Train a **BPE tokenizer** on TinyStories with the `tokenizers` library.
7. Build a **tiny GPT (~10M params)** with pre-LN, GELU, RoPE, and weight tying. Train on TinyStories to loss ≲ 2.0. Generate and report 5 coherent samples.

## Applied

8. **Attention-map visualisations** for 3 prompts; identify heads that appear to track syntactic structure (subject–verb, closing punctuation).

## Grading

Tests in `tests/week_08/` check: BPE round-trip (`decode(encode(x)) == x`); multi-head vs fused-sdpa numerical equivalence; causal mask application.
