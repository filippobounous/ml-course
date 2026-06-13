# Model card — Tiny GPT (TinyStories)

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

## Model details

- **Name.** Tiny GPT on TinyStories.
- **Architecture.** Decoder-only transformer, ~10 M parameters.
  Pre-LN + GELU + learned positional embeddings + weight-tied embedding/head.
  (RoPE is derived in the W8 lecture notes as theory; the shipped model uses
  learned positions for simplicity.)
- **Framework.** PyTorch from scratch (the point of W8 is the from-
  scratch implementation, not `transformers`).
- **Hyperparameters.** AdamW $\beta_1{=}0.9, \beta_2{=}0.95$, weight
  decay 0.1; constant LR (warmup + cosine schedule left as an exercise); batch 64;
  ~1 epoch over TinyStories.

## Intended use

- **Primary.** Demonstrate end-to-end transformer engineering: BPE
  tokenizer training, multi-head attention from scratch, causal
  masking, positional embeddings, training loop, sampling.
- **Out-of-scope.** Generating realistic English text for any
  purpose. This is a small model on a stylistically narrow dataset
  (TinyStories — 4-year-old-reading-level fiction); its outputs are
  not robust to even mildly out-of-distribution prompts.

## Metrics

- **Training cross-entropy loss** (target ≲ 2.0 nats).
- **Generated-sample coherence** — qualitative judgement on 5 prompts.
- **Attention-map plots** for 3 prompts; identification of heads
  tracking syntactic structure.

## Training / evaluation data

- **TinyStories** (Eldan & Li 2023). ~ 4 M short stories at
  4-year-old reading level. Synthetic; no PII; English-only.

## Quantitative analyses

| Metric | Target | Verified |
|---|---|---|
| Train loss after 1 epoch | ≲ 2.0 | ⏳ (aspirational; needs hardware run) |
| Coherent samples on 3+ of 5 prompts | yes | ⏳ |
| Training runtime (MPS) | ~6 h | ⏳ |

## Caveats

- TinyStories is *not* a benchmark for general language modelling.
  Don't extrapolate from this model's behaviour to assumptions about
  GPT-3.5 / Llama at scale.
- The from-scratch BPE tokenizer is byte-level (GPT-2 style, `ByteLevel`
  pre-tokenizer with no normalisation), so `encode(decode(x)) == x` round-trips
  exactly. It is still trained only on TinyStories, so don't use it as a
  drop-in for production tokenization.
- A 10 M parameter model at the Chinchilla-optimal $D = 20 N$ would
  need ~200 M tokens; this run is well under-trained.
