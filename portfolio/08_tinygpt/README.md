# 08 — Tiny GPT on TinyStories

A ~10M-parameter decoder-only transformer trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) —
all on CPU / Apple Silicon.

## Layout

- `model.py` — `GPTConfig`, `GPT`, `Block`, `CausalSelfAttention`, `MLP`.
  Pre-LN, GELU, learned positional embeddings, tied LM head.
- `train.py` — BPE tokenizer trainer (HF `tokenizers`), data loader,
  AdamW + cosine LR, gradient clipping, periodic eval. Writes a checkpoint,
  a sample file, and a training history JSON.

## Reproduce

```bash
python -m pip install -e ".[dl,llm,ops]"

# 1) Download TinyStories and concatenate the shards:
huggingface-cli download roneneldan/TinyStories --repo-type dataset \
  --local-dir data/tinystories_raw
cat data/tinystories_raw/*.json.gz > data/tinystories.txt   # use jq/zcat to extract text fields

# 2) Train (~6 h on M-series, much longer on CPU):
python portfolio/08_tinygpt/train.py \
  --corpus portfolio/08_tinygpt/data/tinystories.txt \
  --max-iters 4000
```

Use `--max-iters 200 --n-layer 2 --d-model 128` for a CI smoke check in a few minutes.

## Target loss regime

- 500 iters: val loss ~4.0 (random-uniform would be ~log(vocab)=9.2).
- 2000 iters: val loss ~2.5, first coherent continuations.
- 4000 iters: val loss ~2.0, clearly in the TinyStories regime.

## Outputs

- `checkpoint.pt` — model + config + history.
- `tokenizer.json` — trained BPE.
- `samples.md` — a handful of 100-token generations from the prompt
  "Once upon a time,".
- `history.json` — train/val loss at each eval step.

## Tests

`tests/week_08/` covers:
- NumPy multi-head attention + softmax stability + RoPE rel-position invariance.
- Byte-level BPE `decode(encode(x)) == x` round-trip.
- Torch-gated: GPT forward-pass shape, causal-mask correctness,
  `generate()` appends the requested number of tokens.

## What I learned

*To be filled in. Suggested bullets: how much work a tokenizer is; why
weight-tying + pre-LN matter; how the loss plateaus before generations
become coherent; how attention maps visibly track syntactic structure.*
