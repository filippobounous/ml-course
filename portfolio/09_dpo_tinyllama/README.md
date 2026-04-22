# 09 — DPO-tuned TinyLlama + Gradio Space

> Populated in Week 9. See `modules/09_llms_dpo/`.

## Problem
SFT + DPO-align a small open LLM (TinyLlama-1.1B) on Apple Silicon with LoRA,
publish a HuggingFace model card, and ship a live Gradio demo.

## Method
- SFT with LoRA on Alpaca-cleaned (via HF TRL or `mlx-lm`).
- DPO on UltraFeedback (≤5k preference pairs).
- MT-Bench-style rubric with 30 prompts; LLM-as-judge harness.
- HF model card + dataset card + Gradio Space.

## Results
*Win-rate table SFT-vs-DPO; sample completions; model card URL.*

## Reproduce
```bash
make -C portfolio/09_dpo_tinyllama reproduce
```

## Why this matters
Modern LLM alignment (DPO, not PPO-RLHF) with a deployed public demo.
