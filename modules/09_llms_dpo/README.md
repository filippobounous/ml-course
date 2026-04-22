# Week 9 — Language models at scale: SFT, DPO, and MLX

## Learning objectives

1. Become fluent in the **HuggingFace ecosystem**: `transformers`, `datasets`, `tokenizers`, `peft`, `trl`, `accelerate`.
2. Read the **scaling-laws** literature and reproduce tiny-scale versions.
3. Perform **supervised fine-tuning (SFT)** with **LoRA** on a small instruction dataset.
4. Align a model with **Direct Preference Optimization (DPO)** — the modern alternative to RLHF-PPO.
5. Use Apple-native **MLX / mlx-lm** for fast LoRA and inference on Apple Silicon.
6. Ship a **Gradio Space** exposing your tuned model.

## Topics

- HF API surface: tokenizers, models, datasets, Trainer vs TRL SFTTrainer.
- Scaling laws (Kaplan, Chinchilla) — replicate the qualitative shape at tiny scale.
- Parameter-efficient fine-tuning: LoRA, QLoRA (note: bitsandbytes has no MPS support).
- Preference optimisation: reward modelling, PPO-RLHF (theory only), **DPO**, ORPO, KTO.
- Evaluation harnesses: MT-Bench-style rubrics, lm-eval-harness (skim).
- Model cards, dataset cards, and Gradio Spaces.

## Deliverables

- Portfolio artifact: `portfolio/09_dpo_tinyllama/` — DPO-tuned TinyLlama-1.1B (LoRA) + eval harness + HF model card + Gradio Space.

## Reading plan

See `readings.md`.
