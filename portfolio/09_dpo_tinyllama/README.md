# 09 — DPO-tuned TinyLlama + eval harness

End-to-end modern LLM alignment: **SFT → DPO → eval → model card → Gradio
Space**, all on Apple Silicon via TRL / PEFT or (recommended) **MLX**.

## Layout

- `dpo_train.py` — TRL-based SFT + DPO LoRA pipeline. Works on MPS in fp16 or
  on CUDA. For Apple Silicon, prefer the MLX path below — 2-5× faster.
- `eval_harness.py` — torch-free pairwise LLM-as-judge harness. Pluggable
  generators + judge; produces a JSON report with aggregate win / tie / loss
  rates and per-prompt margins.
- `model_card_template.md` — drop-in model card following Mitchell et al. (2019).

## Reproduce (TRL on MPS)

```bash
python -m pip install -e ".[dl,llm,ops]"
huggingface-cli login                       # needed to download TinyLlama
python portfolio/09_dpo_tinyllama/dpo_train.py --quick   # smoke
python portfolio/09_dpo_tinyllama/dpo_train.py           # full ~2-4 h on MPS
```

## Reproduce (MLX — recommended on Apple Silicon)

```bash
python -m pip install mlx mlx-lm
# SFT:
python -m mlx_lm.lora \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train --data alpaca.jsonl \
  --iters 1000 --batch-size 1 --lora-layers 16
# DPO:
python -m mlx_lm.lora \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train --data ultrafeedback.jsonl \
  --dpo --iters 1000 --beta 0.1
```

See the `mlx-lm` README for data-format specifics.

## Eval harness

```python
from eval_harness import evaluate_pairwise, aggregate, save, length_preference_judge
results = evaluate_pairwise(prompts, baseline_gen, candidate_gen, judge=length_preference_judge)
save(results, aggregate(results), Path("eval.json"))
```

For real evaluation, swap `length_preference_judge` for a Claude / GPT-4 rubric-
based judge and curate 20–50 prompts covering the behaviours your DPO targets.
Report **win-rate** and its binomial confidence interval.

## Target numbers

- TinyLlama-1.1B-Chat after 1k DPO iters on UltraFeedback (2k samples):
  expect ~55–60% win-rate over the SFT baseline on a matched eval set.
- LoRA with r=16 on all attention projections: ~0.6% of base parameters.
- Memory footprint on MPS: ~4 GB (fp16 base) + adapter.

## Tests

`tests/week_09/` covers the DPO-loss NumPy reference, Chinchilla scaling
helpers, LoRA parameter counting, and the eval harness (deterministic
judges, aggregate stats, round-trip JSON serialisation).

## Model card + Gradio Space

After DPO:
1. Fill out `model_card_template.md` with the target task, eval numbers, and
   known failure modes.
2. `huggingface-cli upload <repo-id> output/dpo --repo-type model`.
3. Copy `gradio_app.py` (see `mlx-lm` docs) into an HF Space; configure it to
   point at your model repo.

## What I learned

*To be filled after running DPO end-to-end.*
