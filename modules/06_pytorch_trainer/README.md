# Week 6 — PyTorch deep-dive + reproducibility stack

## Learning objectives

1. Use **PyTorch idiomatically**: `nn.Module`, `DataLoader`, training loops, hooks, profiling.
2. Understand the **MPS backend**: dtypes on M1/M2/M3, `torch.autocast("mps")`, `torch.compile` caveats.
3. Set up a **reproducibility stack** that makes every subsequent artifact recruiter-grade: seeds, Hydra configs, W&B logging, model cards.
4. Ship a **reusable `mlcourse.Trainer`** that Weeks 7–12 extend.

## Topics

- PyTorch internals: tensors, autograd, `nn.Module`, parameters / buffers, state_dict.
- DataLoaders, `Dataset`, `Sampler`, workers; transforms and augmentations.
- Training-loop patterns: gradient accumulation, gradient clipping, mixed precision.
- Apple Silicon performance: MPS backend, fp16/bf16 notes per chip generation, `torch.compile` limitations.
- Reproducibility: seeds, `torch.use_deterministic_algorithms`, config systems (Hydra / OmegaConf).
- Experiment tracking: W&B (free tier), model cards, dataset cards.

## Deliverables

- Portfolio artifact: `portfolio/06_trainer/` — reusable `mlcourse.Trainer` harness with Hydra configs under `src/mlcourse/configs/`. Used by every subsequent week.

## Reading plan

See `readings.md`.
