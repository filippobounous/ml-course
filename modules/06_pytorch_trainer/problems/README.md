# Problem set — Week 6

## Theory

1. **Autograd bookkeeping.** Explain `requires_grad`, leaf tensors, `.detach()` vs `.clone()`, and the difference between `.backward()` with and without `retain_graph`.
2. **Mixed precision on MPS.** Describe, with references, which dtypes are supported on M1/M2 vs M3. What does `torch.autocast("mps")` actually cast, and what does it leave in fp32?

## Implementation (portfolio)

3. Port the **W5 MLP to PyTorch**. Train on MNIST; compare CPU vs MPS throughput.
4. Build **`mlcourse.Trainer`** in `src/mlcourse/trainer.py`:
   - construct with `Trainer(config=TrainerConfig(...))`, then train via
     `fit(model, train_loader, val_loader=None, *, loss_fn, optimizer)`
     (a `None` `loss_fn` means the model returns its own scalar loss)
   - gradient accumulation, gradient clipping, mixed precision (MPS-aware)
   - checkpointing, resume, deterministic seeding
   - W&B logging (optional, guarded by env var)
   - Hydra-powered configs under `src/mlcourse/configs/`
5. Write a **learning-rate sweep** as a Hydra multirun.

## Applied

6. **Profile** a training step on MPS with `torch.profiler`. Identify the top-3 ops by self-time and write a short note on bottlenecks.

## Grading

Tests in `tests/week_06/` check that `Trainer.fit` runs end-to-end on a tiny synthetic dataset and that checkpoint → resume produces bit-identical weights (given determinism flag).

## Solutions

- Theory (#1–2): [`solutions_theory.md`](solutions_theory.md)
- Implementation (#3–5): [`solutions_implementation.md`](solutions_implementation.md)
- Applied (#6, `torch.profiler`): [`solutions_applied.md`](solutions_applied.md)
