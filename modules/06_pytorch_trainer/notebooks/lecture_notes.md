# Week 6 — PyTorch deep-dive + reproducibility stack (lecture notes)

*Reading pair: PyTorch MPS docs · The Annotated Transformer · Lightning / Hydra docs.*

---

## 1. The PyTorch mental model

Three core abstractions:

- **`Tensor`** — `ndarray` + `device` + `dtype` + `requires_grad`. The last piece is what makes autograd possible.
- **`nn.Module`** — a bag of learnable parameters and sub-modules with a `forward()` method. Implements `state_dict()` / `load_state_dict()` for checkpointing and `.parameters()` for optimisers.
- **`DataLoader`** — wraps a `Dataset` in a multi-process iterator with batching, shuffling, and pinned memory.

The `tensor.requires_grad` and `nn.Module.parameters()` are the two hooks optimisers depend on. `tensor.backward()` triggers the same reverse-mode autograd we built by hand in Week 5 — just vectorised on arrays instead of scalars.

## 2. Apple Silicon (MPS) gotchas

On M-series Macs, PyTorch uses Apple's Metal Performance Shaders via `device="mps"`.

- **fp16 vs bf16**: M1/M2 have fp16 but no bf16; M3+ has bf16. Mixed-precision training uses `torch.autocast(device_type="mps", dtype=torch.float16)`.
- **Non-deterministic ops**: some reductions have non-deterministic variants; for reproducibility set `torch.use_deterministic_algorithms(True, warn_only=True)`.
- **`torch.compile`** is flaky on MPS today — try it, but fall back to eager if compilation fails.
- **Environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`** lets ops with no MPS kernel fall back to CPU instead of erroring. Worth setting for exploration; turn off once you have a stable pipeline.
- **Memory**: MPS memory is shared with the CPU — close other apps before running anything large.

## 3. Reproducibility stack

There is no single lever — reproducibility is a discipline, not a flag.

1. **Seeds everywhere**: `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, DataLoader workers (via a `worker_init_fn`).
2. **Deterministic algorithms** (with the flag above).
3. **Configs under version control**. Hydra is our default: compose configs from groups, override from the command line, persist the resolved config alongside every run.
4. **Environment snapshots**: `pip freeze > environment.lock.txt` or, better, `uv pip compile pyproject.toml -o environment.lock.txt`.
5. **Experiment tracking**: Weights & Biases (or TensorBoard / MLflow). Log hyperparameters, git SHA, environment, metrics, and artefacts.
6. **Model cards** (Mitchell et al. 2019) and **dataset cards** (Gebru et al. 2018) — structured docs that travel with the model.

A workable rule: if your Friday self can rerun Monday's experiment and reproduce the number to 3 significant figures, you have enough reproducibility for a research notebook. If not, bring a lock file and a seed discipline before publishing a paper.

## 4. Training-loop patterns

### Basic shape

```python
for epoch in range(cfg.max_epochs):
    for batch in train_loader:
        logits = model(batch["x"])
        loss = loss_fn(logits, batch["y"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

### Gradient accumulation

When the real batch size does not fit in memory, accumulate gradients over $k$ mini-batches before stepping:

```python
for step, batch in enumerate(train_loader):
    loss = loss_fn(model(batch["x"]), batch["y"]) / accum_steps
    loss.backward()
    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Use it whenever you see exploding gradients (transformers, RNNs); it costs almost nothing.

### Mixed precision

```python
with torch.autocast(device_type="mps", dtype=torch.float16):
    loss = loss_fn(model(batch["x"]), batch["y"])
```

Without a `GradScaler` you may see underflow on fp16; with MPS this is not always stable — profile first.

### Checkpointing

`torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "rng": torch.random.get_rng_state()}, path)`. Resuming means loading all of these and calling the RNG setter — otherwise you lose determinism across checkpoints.

## 5. The `Trainer` we build this week

Requirements, from `portfolio/06_trainer/README.md`:

- Take a `Trainer(config)` and a `fit(model, train_loader, val_loader=..., *, loss_fn, optimizer)` call.
- Pluggable device detection (CUDA → MPS → CPU).
- Gradient accumulation, gradient clipping, mixed precision (MPS-aware).
- Deterministic seeding via `seed_everything(..., deterministic_torch=True)`.
- Checkpoint save / resume with RNG state round-tripped.
- W&B logging guarded by env var (off by default).
- Hydra configs live under `src/mlcourse/configs/` and get reused W7–W12.

This is the harness every subsequent week imports. Keep it simple; the later weeks will extend it with LR schedulers, EMA weights, and evaluation callbacks only when a week actually needs one.

## What to do with these notes

Work the problem set in `../problems/README.md`. Extend the `mlcourse.Trainer`
skeleton under `src/mlcourse/trainer.py`; the portfolio artifact in
`portfolio/06_trainer/` trains a small MLP on a toy regression task and
demonstrates deterministic checkpoint round-trip.
