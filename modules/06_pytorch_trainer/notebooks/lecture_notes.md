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

### Profiling a step

When a step is slower than expected, *measure* — don't guess. `torch.profiler` records per-op time:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(10):
        step()   # one forward / backward / optimiser step
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

Sort by **self** time (an op's own time, not its children's) — that's what you optimise.
Two MPS-specific rules: **synchronise** (`torch.mps.synchronize()`) before reading timings,
because MPS is asynchronous; and **warm up** a few steps first so one-off kernel compilation
doesn't dominate the table. On MPS, kernels currently surface under CPU activities (there is
no `ProfilerActivity.MPS` yet); for true on-device kernel timing reach for Apple's
Instruments. (Worked through in applied problem 6.)

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

**Before the problem set**, walk through [`worked_examples.md`](worked_examples.md) — three concrete REPL-doable exercises (`.detach()` vs `.clone()` traps, gradient accumulation = effective-batch-size $k \cdot B$ math, checkpoint round-trip determinism with all 6 RNG-state pieces).

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1 PyTorch mental model | 2 | Tensor / Module / DataLoader; play with `.requires_grad`, `.detach()`. |
| §2 MPS gotchas | 2 | Run a tiny CPU-vs-MPS benchmark; toggle deterministic algorithms. |
| §3 Reproducibility | 2 | Seed everything; verify with a "did seeding work?" script. |
| §4 Training-loop patterns | 3 | Implement grad accumulation, clipping, autocast on a toy MLP. |
| §5 Build the Trainer | 6 | Implement `mlcourse.Trainer.fit`; ship the portfolio demo. |
| Problem set + W&B | 3 | Hydra multirun LR sweep; W&B logging guarded by env var. |
| Office hours / review | 2 | Cross-check against `problems/solutions_theory.md`. |

## Self-assessment rubric

Before moving to Week 7, you should be able to answer "yes" to all of:

1. Can I explain `requires_grad`, leaf tensors, `.detach()` vs `.clone()`, and the difference between `.backward()` with and without `retain_graph`?
2. Can I describe which ops `torch.autocast("mps")` casts to fp16 and which it leaves in fp32, and explain why both halves matter?
3. Can I write a deterministic training loop where checkpoint → restart produces bit-identical weights, and list the six RNG / state pieces that need to be saved?
4. Can I implement gradient accumulation correctly and explain why dividing the loss by `accum_steps` is what restores mean-gradient semantics?
5. Can I configure a Hydra-driven training script with command-line overrides and a multirun LR sweep?

## Physics bridge

For a theoretical physicist, the most useful re-framings:

- **Reverse-mode autograd on a tensor graph ≡ adjoint method** (same connection as W5, now vectorised on arrays instead of scalars). `tensor.backward()` solves the adjoint equation; `requires_grad=True` is the "tape this trajectory" flag.
- **Mixed precision ↔ multi-scale physics.** You compute the bulk (matmul, conv) cheaply in fp16, but **accumulate in fp32** wherever precision matters (softmax, layer-norm, loss reductions). Same principle as keeping high precision only on the small set of slow / sensitive degrees of freedom while treating the bulk in a coarse-grained approximation — the asymptotic expansion is the physicist's analogue.
- **Gradient accumulation ↔ Riemann sum.** Each backward pass adds a single mini-batch's gradient into `.grad`; averaging over $k$ mini-batches converges to the population gradient like a finite Riemann sum converges to an integral. The $/k$ factor is the integration step $\Delta t$.
- **Determinism ↔ time-reversal symmetry of integrators.** Saving the full RNG state along with weights and optimiser is the discrete analogue of demanding a *symplectic* integrator: it lets you walk back from $t+1$ to $t$ exactly. Without RNG state, the dynamics are irreversible — same as adding numerical viscosity to an N-body integrator.

Keep these bridges live; W7–W12 all import `mlcourse.Trainer`, and every torch-dependent week reuses the same checkpoint-determinism + autocast pattern.
