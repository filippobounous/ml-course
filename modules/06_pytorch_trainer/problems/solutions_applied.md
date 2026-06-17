# Problem set solutions — Week 6 (applied: #6)

## Problem 6 — Profile a training step on MPS with `torch.profiler`

Reuse `MLP` and `loaders` from [`solutions_implementation.md`](solutions_implementation.md).

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = MLP().to("mps")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
xb, yb = next(iter(loaders()[0]))
xb, yb = xb.to("mps"), yb.to("mps")


def step():
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    torch.mps.synchronize()        # MPS is async — sync so the timings are real


for _ in range(3):                 # warm up: discard one-off kernel compilation
    step()

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(10):
        step()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

**How to read it.** Sort by *self* time (an op's own time, excluding children) — that is
what you optimise. Total time double-counts wrappers (`aten::linear` calls `aten::addmm`),
so it points you at the wrapper, not the work. For this MLP the top-3 self-time entries are
typically some mix of:

1. `aten::addmm` / `aten::mm` — the linear-layer matmuls (forward and backward).
2. `aten::copy_` / host↔device transfers — often surprisingly large at small batch, which is
   the real reason MPS trails CPU here (problem #3).
3. the Adam optimiser's elementwise kernels (`aten::add_`, `aten::addcdiv_`).

**Three MPS-specific gotchas.**

- **Synchronise before reading timings.** MPS dispatches asynchronously; without
  `torch.mps.synchronize()` the profiler mis-attributes time to a later sync point.
- **Warm up.** The first few steps pay one-off kernel-compilation cost — discard them or the
  top of the table is compilation, not steady state.
- **MPS kernels surface under CPU activities** in current PyTorch (there is no
  `ProfilerActivity.MPS` analogous to `.CUDA`). You're seeing dispatch + any CPU-fallback
  ops; for true on-device kernel timing, use Apple's Instruments (Metal System Trace).

**The bottleneck note (what the problem asks for).** On this MLP the bottleneck is *not*
compute — it's launch + transfer overhead, which is exactly why batching harder or moving to
a compute-heavy model (W7's ResNet) is what makes MPS pay off. Profiling is how you *discover*
that instead of guessing — the same instinct you'll need when a transformer step in W8 is
unexpectedly slow.
