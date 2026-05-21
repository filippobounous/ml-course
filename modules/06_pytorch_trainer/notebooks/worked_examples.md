# Week 6 — Worked examples

Concrete walk-throughs to accompany `lecture_notes.md`. Each fits in
~15 minutes at a Python REPL.

---

## Example 1 — `.detach()` vs `.clone()` vs `.detach().clone()`

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()                       # graph: x -> square -> sum
```

Four ways to "get the value out":

| Op | Same storage? | In graph? | Gradient flows? |
|---|---|---|---|
| `x` | yes | yes | yes |
| `x.detach()` | yes (shared memory!) | no | no |
| `x.clone()` | no (copy) | yes | yes |
| `x.detach().clone()` | no (copy) | no | no |

### Trap 1 — mutating a detached tensor

```python
z = x.detach()
z[0] = 99.0           # also changes x[0]! Shared storage.
print(x)              # tensor([99.0, 2.0, 3.0], requires_grad=True)
y.backward()          # gradient computation now sees the mutated x
```

If you wanted "value of x, don't perturb the graph," use
`x.detach().clone()`. This is *the* most common autograd footgun.

### Trap 2 — `retain_graph` when reusing a loss

```python
loss = (x ** 2).sum()
loss.backward()                  # graph freed
loss.backward()                  # RuntimeError: ... saved tensors freed
```

Two fixes: re-run the forward pass, or `loss.backward(retain_graph=True)`. Use the latter only when you really need it — it doubles memory pressure on transformers.

---

## Example 2 — Gradient accumulation produces effective batch size $k \cdot B$

Train a tiny MLP on 80 samples, real batch size $B = 8$, accumulation steps $k = 4$. Effective batch size = 32.

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)
for step, (x, y) in enumerate(loader):
    pred = model(x)
    loss = loss_fn(pred, y) / accum_steps          # ← KEY divisor
    loss.backward()                                # accumulates into .grad
    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### Why divide by `accum_steps`?

Each call to `.backward()` *adds* gradients into `.grad`. After $k$ mini-batch backward passes, the accumulated gradient is

$$
\sum_{j=1}^{k} \nabla_\theta \tfrac{1}{B} \sum_{i=1}^{B} \ell(x_i^{(j)}, y_i^{(j)}) = k \cdot \nabla_\theta \!\left[\tfrac{1}{B} \sum_{j, i} \ell\right].
$$

Without the `/k` factor, the optimizer would step with a gradient $k\times$ larger than the per-sample-mean gradient — equivalent to a $k\times$ learning rate, which usually destabilises training. The `/accum_steps` divisor restores the **mean-gradient semantics**.

### Quick sanity check

For a quadratic $\ell(x; \theta) = \tfrac{1}{2}(\theta - x)^2$ with $\theta = 0$ and $x = 1$:

- Without grad accum, one batch of size 32 gives $\nabla_\theta \bar\ell = -\bar x$.
- With $k = 4$ grad accum and $B = 8$, each mini-batch gives $-\bar x_j / k$, summed over $k$ mini-batches → $-(1/k) \sum_j \bar x_j = -\bar x$ overall.

Same gradient → same step. ✓

---

## Example 3 — Checkpoint round-trip determinism

```python
import torch
from mlcourse.trainer import Trainer, TrainerConfig
from mlcourse.utils import seed_everything

seed_everything(0)
model = build_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 1) Train for a few steps and save.
trainer = Trainer(TrainerConfig(max_epochs=3, seed=0, deterministic=True))
trainer.fit(model, train_loader, val_loader, loss_fn=loss_fn, optimizer=opt)
ckpt = trainer.save_checkpoint("step3.pt")
state_before = {k: v.clone() for k, v in model.state_dict().items()}

# 2) Build a fresh model + optimiser, load, compare.
model2 = build_model()
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
trainer2 = Trainer(TrainerConfig(seed=0, deterministic=True))
trainer2.load_checkpoint("step3.pt", model=model2, optimizer=opt2)

# 3) Bit-identical?
for name, p1 in model.state_dict().items():
    p2 = model2.state_dict()[name]
    assert torch.equal(p1, p2), f"mismatch in {name}"
```

### What must be saved for true bit-identity?

Six things:

1. `model.state_dict()` — the weights.
2. `optimizer.state_dict()` — Adam's $m$, $v$, step counter.
3. `torch.random.get_rng_state()` — torch CPU RNG.
4. `torch.cuda.get_rng_state_all()` if you use CUDA.
5. `np.random.get_state()` if you use numpy in the data pipeline.
6. `random.getstate()` if you use Python random.

If you save (1) and (2) only and rerun, the *weights* are identical at $t = 0$, but the next step's RNG (e.g. dropout, data shuffling) starts from a different state — and the trajectories diverge. The W6 Trainer saves at least (1), (2), and (3); add (4)-(6) if your pipeline uses them.

### Trap — non-deterministic ops

Even with all RNG saved, some operations (notably some MPS / CUDA reductions and `index_add`) are non-deterministic by default. Force determinism with

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

and accept the ~5–10% slowdown for the privilege.

---

## What to do with these examples

For Example 1, swap `.detach()` for `.detach().clone()` in a real
script and notice the (small) memory increase — that's the cost of
the deep copy. For Example 2, double `accum_steps` and *also* halve
your batch size, then verify the loss curves overlap — confirmation
that gradient accumulation is purely a memory trick, not a different
optimisation algorithm. For Example 3, deliberately omit the RNG
state from the checkpoint and watch the divergence — useful to *see*
where determinism comes from.
