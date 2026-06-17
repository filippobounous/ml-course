# Problem set solutions — Week 6 (implementation: #3–#5)

Reference solutions for the implementation problems. Problem #4's reference
implementation is the committed `src/mlcourse/trainer.py` — read it alongside this.
Throughput numbers below are **hardware-dependent**; treat them as ballpark intuition,
not verified targets.

## Problem 3 — Port the W5 MLP to PyTorch; MNIST; CPU vs MPS throughput

The Week-5 MLP was `MLP(n_in=2, n_outs=[8, 8, 1])` over scalar `Value`s. The PyTorch port
is the *same architecture* expressed as `nn.Module`s and vectorised over a batch:

```python
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mlcourse.trainer import Trainer, TrainerConfig


class MLP(nn.Module):
    def __init__(self, d_in=784, hidden=(128, 128), n_classes=10):
        super().__init__()
        sizes = [d_in, *hidden]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers += [nn.Linear(sizes[-1], n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))          # (B,1,28,28) -> (B,784)


def loaders(batch_size=128):
    tfm = transforms.ToTensor()
    train = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("data", train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size))


def throughput(device, epochs=1):
    train_loader, val_loader = loaders()
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = TrainerConfig(max_epochs=epochs, device=device, seed=0)
    t0 = time.perf_counter()
    Trainer(cfg).fit(model, train_loader, val_loader,
                     loss_fn=nn.functional.cross_entropy, optimizer=opt)
    dt = time.perf_counter() - t0
    return len(train_loader.dataset) * epochs / dt          # samples/sec


for dev in ("cpu", "mps"):
    print(dev, f"{throughput(dev):.0f} samples/s")
```

**What to expect — and the lesson.** MNIST on a 784→128→128→10 MLP is small and
memory-bound, so **MPS is often *slower* than CPU** at modest batch sizes: kernel-launch and
host↔device copy overhead dominate the tiny matmuls. MPS only pulls ahead once per-step
compute is large enough to amortise that overhead (bigger batches, or conv nets — W7).
Correctness check: ≥97% test accuracy in a couple of epochs. Systems lesson: **measure;
don't assume the accelerator wins.** (Problem #6 profiles *why*.)

## Problem 4 — Build `mlcourse.Trainer`

The reference implementation lives at `src/mlcourse/trainer.py`. The design decisions worth
internalising:

- **`fit(model, train_loader, val_loader=None, *, loss_fn, optimizer)`** — `loss_fn` and
  `optimizer` are *keyword-only* so call sites read unambiguously. `loss_fn=None` is the
  escape hatch for objectives that don't fit `(x, y) → loss`: the model returns its own
  scalar loss (W10 DDPM uses this).
- **Device** — `TrainerConfig(device="auto")` resolves via `detect_device()` (CUDA → MPS → CPU).
- **Gradient accumulation** — divide the loss by `grad_accum_steps`, and only `optimizer.step()`
  every `k` micro-batches; the `/k` restores mean-gradient semantics (see the worked example).
- **Gradient clipping** — `clip_grad_norm_` is applied *at the accumulation boundary, before*
  `step()`.
- **Mixed precision** — `torch.autocast(device_type=…, dtype=fp16 on mps/cuda, bf16 on cpu)`,
  gated by `mixed_precision`.
- **Determinism + checkpoints** — `seed_everything(seed, deterministic_torch=True)`, and the
  checkpoint round-trips model + optimiser + **RNG state**. The RNG state is what makes
  save→load→resume bit-identical (verified by `portfolio/06_trainer/demo.py` and the W7 slow
  integration test).
- **W&B** — `_init_wandb()` is a no-op unless `MLCOURSE_WANDB=1`, so the harness carries no
  hard dependency on `wandb`.

Acceptance: `tests/week_06/` asserts `fit` runs end-to-end on a tiny synthetic dataset and
that checkpoint → resume yields bit-identical weights.

## Problem 5 — LR sweep as a Hydra multirun

`portfolio/06_trainer/demo.py` is already a `@hydra.main` entry point reading
`src/mlcourse/configs/week06/trainer_demo.yaml`, so a learning-rate sweep needs **no code
change** — just multirun:

```bash
python portfolio/06_trainer/demo.py --multirun trainer.lr=1e-2,3e-3,1e-3,3e-4
```

Hydra creates one run directory per value under `multirun/<date>/<time>/<n>/`, each with the
**resolved `config.yaml` persisted next to it** — that snapshot is the reproducibility win
(you can rerun any point exactly). To pick the winner, read each run's `report.md` (the demo
writes final train/val loss there) or its `config.yaml`. For a non-grid search
(random / Bayesian), add the `hydra-optuna-sweeper` plugin and declare the search space in
the config — the `--multirun` mechanism is identical, only the sampler changes.
