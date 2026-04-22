# 06 — Reusable `mlcourse.Trainer` harness

> Populated in Week 6. See `modules/06_pytorch_trainer/`.

## Problem
A reusable PyTorch training harness that every subsequent artifact
(Weeks 7–12) extends — gradient accumulation, mixed precision on MPS,
Hydra configs, W&B logging, deterministic checkpoints.

## Method
- `mlcourse.Trainer` + `TrainerConfig` (implemented in `src/mlcourse/trainer.py`).
- Hydra config tree under `src/mlcourse/configs/`.
- W&B logging guarded by env var.
- MPS-aware autocast and device detection.

## Results
*Latency / throughput table (CPU vs MPS), deterministic-checkpoint proof.*

## Reproduce
```bash
make -C portfolio/06_trainer reproduce
```

## Why this matters
Demonstrates research-grade engineering hygiene: reproducibility, configuration,
logging, checkpointing.
