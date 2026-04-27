# 06 — Reusable `mlcourse.Trainer` harness

A compact PyTorch training harness designed to be reused across every
subsequent week (Weeks 7–12). Lives in `src/mlcourse/trainer.py`.

## Features

- Device auto-detection (`cuda` → `mps` → `cpu`) via `mlcourse.utils.detect_device`.
- **Gradient accumulation** and **gradient clipping**.
- **Mixed precision** via `torch.autocast` (CUDA, MPS, or CPU-bf16).
- **Deterministic seeding** hooks into `torch`, `numpy`, `random`.
- **Checkpoint save / resume** with full RNG state round-trip — the demo
  verifies that reloading yields bit-identical model weights.
- **Optional Weights & Biases logging** gated by `MLCOURSE_WANDB=1`.
- **Hydra configs** under `src/mlcourse/configs/` (the demo is a `@hydra.main`
  entry point — every knob is overridable from the command line).

## Layout

```
src/mlcourse/
├── trainer.py                    ← Trainer + TrainerConfig
└── configs/
    ├── trainer/default.yaml      ← base TrainerConfig (group default)
    └── week06/trainer_demo.yaml  ← composes /trainer:default

portfolio/06_trainer/
├── demo.py                       ← @hydra.main entry point
└── README.md
```

## Reproduce

```bash
python -m pip install -e ".[dl,ops]"
python portfolio/06_trainer/demo.py                         # defaults
python portfolio/06_trainer/demo.py trainer.max_epochs=1    # CI smoke
python portfolio/06_trainer/demo.py trainer.lr=1e-3 data.batch_size=32
```

If torch is not installed the script prints a friendly skip message and exits.

## Tests

`tests/week_06/` covers:

- `TrainerConfig` defaults / overrides.
- `detect_device()` returns a valid string.
- `Trainer.fit` runs end-to-end on a tiny tensor dataset.
- Save → fresh-model → load → weights match bit-for-bit.

## Why it's shippable

Demonstrates research-grade engineering hygiene: reproducibility (seeds + RNG
round-trip), configuration (Hydra), logging (W&B opt-in), and MPS-aware
mixed precision — ready for the much larger artefacts in Weeks 7–12.
