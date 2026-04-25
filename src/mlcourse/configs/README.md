# Hydra config tree

Hydra configs used by the `mlcourse.Trainer` and downstream weeks.

## Files

- `trainer.yaml` — default `TrainerConfig` values. Compose or override from the
  command line with Hydra syntax (`trainer.lr=3e-4`, `trainer.max_epochs=20`).

Subsequent weeks add group subdirectories here (e.g. `model/`, `data/`) so
that each experiment's full config can be frozen to disk for reproducibility.
