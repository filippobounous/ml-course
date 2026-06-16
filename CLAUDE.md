# CLAUDE.md

Guidance for AI agents working in this repo. Keep it accurate — if you change how the
project builds, tests, or is laid out, update this file.

## What this is

A **12-week core ML/AI curriculum (W1–W12) plus an optional Week 13**, for quantitatively
literate learners. Two things live here:

1. **The course** — `modules/NN_<name>/` (lectures + problems) and `portfolio/NN_<name>/`
   (shippable artifacts). This is what a learner works through, W1 → W13.
2. **The Python package** `mlcourse` (in `src/mlcourse/`) — reusable code the artifacts
   import (autograd engine, `Trainer`, Hydra configs).

Runs on **CPU / Apple-Silicon MPS** — no CUDA. Every demo is meant to fit in ~1 hour on an
M-series Mac.

## Setup & commands

Editable install with extras (hatchling, `src/` layout):

```bash
pip install -e ".[dev]"        # base + dev (pytest, ruff, mypy)
make week-5                     # install just what week N needs, then points at its README
```

Extras: `dl` (torch/lightning), `llm` (transformers/peft/trl, +mlx on macOS), `diffusion`,
`rl` (gymnasium), `sciml` (torchdiffeq/pysr/arch), `ops` (hydra/wandb/xgboost/lightgbm/gradio),
`docs`, `all`.

Make targets: `test`, `test-slow`, `test-week-N`, `lint` (ruff + mypy), `format`,
`docs` / `docs-serve`, `portfolio-build`, `fetch-data`, `docker-dev`, `clean`.

## Running tests — read this first

- The package must import as `mlcourse`. Either `pip install -e ".[dev]"` **or** run with
  `PYTHONPATH=src`. A bare `pytest` from a fresh checkout will fail with
  `ModuleNotFoundError: mlcourse`.
- `pyproject.toml` sets `minversion = "8.0"`. If the ambient interpreter has older pytest,
  install `[dev]` into a clean env — don't routinely bypass the config.
- Slow tests are gated: `@pytest.mark.slow` is skipped unless you pass `--run-slow`
  (defined in `tests/conftest.py`). They train real models (e.g. W5 micrograd on two-moons).
- CI (`.github/workflows/ci.yml`): a `test` job (ruff + mypy + `pytest --cov`, Python 3.11 &
  3.12), a `docs-build` job (`mkdocs build --strict`), and a `test-dl` job
  (torch + `pytest --run-slow`).

## Layout

```
modules/NN_<name>/
  README.md  readings.md
  notebooks/lecture_notes.md  notebooks/worked_examples.md
  problems/README.md  problems/solutions_theory.md  problems/solutions.py
portfolio/NN_<name>/        # README.md, demo.py (one-command run), model_card.md, figures
src/mlcourse/              # trainer.py, autograd/, models/, data/, utils/, configs/ (Hydra)
tests/week_NN/             # per-week unit + (slow) integration tests
scripts/build_docs.py      # stages markdown into a gitignored docs/ tree for mkdocs
```

## Conventions that bite

- **Honesty table** (`README.md` "Verified vs aspirational"): never flip a row ⏳ → ✅
  without committing a real run/log to `portfolio/<artifact>/verified.md`. Most second-half
  metrics are still aspirational.
- **Taught vs implemented:** a topic appears in headline claims / model cards **only if the
  artifact implements it.** Some topics are taught as theory but deliberately not in the
  shipped artifact (e.g. W8 RoPE — model uses learned positions; W8 warmup/cosine LR — uses
  constant LR; W5 BatchNorm — not in the scalar autograd). These are labeled at each artifact
  and in the README honesty table; keep that discipline.
- **Model cards:** every trained artifact carries a `model_card.md` (Mitchell et al. 2019
  schema; template at `portfolio/model_card_template.md`).
- **Docs:** markdown in `modules/`, `portfolio/`, and root is the source of truth;
  `scripts/build_docs.py` stages it into a gitignored `docs/` for mkdocs. Don't hand-edit
  `docs/`.
- **Style:** ruff (line-length 100, Greek-letter identifiers allowed) + mypy on `src/`. Run
  `make format` before committing.

## Weeks vs waves (don't confuse them)

- **Weeks (W1–W13):** the learner curriculum. Sequential; each week ships an artifact.
- **Waves (PR_PLAN.md):** a backlog of ~38 PRs that polish the course. Orthogonal to the
  weeks.

## Status & planning docs — trust code, not the backlog

- `REVIEW.md` — module-by-module review: prioritized findings + a code-verified status table.
- `PR_PLAN.md` §Implementation status — **authoritative** per-PR completion (verified against
  the repo).
- `TODO.md` — the backlog. Reconciled 2026-06-16, but it has drifted before; **cross-check the
  code before trusting any status claim here.**
- `SYLLABUS.md` / `STUDY_GUIDE.md` / `PORTFOLIO.md` — curriculum plan, how to work through it,
  artifact showcase. `CHANGELOG.md` — release history.
