# TODO — explicit gaps from the internal review

This file tracks the review recommendations that **Phase E did not close**.
The course is usable as-is; this is the explicit backlog for making it a real
20-hr/week intensive rather than a strong skeleton.

> **See [`PR_PLAN.md`](PR_PLAN.md) for the concrete plan that breaks this
> backlog into ~38 focused pull requests grouped into 7 waves with explicit
> dependencies, acceptance criteria, and a minimum-viable-honest landing
> order.**

## High priority (pedagogy)

- **Finish the starter/reference split.** Problem-set code is now gated: the
  learner implements `modules/NN_*/problems/starter.py` and `pytest` grades
  *that*, with the answer key moved to `problems/_reference/solutions.py` and
  reachable via `MLCOURSE_SOLUTIONS=reference` (see `tests/conftest.py`). Nine
  weeks are covered (W1, W3, W4, W7, W8, W9, W10, W11, W12). **Still
  ungated:**
  - Every **portfolio artifact** — `portfolio/*/` ships complete, including
    the committed figures and `report.md` / `results.md` the learner is
    supposed to generate. Tests for W2, W4, W7–W13 load those files directly,
    so they pass on a fresh clone.
  - **W5 autograd** (`src/mlcourse/autograd/`) and **W6 `Trainer`**
    (`src/mlcourse/trainer.py`) — the two headline "build it yourself"
    artifacts, both shipped working.
  - **W2, W5, W6, W13** have no `problems/starter.py` at all; their tests
    target library or portfolio code only.

  The same mechanism extends to these: add `portfolio/NN_*/starter_*.py`,
  move the shipped implementation under `_reference/`, and point the
  artifact tests at the loader fixture.

- **3× expansion of lecture notes.** Current density is ~1 page/week; a real
  grad-level week is 10–15 pages. Needs:
  - Worked examples for every non-trivial derivation. Present for W1–W12;
    **W13 has none**.
  - Per-section time budgets. Present for W1–W12; **W13 has none**.
  - Physics-bridge callouts (Fisher–Rao, adjoint method, tempered Gibbs).
    Present for W1–W12; **W13 has none**.
- **Self-assessment rubrics** (5 yes/no questions) — present for W1–W12;
  **W13 has none**.
- **Solutions for theory problems.** Phase E shipped `modules/NN/problems/solutions_theory.md` for W1–W13. Some are sketchy (e.g. W4 Avellaneda–Lee cross-reference, W6 has only two solutions); expand.
- **Long-form problems** — each week should have at least one 6-8 hour problem that ties multiple ideas together. Currently absent.

## High priority (correctness / completeness)

- **Verify every compute claim on real hardware.** The `Verified vs aspirational` table in `README.md` has ⏳ entries for every torch-dependent artifact. Each needs a real run with a committed log.
- **Commit reference trained checkpoints** for W7 (ResNet-18 CIFAR-10), W8 (tiny GPT on TinyStories), W9 (DPO LoRA adapter), W10 (DDPM ε-model), W12 (PINN). Enables learners to skip training and still use the evaluation / Grad-CAM / sampling pipelines.
- **Wire `mlcourse.Trainer` into W11.** Done for W7 and W10; W11 (`train_ppo.py`) still writes a bespoke training loop.
- **MLX-native DPO path** for W9 (currently only described in README — `mlx-lm` commands cited but not implemented).

## Medium priority (missing topics)

- **Distributed training** module (or section) — FSDP / ZeRO / tensor parallelism mental model, even without a GPU.
- **Quantisation** — int8, GPTQ, AWQ, MLX 4-bit. Especially relevant for the Apple-Silicon learner.
- **Graph neural networks** — good bridge to physics (message passing ↔ lattice models) and quant (transaction graphs).
- **Causal inference / do-calculus** — essential for scientific-ML claims.
- **Modern time-series** (TFT, PatchTST, TimesNet) — the W12 finance track currently only teases them.
- **Speculative decoding / MoE / KV-caching** — all currently unmentioned.
- **Interpretability at depth** — SHAP, probing classifiers, attention rollout, sparse autoencoders.
- **Safety / red-teaming / jailbreaks** — touched in W13 readings; no hands-on exercise.

## Medium priority (infra)

- **CI matrix**. Current CI runs `ruff + mypy + pytest` on ubuntu-latest. Add a `--run-slow` job and install the `[dl,ops]` extras so Tier-B integration tests run on every push.
- **Coverage reporting**. `pytest-cov` is installed but not used; publish coverage as a badge.
- **Docs rendering**. mkdocs or Sphinx so the lecture notes + module READMEs render with cross-references, math, and figures.
- **Full Hydra refactor** of every training script (W6 scaffolds it; only the W6 demo actually uses it).
- **Gradio Space** for W9 — README gives deployment instructions; no runnable `gradio_app.py` is committed.

## Lower priority (nice-to-have)

- **Dataset cards** alongside model cards in W9.
- **Docker image** with the `[dl,llm,diffusion,rl,sciml,ops]` extras pre-installed.
- **Changelog**. Per-week `CHANGELOG.md` so a reader can see what landed when.
- **Per-week notebook** versions of the lecture notes (currently Markdown only; some topics benefit from interactive plots).

## Phase E delivered (for reference)

Closed in `claude/ml-ai-course-design-JlOPK`:

1. ✅ Tier-B `slow` pytest infrastructure + an example W5 integration test.
2. ✅ Theory-problem solutions (`solutions_theory.md`) for W1–W13.
3. ✅ W7 refactored to use `mlcourse.Trainer`.
4. ✅ Classifier-free guidance in W10 (model + samplers + tests).
5. ✅ PPO "37 details" subset in W11 (obs normalisation, adv normalisation, LR annealing, value clipping + tests).
6. ✅ GradNorm loss reweighter for W12 PINN + test.
7. ✅ New Week 13 module: LLMs as a development surface (notes + MCP demo + LLM-judge + cost model + demo).
8. ✅ Pedagogical block (time budget + rubric + physics bridge) in W1, W5, W10.
9. ✅ Verified-vs-aspirational table in top README.
10. ✅ This `TODO.md`.
