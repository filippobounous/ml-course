# TODO — explicit gaps from the internal review

This file tracks the review recommendations that **Phase E did not close**.
The course is usable as-is; this is the explicit backlog for making it a real
20-hr/week intensive rather than a strong skeleton.

## High priority (pedagogy)

- **3× expansion of lecture notes.** Current density is ~1 page/week; a real
  grad-level week is 10–15 pages. Needs:
  - Worked examples for every non-trivial derivation.
  - Per-section time budgets (template in W1, W5, W10 — propagate to W2–W4, W6–W9, W11–W13).
  - Physics-bridge callouts like W1 / W5 / W10 (Fisher–Rao, adjoint method, tempered Gibbs). Missing in W2–W4, W6–W9, W11–W13.
- **Self-assessment rubrics** (5 yes/no questions) for every week — template in W1/W5/W10, missing elsewhere.
- **Solutions for theory problems.** Phase E shipped `modules/NN/problems/solutions_theory.md` for W1–W13. Some are sketchy (e.g. W4 Avellaneda–Lee cross-reference, W6 has only two solutions); expand.
- **Long-form problems** — each week should have at least one 6-8 hour problem that ties multiple ideas together. Currently absent.

## High priority (correctness / completeness)

- **Verify every compute claim on real hardware.** The `Verified vs aspirational` table in `README.md` has ⏳ entries for every torch-dependent artifact. Each needs a real run with a committed log.
- **Commit reference trained checkpoints** for W7 (ResNet-18 CIFAR-10), W8 (tiny GPT on TinyStories), W9 (DPO LoRA adapter), W10 (DDPM ε-model), W12 (PINN). Enables learners to skip training and still use the evaluation / Grad-CAM / sampling pipelines.
- **Wire `mlcourse.Trainer` into W10 and W11.** Phase E did it for W7 only; W10 (`train.py`) and W11 (`train_ppo.py`) still write bespoke training loops.
- **Full FID via InceptionV3** for W10 (currently a pixel-statistics proxy).
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
- **Paper-reproduction code** under `portfolio/12_capstone/paper_reproduction/` (currently only `PLAN.md`).

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
