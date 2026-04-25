# PR Plan — closing the review backlog

This file breaks every item in `TODO.md` into an explicit pull-request plan.
Each PR is sized so that (a) a single reviewer can land it in one sitting, (b)
it doesn't break `main` on its own, (c) its acceptance criteria can be checked
mechanically. PRs are grouped into **waves** by dependency; waves are
independent of each other and can run in parallel.

## Conventions

- **Size.** S = < 200 LOC, M = 200–800 LOC, L = 800–2k LOC. Anything above L
  should be split.
- **Branch naming.** `claude/pr-<NN>-<slug>`.
- **Test plan.** Every PR lists the exact tests to run. `pytest -q` (fast)
  must stay green; `pytest --run-slow` adds the integration tier when a PR
  touches trained artifacts.
- **Definition of Done.** ruff, mypy, pytest all green; the README's
  **Verified vs aspirational** table is updated if the PR changes the status
  of an artifact.

---

## Wave 0 — Infrastructure unblockers (land first)

These unblock everything that depends on running real models in CI or on
committed checkpoints. Keep them small and self-contained.

### PR 1 — CI matrix: `[dl,ops]` job + `--run-slow` tier (**S**)

- **Why.** Today CI runs only on the base install. The slow integration
  tests never exercise; torch-gated tests never exercise; metric claims
  drift silently.
- **Scope.**
  - Extend `.github/workflows/ci.yml`: add a second job `test-dl` that
    installs `-e ".[dev,dl,ops]"` and runs `pytest --run-slow -q -m "not gpu"`.
  - Add a matrix dimension `python-version: ["3.11", "3.12"]`.
  - Pin torch to CPU wheels so the job stays under the 6-min GH-Actions
    free-tier envelope.
- **Out of scope.** GPU runners, MLX/MPS (no Apple-Silicon runners on the
  free tier).
- **Tests.** Existing suite. New job must show ~114 passed / 33 skipped.
- **Acceptance.** CI shield in the README turns green on `test-dl`.
- **Deps.** None.

### PR 2 — Coverage reporting + badge (**S**)

- **Why.** `pytest-cov` is installed but unused. No signal on which code
  paths are tested.
- **Scope.** `pytest --cov=src/mlcourse --cov-report=xml` in CI; upload to
  codecov (or keep artifact only, codecov optional); add badge to
  `README.md`.
- **Out of scope.** Chasing a specific coverage target.
- **Tests.** Existing + the coverage report itself.
- **Acceptance.** `coverage.xml` appears as a CI artifact.
- **Deps.** PR 1.

### PR 3 — Dev Docker image (**M**)

- **Why.** Install friction for a learner setting up on a fresh M-series
  Mac is non-trivial. Ship a Dockerfile with `[dl,llm,diffusion,rl,sciml,ops]`
  pre-installed; document the `--platform=linux/arm64` flag for MPS-less
  docker use.
- **Scope.** `docker/Dockerfile.dev` + `docker-compose.dev.yml` + README
  section + `make docker-dev`.
- **Out of scope.** A shipping image with trained checkpoints baked in.
- **Tests.** `docker build . -f docker/Dockerfile.dev` succeeds; `docker run
  … pytest -q` green.
- **Acceptance.** Someone with docker and zero Python can `make docker-dev &&
  make test`.
- **Deps.** None (parallel to PR 1).

---

## Wave 1 — Wire the harness (structural correctness)

The "reusable harness" claim from W6 needs to be actually reusable. Three
short structural PRs.

### PR 4 — Refactor W10 `train.py` to use `mlcourse.Trainer` (**S**)

- **Why.** W10 still writes its own training loop. W7 already uses the
  trainer; follow the same pattern.
- **Scope.** `portfolio/10_ddpm/train.py` calls `Trainer.fit` with a custom
  `loss_fn` that wraps `ddpm_loss`. Keep the post-training sample-grid code
  outside the trainer.
- **Out of scope.** Changing the UNet or sampler.
- **Tests.** New slow-marked test that trains for 1 epoch on a 500-sample
  FashionMNIST subset and asserts the loss drops.
- **Acceptance.** Search `portfolio/10_ddpm/` for `for epoch in range` →
  no hits outside the post-training sample code.
- **Deps.** None.

### PR 5 — Refactor W11 `train_ppo.py` to use `mlcourse.Trainer` (**S**)

- **Why.** Symmetric to PR 4.
- **Scope.** Trickier — PPO has a rollout+update loop that isn't a plain
  `fit`. Two options:
  1. Leave `ppo.train` as-is and document that RL is a deliberate
     exception (small doc-only change).
  2. Add a `Trainer.rollout_and_update` method so `ppo.train` is a thin
     shim.
  **Pick option 2 iff two or more downstream RL weeks would benefit;
  otherwise go with option 1.**
- **Out of scope.** Anything that adds complexity to the base Trainer.
- **Tests.** Existing W11 slow tests still pass.
- **Acceptance.** Clear doc comment in `ppo.py` explains the choice either
  way.
- **Deps.** None.

### PR 6 — Full Hydra refactor of every training script (**M**)

- **Why.** `src/mlcourse/configs/trainer.yaml` is a scaffold nobody
  consumes. Fix.
- **Scope.** Every `demo.py` / `train.py` under `portfolio/` that uses
  `Trainer` is converted to `@hydra.main(config_path="../../src/mlcourse/configs",
  config_name="...")`. Add a `src/mlcourse/configs/<weekNN>/<name>.yaml` per
  artifact.
- **Out of scope.** Exhaustive sweep configurations; leave overrides to
  command-line.
- **Tests.** Existing demos should accept `+trainer.max_epochs=1` via
  Hydra CLI.
- **Acceptance.** Each artifact README shows a Hydra command that resolves.
- **Deps.** PR 4, PR 5.

---

## Wave 2 — Verify compute claims & commit reference checkpoints

The **Verified vs aspirational** table in `README.md` has ⏳ for every
torch-dependent artifact. One PR per artifact; flip ⏳ → ✅ in the same PR.

### PR 7 — Verify W7 CIFAR-10, commit checkpoint (**M**)

- **Why.** "ResNet-18 reaches 90% in 30 min on MPS" is the current claim; no
  one has checked.
- **Scope.**
  - Run `portfolio/07_vision_classifier/demo.py` on MPS. Record wall-clock.
  - Commit `portfolio/07_vision_classifier/checkpoint.pt` (≤ 50 MB,
    gitignored for LFS if it's larger).
  - Update the README table row to ✅ with actual numbers.
  - Add a verified-metrics block to `portfolio/07_vision_classifier/verified.md`.
- **Out of scope.** Improving accuracy.
- **Tests.** New slow test loads the checkpoint and asserts test accuracy
  within ±1% of the committed number.
- **Deps.** PR 1 (for CI to run the slow test).

### PRs 8–12 — Symmetric per-artifact verifications

- **PR 8** W8 tiny GPT on TinyStories (**M**): commit 10 M-param checkpoint
  + 5 generated samples + verified-val-loss number. Runtime cap ~6 h MPS.
- **PR 9** W9 DPO TinyLlama (**M**): commit LoRA adapter + model-card +
  pairwise-win-rate JSON. Needs HF token in CI secrets.
- **PR 10** W10 DDPM (**S**): commit ε-model checkpoint + an
  `ablation_samples.png` grid.
- **PR 11** W11 PPO (**S**): commit PPO checkpoint on SimpleMarketMakerEnv
  + a learning curve PNG; also the torch-free agent eval artifact.
- **PR 12** W12 PINN (**S**): commit PINN checkpoint + the `pinn_vs_exact.png`
  panel + an $L^2$ error number.
- **Deps (all).** PRs 4–6 for the underlying refactor; PR 7 for the pattern.

---

## Wave 3 — Correctness additions beyond the surface

### PR 13 — Full FID via InceptionV3 in W10 (**M**)

- **Why.** Current `pixel-stat distance` proxy is honest but weak.
- **Scope.** Add `portfolio/10_ddpm/fid.py` using `torchmetrics.image.fid`
  or a minimal from-scratch InceptionV3 feature extractor. Switch
  `ablate.py` to report FID as the primary metric, with pixel-stat as a
  secondary cross-check. Note the ImageNet distribution-mismatch caveat
  for FashionMNIST in the README.
- **Out of scope.** Retraining the DDPM.
- **Tests.** Torch-gated FID shape test + a sanity test that identical
  distributions score FID ≈ 0.
- **Deps.** PR 10 (needs a committed checkpoint).

### PR 14 — MLX-native DPO path for W9 (**M**)

- **Why.** README cites `mlx-lm` but ships no code. Apple-Silicon is the
  target platform; this is the fastest actual path.
- **Scope.** `portfolio/09_dpo_tinyllama/mlx_train.sh` with the real
  `mlx_lm.lora` invocations; `mlx_convert.py` that converts TRL LoRA
  adapters to MLX-compatible format and back; README diff explaining when
  to prefer MLX vs TRL.
- **Out of scope.** Full MLX re-implementation of DPO from scratch.
- **Tests.** Unit test on the adapter conversion on mock weights (skipped
  unless `mlx` installed).
- **Deps.** PR 9 (HF-path baseline exists).

### PR 15 — Gradio Space for W9 (**S**)

- **Why.** Portfolio promises a live demo; no runnable code today.
- **Scope.** `portfolio/09_dpo_tinyllama/gradio_app.py` loads the DPO
  adapter and serves a chat UI; README has HF-Spaces deploy commands.
- **Out of scope.** CI deployment on every merge.
- **Tests.** Import smoke test (no launch).
- **Deps.** PR 9.

### PR 16 — Paper-reproduction code for W12 (**M**)

- **Why.** `portfolio/12_capstone/paper_reproduction/` ships only `PLAN.md`.
- **Scope.** Pick LoRA (Hu 2021). Reproduce Figure 2 — adapter-rank ablation
  on GLUE SST-2 — at tiny scale (TinyLlama base + 500-sample SST-2
  subset). Ship `run_ablation.py` + `figure_2.png` + a `findings.md` with
  the table + an extra configuration beyond the paper's.
- **Out of scope.** ImageNet-scale experiments.
- **Tests.** Smoke on 10 samples, 1 epoch.
- **Deps.** PR 9 (same HF ecosystem).

---

## Wave 4 — Pedagogy depth (per-week polish)

The review's biggest finding was lecture-notes density. Break by week. Each
PR ships: (a) time budget + self-assessment rubric + physics-bridge
callout (template already in W1/W5/W10), (b) expanded theory solutions,
(c) at least one long-form problem (6–8 hr) in `problems/long_form.md`.

- **PR 17** — W2 stat-learning polish (**M**). VC-dimension + Rademacher
  worked examples; concentration-inequality long-form problem (derive
  Hoeffding, apply to a specific CV bound).
- **PR 18** — W3 classical supervised (**M**). Kernel-trick from Mercer;
  long-form problem: implement XGBoost's exact 2nd-order split-gain and
  reproduce it on Adult.
- **PR 19** — W4 unsupervised (**M**). Avellaneda–Lee derivation fleshed
  out; long-form problem: OU half-life estimation on Ken French data.
- **PR 20** — W6 PyTorch / reproducibility (**M**). Expand the thin theory
  solution; long-form problem: full reproducibility audit of a given
  training run (seed sweep, stat test).
- **PR 21** — W7 CNNs (**M**). ConvNeXt vs ResNet comparison; long-form
  problem: from-scratch ViT on CIFAR-10.
- **PR 22** — W8 transformers (**M**). FlashAttention + KV-cache mechanics;
  long-form problem: implement KV-caching end-to-end in the tiny GPT.
- **PR 23** — W9 LLMs (**M**). ORPO + KTO comparison to DPO; long-form
  problem: build an MT-Bench-like judge and measure flip rate.
- **PR 24** — W11 RL (**M**). SAC / DQN contrast; long-form problem:
  implement one of them on a new env.
- **PR 25** — W12 applied (**M**). Neural-ODE adjoint-method derivation
  fleshed out; long-form problem: custom PINN on a PDE of the learner's
  choice.
- **PR 26** — W13 dev-surface (**S**). Claude-Code-refactor exercise
  walkthrough + a real LLM-judge reliability measurement on 50 prompts.
- **Deps (all).** None within the wave (per-week independent).

---

## Wave 5 — New topic modules

Each of these is a coherent new module / week addendum. Ordered by
leverage.

- **PR 27** — Quantisation mini-module (**M**). int8 + GPTQ + MLX 4-bit.
  A hands-on notebook measuring accuracy degradation on TinyLlama at
  different bit depths. Lives under `modules/14_quantisation/`.
- **PR 28** — Distributed-training mental-model (**S**). No GPU, no code.
  A single long markdown under `modules/15_distributed/` covering FSDP,
  ZeRO, tensor / pipeline / sequence parallelism with concrete memory
  math. Problem set = paper questions.
- **PR 29** — Interpretability deep-dive (**M**). Under
  `modules/16_interpretability/` with: probing classifiers (linear probe
  on GPT-2 layers), attention rollout on the W7 ViT preview,
  attribution-based SHAP on the W3 XGBoost, and a small sparse-autoencoder
  demo on TinyLlama activations.
- **PR 30** — GNN mini-module (**M**). `modules/17_gnn/`. Message-passing
  first principles; `GCN` + `GAT` implementations; applied to a quantum-
  physics benchmark (QM9 subset) and a transaction-graph toy. ~300 LOC.
- **PR 31** — Causal inference primer (**S**). `modules/18_causal/`.
  do-calculus, Pearl's hierarchy, confounding, instrumental variables —
  paper-style notes; one applied exercise using Fama-French residuals.
- **PR 32** — Time-series deep-learning addendum (**M**). Extend
  `modules/12_applied_capstone/`: add TFT + PatchTST as deep baselines
  with a reproducible fit on simulated returns.
- **PR 33** — Safety / red-teaming lab (**S**). `modules/19_safety/` with
  a hands-on harness: generate jailbreak prompts against a small local
  model, measure refusal rate; discuss constitutional AI, PI-classifiers.
- **PR 34** — Speculative decoding / MoE / KV-caching notes (**S**). A
  single extension document in `modules/08_transformers/notebooks/advanced.md`.
  Doc-only; no new code (implementation deferred unless there's demand).
- **Deps.** PR 9 for anything touching TinyLlama (27, 29, 33).

---

## Wave 6 — Docs & polish

- **PR 35** — mkdocs rendering (**M**). `mkdocs.yml` + `docs/` layout;
  integrate lecture notes with cross-refs, math rendering via
  `mkdocs-material` + `pymdownx.arithmatex` (KaTeX).
- **PR 36** — Per-week notebooks (**L, optional**). Convert the
  markdown lecture notes to Jupyter notebooks with interactive plots for
  W1 (Langevin), W4 (PCA), W5 (loss-landscape), W10 (diffusion steps).
  Keep markdown as source of truth; notebooks are generated via
  `jupytext` to avoid drift.
- **PR 37** — Dataset + model cards (**S**). Fill the W9 model card
  template with real numbers; add dataset cards for the W3 Adult run and
  the W4 Ken French dataset.
- **PR 38** — Per-week `CHANGELOG.md` (**S**). Trivial but useful for a
  recruiter reading the repo — shows when each artifact shipped.
- **Deps.** PRs 7–12 (for real metric numbers in cards).

---

## Dependency graph (summary)

```
Wave 0 (infra) ─────────────┐
                            ↓
                     Wave 1 (harness) ──────────────────────┐
                            │                               ↓
                            ↓                        Wave 2 (verify)
                     Wave 3 (correctness)                    │
                            │                               ↓
                            ↓                        Wave 4 (pedagogy polish)
                     Wave 5 (new topics)
                            │
                            ↓
                     Wave 6 (docs)
```

Waves 4 and 5 are independent of each other; within each wave, PRs are
independent.

## Suggested landing order

Minimum viable "course is honest" sequence — 10 PRs, roughly 3–4 weeks of
focused work:

1. PR 1 (CI matrix).
2. PR 4 (W10 uses Trainer).
3. PR 5 (W11 uses Trainer or doc-exception).
4. PR 7 (W7 verified + checkpoint).
5. PR 10 (W10 verified + checkpoint).
6. PR 11 (W11 PPO verified).
7. PR 12 (W12 PINN verified).
8. PR 14 (real MLX-DPO path).
9. PR 15 (Gradio Space).
10. PR 17 (W2 pedagogy polish, as a template for the rest of Wave 4).

After those ten, every weekly artifact has a verified number and a working
reusable-harness. The rest of the backlog becomes optional polish.

## PR size distribution

S = 17 · M = 18 · L = 1. Weighted estimate ~ 25–35 engineer-days of focused
work — realistic for one contributor over 5–7 weeks, or a small team over
a sprint.

## When to close `TODO.md`

Mark `TODO.md` as **closed** once:

- All Wave 0–3 PRs have landed (infra + harness + verify + correctness).
- Wave 4 is complete (or explicitly deferred with updated pedagogical
  expectations in `SYLLABUS.md`).
- Wave 5 has a decision (either shipped or explicitly punted).

At that point, the course is a real 20-hr/week intensive — not the strong
scaffold it is today.
