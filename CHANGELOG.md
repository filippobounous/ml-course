# Changelog

All notable changes to this course are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
on the `mlcourse` Python package only — curriculum content is versioned
informally by milestone.

## [Unreleased]

### Added

- **PR 36** — portfolio gallery as an mkdocs landing page with
  grid-card layout of every W2–W13 artifact.
- **PR 37** — Mitchell-2019 model cards: template + seven
  per-artifact cards (W5, W7, W8, W9, W10, W11, W12) with explicit
  caveats and Verified ⏳/✅ markers.
- **PR 38** — this CHANGELOG.

### Open / not yet merged

- See `PR_PLAN.md` and the [pull-requests page](https://github.com/filippobounous/ml-course/pulls).

---

## [0.1.0] — 2026-05-21

The first integrated release: the 12-week ML/AI course is end-to-end
walkable, with infrastructure, Trainer integration, two of four
Wave-3 features, full per-week pedagogy polish, and the mkdocs site
deployed to GitHub Pages.

### Course content

- **Phases A–E** — initial scaffold (Phase A), W1–W6 content
  (Phase B), W7–W10 content (Phase C), W11–W12 content (Phase D),
  review-driven hardening (Phase E): Trainer integration, CFG, PPO
  37-details subset, W13 (LLMs as dev surface), theory solutions.

### Wave 0 — Infrastructure

- **PR 1 (#7)** — CI matrix on Py 3.11/3.12 + `test-dl` slow
  integration tier behind `--run-slow`.
- **PR 2 (#8)** — `pytest-cov` + coverage XML artifact + README
  badges.
- **PR 3 (#9)** — dev Docker image with CPU torch + every extra
  preinstalled; `make docker-dev{,-shell,-test}` ergonomic targets.

### Wave 1 — Trainer integration

- **PR 4 (#10)** — `Trainer.fit(loss_fn=None, …)` custom-loss path
  for DDPM (and any objective that doesn't fit the standard
  `(x, y) → loss` shape).
- **PR 5 (#11)** — W11 PPO documented as a deliberate Trainer
  exception (rollout → GAE → K-epoch minibatch update doesn't fit
  a DataLoader).
- **PR 6 (#14)** — Hydra refactor of W6 / W7 / W10 demos:
  `@hydra.main` entry points, `trainer/default.yaml` as a group
  default, per-week configs in `src/mlcourse/configs/weekNN/`.

### Wave 3 — High-leverage features (2 of 4)

- **PR 13 (#28)** — full FID via InceptionV3 in W10
  (`portfolio/10_ddpm/fid.py`): Heusel-2017 formula with
  `scipy.linalg.sqrtm`, FashionMNIST-to-Inception adapter,
  pluggable feature extractor for tests.
- **PR 16 (#29)** — paper-reproduction code at W12:
  Schulman 2017 Figure 6 (PPO clip-ε ablation) on CartPole-v1 via
  the W11 PPO end-to-end.
- **PR 14, 15** — MLX-native DPO (Mac-only) and Gradio Space
  (HF token) deferred.

### Wave 4 — Per-week pedagogy polish (W1 → W12)

Every week W1–W12 received the same polish block:

- **Worked numerical examples** (one per week, paper-doable in
  10–15 min).
- **Time budget** breakdown of the ≈ 20 hr/week target.
- **Self-assessment rubric** — 5 yes/no questions to ask before
  moving to the next week.
- **Physics-bridge callouts** — 60+ connections threading the
  course (W2: ERM ↔ zero-temperature Gibbs; W7: ResNet skip ≡
  identity-channel propagator; W11: PPO clip ↔ TRPO trust region;
  W12: PINN ↔ weak PDE formulation; etc.).

### Wave 6 — Docs (1 of 4)

- **PR 35 (#30)** — mkdocs-material site auto-deployed to GitHub
  Pages on every push to `main`. Single source of truth: markdown
  lives under `modules/`, `portfolio/`, repo root; staged into
  gitignored `docs/` by `scripts/build_docs.py`.
- **STUDY_GUIDE.md** — top-level "how to work through this course"
  doc: weekly rhythm, suggested order, self-check feedback loops,
  wave-vs-week mapping table, recommended operating split.

### Honesty markers

- Top-level **verified-vs-aspirational** table in `README.md` —
  every torch-dependent metric carries an explicit ⏳ until a
  real-hardware run flips it to ✅.

### Stats at release

- **13 modules** (W1 → W13).
- **12 portfolio artifacts** (W2 → W13).
- **143+ tests** passing in the base suite.
- **30+ sub-PRs** rolled up via the umbrella PR (#31).

---

## Versioning

The Python package `mlcourse` ships an `0.1.0` version pinned in
`pyproject.toml`. Future bumps follow Semantic Versioning **on the
Python API only**:

- **Patch** — bug fixes, documentation typos, single-file pedagogy
  fills.
- **Minor** — new modules, new portfolio artifacts, new package APIs.
- **Major** — breaking changes to `mlcourse.Trainer`, `mlcourse.autograd`,
  or removal of a portfolio artifact.

The curriculum itself is versioned by milestone (Wave 0, Wave 1, …),
mapped to the [`PR_PLAN.md`](PR_PLAN.md) backlog.
