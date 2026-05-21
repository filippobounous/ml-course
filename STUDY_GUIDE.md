# Study guide

How to approach this course as a learner. **Read this once, then start
W1.** You don't need the entire backlog to be done before beginning — the
course is usable end-to-end today.

---

## What you do (the curriculum) vs what I'm doing (the PR plan)

Two parallel tracks run on this repo:

- **You** work through the **weeks** (W1 → W13). Each week ships an
  artifact. The artifacts together are your portfolio.
- **I** work through the **waves** in
  [`PR_PLAN.md`](PR_PLAN.md) — a backlog of ~38 PRs polishing the
  course. Most waves *enhance* the curriculum; none of them *block* it.

These are orthogonal. Your job is the weeks. The waves are mine.

---

## The weekly rhythm

Each week follows the same shape (see `modules/NN_*/`):

1. **Readings + lecture notes** (`modules/NN_*/notebooks/lecture_notes.md`
   + `modules/NN_*/readings.md`) — ~4 hr.
2. **Problem set** (`modules/NN_*/problems/README.md`) — 2 theory +
   2 implementation + 1–2 applied — ~8 hr. Solutions are in
   `solutions.py` (code) and `solutions_theory.md` (proofs).
3. **Portfolio artifact** (`portfolio/NN_*/`) — the shareable piece —
   ~6 hr. Each artifact has a `README.md` (what it is) and a `demo.py`
   (one command to run it end-to-end).
4. **Reflection** — fill in the "What I learned" bullets at the bottom
   of the artifact's `README.md`. Hiring managers skim those.

Target: **~20 hr/week × 13 weeks**. CPU or Apple Silicon (MPS) is enough.
No CUDA GPU required. Each demo fits in under an hour on an M-series Mac.

---

## Suggested order

**Sequential, W1 → W13.** Each week assumes the previous one:

- **W1–W4** are pure NumPy / sklearn (math + classical ML).
- **W5** hands you an autograd engine (`mlcourse.autograd`) you'll reuse
  understanding W6's `Trainer`.
- **W6** ships the harness; **W7, W10** consume it via `Trainer.fit`.
- **W8** (transformers from scratch) is the **capstone kick-off** —
  start the multi-week tiny-GPT project here.
- **W9–W12** are applied tracks (LLMs, diffusion, RL, capstone).
- **W13** is the meta module: LLMs as a development surface.

**Cherry-picking** is fine if you already know an area. Skip W2 if
you've done linear models; jump to W8 directly if transformers is your
goal. Each week's `tests/week_NN/` gives you a confidence check without
doing the full week.

---

## How to check yourself

Three feedback loops, in increasing order of strength:

1. `pytest tests/week_NN/` should pass after you finish the problem set.
2. `pytest --run-slow tests/week_NN/` runs the integration tier (actually
   trains a small model) for weeks that have one — best smoke check that
   your environment is healthy.
3. The portfolio artifact's `demo.py` should run end-to-end. The numbers
   you produce go in the **"Verified vs Aspirational" table** at the top
   of [`README.md`](README.md) — flip ⏳ → ✅ when yours match by
   filing a PR with a `portfolio/<artifact>/verified.md` log.

---

## Waves vs weeks — the mapping

Most waves touch multiple weeks; one wave adds entirely new weeks. None
block the learner (you can start W1 today on the current integration
branch). The waves that *most* affect your experience are bolded.

| Wave | What it ships | Weeks affected | Blocks the learner? |
|---|---|---|---|
| 0 (PR 1–3) ✅ done | CI matrix, coverage, Docker | All — content unchanged | No |
| 1 (PR 4–6) ✅ mostly | Trainer integration, Hydra | W6, W7, W10, W11 | No |
| 2 (PR 7–12) | Verify compute claims + commit reference checkpoints | One PR each for W7, W8, W9, W10, W11, W12 | No — your own runs flip ⏳ → ✅ |
| 3 (PR 13–16) | FID, MLX, Gradio Space, paper reproduction | PR 13 → W10; PR 14, 15 → W9; PR 16 → W12 | No — nice-to-have features |
| **4 (PR 17–26)** | **Theory solutions + time budget + rubric + physics bridge** | One PR per week, **W1–W12** | **Partially** — without `solutions_theory.md`, theory problems are hard to self-check |
| 5 (PR 27–34) | New topic modules (quantisation, distributed, GNN, causal, time-series, safety, speculative, interpretability) | None of W1–W13 — these are *new* weeks (W14+) | No |
| 6 (PR 35–38) | mkdocs site, notebook gallery, model cards, changelog | All — cross-cutting docs | No |

The **bolded Wave 4** is the one with direct day-to-day impact on you.
The shipping order will be `W1 → W12` so theory solutions stay one or
two weeks ahead of where you are.

---

## Recommended operating split

- **You**: work the curriculum starting at W1, generating Wave-2
  verification numbers as you go.
- **Me**: ship Wave 4 (pedagogy polish) in week-order so theory
  solutions stay one or two weeks ahead. Pivot to Wave 3 features
  (FID, MLX, Gradio) when you're nearing W9 / W10 / W12.

---

## Practical operating manual

- **Install per week**: `make week-5`, `make week-6`, … rather than
  `pip install .[all]` upfront — saves you fighting wheels you don't
  need yet. See [`README.md`](README.md) §Setup.
- **Don't skip the "What I learned" bullets** in each artifact's
  `README.md`. They're what makes the portfolio shareable; tests verify
  the code; the bullets verify *you*.
- **The honesty table is honest**: torch-dependent metrics in
  [`README.md`](README.md) are ⏳ until someone runs them. Your runs
  are how those become ✅.
- **The course is a strong skeleton, not a finished intensive.** Pair
  lecture notes with a textbook (Goodfellow, Murphy, Sutton-Barto, …);
  the per-week `readings.md` lists the canonical ones.

---

## If you want a sampler before committing 13 weeks

Do **W2 + W4 + W5** as a three-weekend sampler:

- W2: NumPy linear regression (matches sklearn to 1e-9).
- W4: PCA stat-arb notebook on simulated cross-sectional returns.
- W5: micrograd-style scalar autograd + an MLP that solves two-moons.

All three are pure-NumPy / sklearn / Python on CPU, sub-1-min each.
They tell you whether the course's pacing and depth match your taste
before you commit to W6 onwards (where torch and longer runtimes
come in).
