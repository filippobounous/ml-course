# Capstone (working directory)

Kicks off in **Week 8**, develops in parallel with Weeks 9–12, ships in **Week 12**.

This directory holds the in-progress capstone. The *final* artifact is moved to
`portfolio/12_capstone/` when Week 12 completes.

## Week 8 deliverables

- `proposal.md` — one-page proposal (goal, success metric, dataset, compute budget, timeline, risks).

## Weeks 9–11 deliverables

- Weekly 30-min sync notes in `progress/`.
- A minimal end-to-end pipeline that runs on toy data before chasing the full dataset.

## Week 12 deliverable

- Move the final artifact to `portfolio/12_capstone/` and produce the bonus paper-reproduction piece in `portfolio/12_capstone/paper_reproduction/`.

## Grading rubric

Aim for **≥ 80%** before calling the capstone portfolio-ready (same bar as the weekly
problem sets). Score yourself honestly:

| Dimension | Weight | Full marks |
|---|---|---|
| Proposal & scoping | 15% | One-page `proposal.md` with a single primary metric, a trivial **and** a strong baseline, a compute cap, and concrete risks |
| Correctness | 25% | Sound method; results checked against a baseline or analytical reference; no look-ahead leakage (Track B) / boundary + initial conditions satisfied (Track A) |
| Reproducibility | 20% | One-command repro (`demo.py` / `make reproduce`), seeded, Hydra config + committed run log, model **and** dataset card |
| Results & analysis | 20% | Metric reported with units and a baseline comparison; ≥ 1 ablation; an honest "what broke / what I'd do with more compute" |
| Communication | 10% | README answers problem → method → results → reproduce → what-I-learned; figures legible and captioned |
| Paper-reproduction bonus | 10% | One figure reproduced at tiny scale with an extra configuration beyond the paper |

Two hard gates regardless of score:

- **Honesty.** Flip the artifact's row in the README "verified vs aspirational" table to ✅
  only once the metric is reproduced on real hardware with a committed log — never on the
  strength of the code alone.
- **Scope clarity.** Your README must state which pieces are your contribution vs the
  provided reference (see `modules/12_applied_capstone/README.md` → "What ships as reference
  vs what you build").
