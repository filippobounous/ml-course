# Course review — structure, content, consistency, gaps

**Date:** 2026-06-13
**Scope:** All 13 modules, 12 portfolio artifacts, governing docs (README, SYLLABUS,
STUDY_GUIDE, PORTFOLIO, TODO, CHANGELOG, gallery, mkdocs), tests, and `src/mlcourse`.
**Method:** 20-agent review — one deep reader per module (read docs + matching artifact
+ tests), independent skeptic re-checks of every high-severity accuracy claim, three
cross-cutting analysts (structure / consistency / gaps), and a synthesis pass.
**Tally:** 81 raw findings → 26 deduped & ranked. 3 high-severity accuracy claims
independently verified, 0 refuted.

---

## Verdict

A genuinely strong, mathematically rigorous **skeleton** of a graduate-level ML/AI
intensive — but not yet the finished 20-hr/week course it advertises. Two systemic
themes dominate, plus a handful of concrete bugs that reach the learner directly.

1. **Promise/deliver gap (the biggest theme).** Topics are taught — often with
   derivations, worked examples, and model-card claims — but absent from the runnable
   artifact (recurs across ≥8 modules).
2. **Verifiability gap.** Most second-half artifacts are code-complete but never run, so
   "What I learned" reflections, checkpoints, training curves, and aspirational metrics
   are unfilled or unverified.
3. **Framing inconsistencies.** "12 weeks" vs 13 modules; an advertised 30/40/30
   theory/code/applied split that is actually ~46/31/24.

The README honesty table and verified-artifact discipline are a real strength — the
recommendation is to *extend* that discipline to cover promise/deliver gaps, not just
metric verification.

## Strengths (do not lose these)

- Mathematically rigorous, correct lecture content across all 13 modules (Bellman
  contraction, policy gradient, PPO/GAE, DDPM ELBO, RoPE invariance, backprop, convex
  optimization all spot-checked).
- Consistent, high-quality pedagogical scaffold: every module except W13 has lecture
  notes, worked examples, theory solutions, a time budget, a self-assessment rubric, and
  a "physics bridge" tailored to quantitatively-trained learners.
- Sound prerequisite flow: W1–W4 foundations → W5 autograd → W6 Trainer → W7/W10
  consumers → W8 transformers/capstone kickoff → W9–W12 applied.
- Genuinely verified artifacts where claimed: W2 NumPy linreg matches sklearn to ~1e-15
  (8 tests), W4 stat-arb commits real `results.md` (IS Sharpe 3.21 / OOS 2.93), W5
  micrograd has a slow test asserting ≥0.88 on two-moons.
- An honesty table that distinguishes verified outputs from aspirational targets — a
  rare, trust-building discipline.
- Clean, readable artifact code; docs build infrastructure (build_docs.py, mkdocs nav,
  symlink mapping) is internally consistent with no broken links.

---

## Fixes applied — pass 1 (correctness bugs)

| # | Fix | File(s) |
|---|-----|---------|
| 1 | W5 Glorot init corrected to `6/(n_in + n_out)` (was `6/(n_in + 1)`) | `src/mlcourse/autograd/nn.py`; slow-test budget 40→60 epochs in `tests/week_05/test_engine_slow.py` (threshold unchanged at 0.88) |
| 2 | W2 optimal-λ corrected `0.9 → σ² = 0.09`, table & prose recomputed | `modules/02_stat_learning/notebooks/worked_examples.md` |
| 12 | W8 model card: tokenizer **is** byte-level and round-trips (was "not byte-level") | `portfolio/08_tinygpt/model_card.md` |
| 18 | W2 quickstart import `from portfolio.linreg` → `from linreg` (run from dir) | `portfolio/02_numpy_linreg/README.md` |

> The Glorot fix exposed a slow test that was implicitly calibrated to the buggy init
> (`seed=0` was an unlucky 0.855 outlier; 9/10 other seeds passed). The fix is correct;
> the test was given more training budget rather than a lower bar. Consider adding the
> variance-preservation regression test recommended under finding #1.

## Fixes applied — pass 2 (Direction A: honesty & correctness sweep)

| # | Change | File(s) |
|---|--------|---------|
| 7 | W6 `Trainer.fit()` problem spec now matches the implementation (`val_loader=None, *, loss_fn, optimizer`) | `modules/06_pytorch_trainer/problems/README.md` |
| 20 | W3 aligned to code: report covers all benchmarked methods (incl. sklearn GBDT), dataset is **Adult** (not Covertype), grading claim corrected (IRLS + info-gain; benchmark emits markdown+PNG, not JSON, and is not auto-graded) | `modules/03_classical_supervised/problems/README.md`, `.../notebooks/lecture_notes.md` |
| 26 | `build_docs.py` W13 comment corrected (intentional exclusion of the optional week, not "missing") | `scripts/build_docs.py` |
| 11 | W12 stat-arb de-promised to **embargo-only**; label-horizon purging marked as an extension (artifact + problem #2b) | `portfolio/12_capstone/README.md`, `modules/12_applied_capstone/problems/README.md` |
| 5 | Framing standardized to **"12-week core (W1–W12) + optional Week 13"**; 30/40/30 balance now caveated (early weeks run theory-heavier) | `README.md`, `PORTFOLIO.md`, `STUDY_GUIDE.md`, `SYLLABUS.md` |
| 3, 4, 9, 10 | Honesty labels for taught-but-not-implemented: W8 model card now says **learned positional embeddings** (not RoPE) and **constant LR** (not warmup/cosine); W9 Gradio Space + MLX path marked **optional** and the dead "Gradio module imports" grading criterion removed; W5 BatchNorm note on problem #5; W7 note that `demo.py` does not run the transfer baseline; new **"Taught vs implemented"** note in the README honesty table | `portfolio/08_tinygpt/model_card.md`, `portfolio/09_dpo_tinyllama/README.md`, `modules/09_llms_dpo/problems/README.md`, `portfolio/07_vision_classifier/README.md`, `modules/05_nn_from_scratch/problems/README.md`, `README.md` |

> Pass 2 are **de-promotions / honesty labels**, not implementations — the underlying
> features (W8 RoPE + LR schedule, W9 Gradio app + native-MLX DPO script, W12 purging,
> W5 BatchNorm) remain open as Direction B/C work. Verified: full fast suite **140 passed**
> after the sweep (the 5 W7/W12 failures are pre-existing — missing torch/torchvision and a
> numpy-version issue in this environment — not introduced by these edits).

---

## Finding status (audited 2026-06-16)

Status of all 26 findings against the current repo, re-verified by reading the code (not
`TODO.md`). Implementation status of the PR backlog lives in `PR_PLAN.md` §Implementation
status.

| # | Finding | Status |
|---|---------|--------|
| 1 | W5 Glorot init | ✅ FIXED |
| 2 | W2 optimal-λ worked example | ✅ FIXED |
| 3 | Promise/deliver gap | ◑ PARTIAL — W8 RoPE/LR, W9 Gradio/MLX, W5 BatchNorm now labeled; W7 transfer baseline still not run; underlying features unimplemented |
| 4 | Second-half artifacts never executed | ☐ OPEN (Wave 2, PR 7–12) |
| 5 | 12-vs-13-week framing + balance claim | ✅ FIXED |
| 6 | W6 solutions cover only 2 of 6 | ☐ OPEN |
| 7 | W6 `Trainer.fit()` signature mismatch | ✅ FIXED |
| 8 | "Trainer never wired into W10/W11" | ✗ WITHDRAWN — false positive (W10 uses Trainer; W11 documented exception) |
| 9 | W9 Gradio app missing | ◑ PARTIAL — labeled optional; implementation = PR 15 |
| 10 | W9 MLX DPO missing | ◑ PARTIAL — labeled optional; implementation = PR 14 |
| 11 | W12 purging vs embargo | ✅ FIXED (de-promoted to embargo-only) |
| 12 | W8 model-card tokenizer claim | ✅ FIXED |
| 13 | W13 missing scaffold + safety topic | ◑ PARTIAL — has solutions_theory; still lacks worked_examples/rubric/physics-bridge; safety listed not covered |
| 14 | Capstone proposal template/rubric | ◑ PARTIAL — `capstone/proposal.md` template exists; rubric + core-vs-extension scope still missing |
| 15 | W2 MDP problem #7 no solution | ☐ OPEN |
| 16 | W11 agent harness 4 vs 20 tasks | ☐ OPEN |
| 17 | W10 CFG reading missing | ☐ OPEN |
| 18 | W2 quickstart import | ✅ FIXED |
| 19 | W8 attention-map viz code | ☐ OPEN |
| 20 | W3 spec/dataset/test drift | ✅ FIXED |
| 21 | W1 PL condition listed-not-covered | ☐ OPEN |
| 22 | W6 worked example uses Trainer early | ☐ OPEN |
| 23 | W4/W5 rubrics require untaught techniques | ☐ OPEN |
| 24 | Worked-example false precision | ◑ PARTIAL — W10 table uses ~ marks; minor residue |
| 25 | Cross-module forward/back refs | ☐ OPEN |
| 26 | `build_docs.py` stale W13 comment | ✅ FIXED |

**Tally:** 9 fixed · 1 withdrawn · 6 partial · 10 open.

> Finding #8 was a false positive inherited from stale `TODO.md` — the gaps analyst trusted
> the backlog instead of the code. `portfolio/10_ddpm/train.py` calls `trainer.fit(...)` and
> `portfolio/11_rl_agent/ppo.py` documents PPO as a deliberate Trainer exception (PR 4 / PR 5).
> Lesson: trust code over `TODO.md` for status.

---

## Prioritized findings

> Entries below are the original review text. See the status table above for current state —
> #8 in particular is **withdrawn**.

Severity: 🔴 high · 🟡 medium · 🟢 low. Dimension: structure / accuracy / consistency / gaps.

### 🔴 High

**1. [accuracy] W5 Glorot initialization formula is wrong in the artifact** — ✅ FIXED
`Neuron.__init__` used `a = (6.0 / (n_in + 1)) ** 0.5`; Glorot/Xavier requires fan-in
*and* fan-out: `(6 / (n_in + n_out)) ** 0.5`. The `n_in+1` proxy is only correct for a
single-output neuron; any layer with `n_out > 1` got the wrong init variance —
contradicting the variance-preservation theory the module teaches.
*Loc:* `src/mlcourse/autograd/nn.py:44`. *Rec:* (done) Neuron now accepts `n_out`; Layer
passes its real fan-out. Add a regression test asserting activation variance stays ~constant
across a deep net's layers under Glorot.

**2. [accuracy] W2 optimal-λ worked example off by ~10×** — ✅ FIXED
Claimed `λ* = 10σ²/(β*)² = 0.9` ("≈1"); differentiating `(λ²+10σ²)/(10+λ)²` gives
`λ* = σ² ≈ 0.09`. A learner trusting it drew the wrong bias-variance conclusion.
*Loc:* `modules/02_stat_learning/notebooks/worked_examples.md:43,46`. *Rec:* (done) formula,
table, and prose recomputed.

**3. [consistency] Systematic promise/deliver gap: lectures & model cards advertise components the artifacts never implement**
*The single biggest theme.* W8 RoPE is derived in depth yet `model.py` uses learned
positional embeddings; W8 warmup+cosine LR promised but `train.py` uses fixed LR; W7
transfer-learning baseline (`transfer_resnet18` exists but never called) and failure-mode
report absent from `demo.py`; W12 Neural ODEs/PySR/Track-B DL lectured but unimplemented;
W5 BatchNorm/LayerNorm taught and required by problem #5 but not in micrograd; W4 factor
models & power iteration referenced but never taught.
*Loc:* `portfolio/08_tinygpt/model.py` vs `model_card.md:10` + `lecture_notes.md:45,124`;
`portfolio/08_tinygpt/train.py:94`; `portfolio/07_vision_classifier/demo.py` vs
`classifier.py:85-97`; `modules/12_applied_capstone/notebooks/lecture_notes.md:33-43,68-74`;
`src/mlcourse/autograd/nn.py` vs `modules/05_nn_from_scratch/problems/README.md:19`;
`modules/04_classical_unsupervised/README.md:12`.
*Rec:* Adopt a per-module rule — a topic appears in headline promises/model-cards only if
the artifact implements it. Decide implement-or-depromise per item. Highest-value
implements: W8 RoPE + LR schedule, W5 BatchNorm, W7 transfer + failure-mode.

**4. [gaps] Second-half artifacts are code-complete but never executed**
W7–W12 ship no trained checkpoint, no committed training curves, and placeholder "What I
learned" sections. Model cards mark FID/win-rate/return/runtime aspirational. None of the
second-half claims are reproducible without the learner first burning hardware; W12's
`paper_reproduction/findings.md` is empty.
*Loc:* `portfolio/07_vision_classifier/model_card.md:35-41`; `08_tinygpt/README.md:55-59`;
`09_dpo_tinyllama/README.md:76-78`; `10_ddpm/README.md:96` + `model_card.md:45-47`;
`11_rl_agent/model_card.md:74-78`; `12_capstone/paper_reproduction/findings.md`.
*Rec:* One hardware pass per artifact (commit checkpoints or a download, learning-curve
PNGs, filled reference reflections), or a prominent "code-only scaffold; not hardware-verified" banner.

**5. [consistency] Framed as "12-week" but structurally 13 weeks; 30/40/30 balance is actually ~46/31/24**
README/PORTFOLIO say "12-week"; SYLLABUS says "12 + optional W13"; STUDY_GUIDE says "13
weeks". 13 module/test dirs, 12 portfolios. Summed time budgets ≈ 45.6% theory / 30.7%
code / 23.7% applied; W1–W4 average 54% theory, front-loading a theory gauntlet that
conflicts with the "recruiter-ready artifacts" positioning.
*Loc:* `README.md:6,14-15`; `SYLLABUS.md:1,7`; `STUDY_GUIDE.md:38`; `PORTFOLIO.md:3`.
*Rec:* Pick one framing (recommend "13-week: W1–W12 core + W13 optional"); either correct
the balance claim to ~45/31/24 or rebalance W1–W3 toward applied work.

**6. [gaps] W6 problem solutions cover only 2 of 6 problems**
`solutions_theory.md` (~331 words) covers #1–2; implementation #3–5 and applied #6 have no
solutions. `torch.profiler` is required by #6 but never taught.
*Loc:* `modules/06_pytorch_trainer/problems/solutions_theory.md`; `.../lecture_notes.md`;
`.../problems/README.md:21`. *Rec:* Add `solutions_implementation.md` + `solutions_applied.md`
and a short profiler subsection to the notes.

**7. [consistency] W6 `Trainer.fit()` signature in the problem statement ≠ implementation**
`problems/README.md:12` specifies `fit(model, train_loader, val_loader, *, config)` but
`trainer.py:66` uses `fit(..., *, loss_fn, optimizer)`. Coding to spec raises `TypeError`.
The lecture notes have the correct signature, so the problem statement is the outlier.
*Loc:* `modules/06_pytorch_trainer/problems/README.md:12`; `src/mlcourse/trainer.py:66-74`.
*Rec:* Fix the problem statement; add a formal API section to `portfolio/06_trainer/README.md`.

### 🟡 Medium

**8. [gaps] ~~`mlcourse.Trainer` never wired into W10/W11~~ — ✗ WITHDRAWN (false positive).**
Inherited from stale `TODO.md`, not verified against code. In fact W10
`portfolio/10_ddpm/train.py` calls `trainer.fit(..., loss_fn=None, optimizer=...)` (PR 4),
and W11 `portfolio/11_rl_agent/ppo.py` documents PPO as a *deliberate* Trainer exception
with rationale (PR 5). The payoff is demonstrated (W7, W10) and the one exception is
intentional and explained. No action needed.

**9. [gaps] W9 Gradio Space promised and graded against, but `gradio_app.py` doesn't exist.**
A grading criterion points at a nonexistent file.
*Loc:* `portfolio/09_dpo_tinyllama/README.md:73`; `modules/09_llms_dpo/problems/README.md:21`.

**10. [gaps] W9 MLX-native DPO "recommended" but only the TRL+MPS path is implemented.** No
`mlx_dpo_train.py`; the recommended path can't be reproduced.
*Loc:* `portfolio/09_dpo_tinyllama/dpo_train.py`; `README.md:4,24-37`.

**11. [consistency] W12 README claims "purging + embargo" but only embargo is implemented.**
`statarb_walkforward.py:78-87` applies embargo only — no label-horizon purge. Leakage
control is exactly the lesson; the worked examples/theory describe both, so code is the outlier.
*Loc:* `portfolio/12_capstone/README.md:31`; `statarb_walkforward.py:78-87`;
`modules/12_applied_capstone/problems/solutions.py:77-112`.

**12. [consistency] W8 model card falsely claimed the tokenizer is "not byte-level".** — ✅ FIXED
`train.py:30` uses HF `ByteLevel` with no normalization — it *is* byte-level and round-trips.
*Loc:* `portfolio/08_tinygpt/model_card.md:51-53`; `train.py:30`.

**13. [structure] W13 missing the standard scaffold and a listed topic.** No
`worked_examples.md`, rubric, or physics bridge (every other module has them); "safety" is
a README topic but absent from the notes. The artifact itself is good and unit-tested.
*Loc:* `modules/13_llms_dev_surface/notebooks/lecture_notes.md`; `README.md:22`.

**14. [structure] Capstone (W8 kickoff → W12 delivery) lacks a proposal template, rubric, and core-vs-extension scope.**
W8 requires a one-page proposal in `capstone/proposal.md` with no template/rubric; W12
Track A/B promise unimplemented components without marking reference vs student-built.
*Loc:* `SYLLABUS.md:20-27`; `modules/08_transformers/README.md:22`; `modules/12_applied_capstone/README.md:1-32`.

**15. [gaps] W2 applied MDP problem (#7) has no reference solution; grading path unclear.**
`solutions_theory.md` covers #1–4 only; `tests/week_02/test_linreg.py` covers only the
linreg artifact, so #7 can't be graded/self-checked.
*Loc:* `modules/02_stat_learning/problems/README.md:17,21`; `tests/week_02/test_linreg.py`.

**16. [gaps] W11 agent eval harness ships 4 tasks, not the promised 20.**
*Loc:* `portfolio/11_rl_agent/agent.py:242-248`; `modules/11_rl_agents/problems/README.md:17`.

**17. [gaps] W10 classifier-free guidance is taught/implemented/tested but Ho & Salimans (2022) is missing from readings.**
*Loc:* `modules/10_diffusion_multimodal/readings.md`.

**18. [consistency] W2 README quickstart import path raised `ModuleNotFoundError`.** — ✅ FIXED
`from portfolio.linreg import ...` — no such package; demo uses `from linreg import ...`.
*Loc:* `portfolio/02_numpy_linreg/README.md:25`.

**19. [gaps] W8 attention-map visualisation is a required deliverable with no code or example plots.**
The model exposes attention weights but no plotting utility/example plots exist.
*Loc:* `modules/08_transformers/problems/README.md:18`; `portfolio/08_tinygpt/`.

### 🟢 Low

**20. [consistency] W3 spec/artifact/test mismatches** — problem says "four methods" but
`benchmark.py` runs 5; notes say Covertype, code uses Adult; test claim says it verifies
benchmark JSON but tests cover IRLS/info-gain and output is markdown+PNG.
*Loc:* `modules/03_classical_supervised/problems/README.md:16,20`; `notebooks/lecture_notes.md:83`;
`portfolio/03_tabular_benchmark/benchmark.py:39,102-157`; `tests/week_03/test_problem_set.py`.

**21. [gaps] W1 lists Polyak-Łojasiewicz as a topic but never covers it.**
*Loc:* `modules/01_math_foundations/README.md:16`.

**22. [structure] W6 worked example uses the Trainer before students build it.**
*Loc:* `modules/06_pytorch_trainer/notebooks/worked_examples.md:87-113`.

**23. [gaps] W4/W5 self-assessment rubrics require untaught techniques** (W4 PCA via power
iteration; W5 empirical Glorot variance verification — no starter code).
*Loc:* `modules/04_classical_unsupervised/problems/README.md:12`; `modules/05_nn_from_scratch/notebooks/worked_examples.md:111`.

**24. [accuracy] Several worked-example tables present illustrative numbers as if exact**
(W10 noise-schedule table ~1–2% off with inconsistent `~`; W4 k-means distortion jump
without the intermediate value).
*Loc:* `modules/10_diffusion_multimodal/notebooks/worked_examples.md:15-20`;
`modules/04_classical_unsupervised/notebooks/worked_examples.md:71`.

**25. [structure] Cross-module forward/back references promised but not coordinated** — W1
SDE primer "for W10" and W2 MDP primer "for W11" are never called back from W10/W11/W12.
*Loc:* `modules/01_math_foundations/notebooks/lecture_notes.md:4-8`;
`modules/02_stat_learning/notebooks/lecture_notes.md:67-73`; `modules/08_transformers/notebooks/lecture_notes.md:138-149`.

**26. [structure] `build_docs.py` comment wrongly says W13 solutions_theory is missing** —
the file exists with 3 complete solutions.
*Loc:* `scripts/build_docs.py:74`; `modules/13_llms_dev_surface/problems/solutions_theory.md`.

---

## Open questions (author decisions)

1. **12 weeks or 13?** Pick one framing and propagate across README/SYLLABUS/STUDY_GUIDE/
   PORTFOLIO. If W13 is truly optional, decide whether its module/tests live in the core
   tree or a "stretch" section.
2. **For each promise/deliver gap (W8 RoPE + LR, W5 BatchNorm, W7 transfer/failure-mode,
   W9 Gradio/MLX, W12 Neural ODE/PySR/Track-B DL): implement or de-promise?** The central
   editorial decision; make it explicitly per item.
3. **Hardware-run reference artifacts before release, or ship as explicitly-labeled
   code-only scaffolds?** Determines whether checkpoints/curves/reflections get committed.
4. **Correct the 30/40/30 balance claim to ~45/31/24, or rebalance W1–W3 toward applied?**
5. **Honesty-table policy:** should it also flag taught-but-not-implemented gaps, not just
   metric verification?
6. **W12 Track B scope:** classical-only (PCA residual stat-arb) or must it include a DL
   component? Determines whether to trim the lecture or extend the artifact.
7. **Is `mlcourse.Trainer` the universal training abstraction (so W10/W11 adopt it), or are
   RL/diffusion explicitly exempt (say why in the notes)?**
8. **Should rubrics be constrained to taught techniques, or should lectures expand to match
   the rubrics?** (W4 power iteration, W5 empirical Glorot, W12 adjoint/Neural ODE.)
