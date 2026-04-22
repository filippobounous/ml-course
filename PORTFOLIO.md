# Portfolio

A curated list of artifacts produced during the 12-week course, intended to be shared
with prospective employers and collaborators. Each artifact is a self-contained
sub-directory under `portfolio/` with:

- a clean `README.md` (problem statement, method, results, figures, reproducibility command),
- pinned code and configs,
- unit tests where meaningful,
- a short "what I learned" note.

Build a static index (Markdown → HTML) with:

```bash
make portfolio-build
```

---

## Headline artifacts (recruiter-grade)

| # | Artifact | Why it matters |
|---|---|---|
| 05 | **Micrograd-style autograd engine** — `portfolio/05_micrograd/` | Demonstrates you understand backprop from first principles. ~300 lines, unit-tested, blog-post README. Classic signal. |
| 08 | **Tiny GPT on TinyStories** — `portfolio/08_tinygpt/` | End-to-end transformer from scratch: BPE tokenizer, attention, training loop, generation samples, attention-map viz. |
| 09 | **DPO-tuned TinyLlama-1.1B + Gradio Space** — `portfolio/09_dpo_tinyllama/` | Modern LLM alignment (DPO, not PPO) on Apple Silicon via LoRA + MLX; published HF model card and live demo. |
| 12 | **Capstone** — `portfolio/12_capstone/` | Either a PINN solving Burgers' with analytical error bounds (**physics/sci-ML**) or a walk-forward stat-arb backtest with transaction-cost-aware Sharpe (**quant**). |

## Supporting artifacts

| # | Artifact | Focus |
|---|---|---|
| 02 | NumPy linear-models mini-library — `portfolio/02_numpy_linreg/` | Solid engineering fundamentals |
| 03 | Tabular benchmark (XGBoost / LightGBM) — `portfolio/03_tabular_benchmark/` | Practical classical-ML comparison |
| 04 | **PCA stat-arb** notebook — `portfolio/04_pca_statarb/` | Quant finance |
| 06 | Reusable `mlcourse.Trainer` harness — `portfolio/06_trainer/` | Research-grade reproducibility stack |
| 07 | CIFAR-10 classifier + failure analysis — `portfolio/07_vision_classifier/` | Computer vision with real critique |
| 10 | DDPM vs DDIM ablation mini-paper — `portfolio/10_ddpm/` | Diffusion from scratch |
| 11 | Custom-env PPO + tool-use agent — `portfolio/11_rl_agent/` | RL + agents, distinctive (not stock CartPole) |

## Bonus / stretch

- **Paper-reproduction ablation** — reproduce a figure from LoRA or DDPM at tiny scale with an ablation table. Lives under `portfolio/12_capstone/paper_reproduction/`.
- **End-to-end deployed demo** — one Gradio/HF Space (most naturally from artifact 09 or 10).

---

## How to present this portfolio

1. **Landing page:** this file, plus `make portfolio-build` produces a static HTML index.
2. **Recruiter pitch (1 paragraph):** pick the headline artifact closest to the role
   (e.g. artifact 09 for LLM roles, 12 for quant/physics, 08 for research).
3. **Per-artifact README** should always answer: problem, method, results, what I
   would do with more compute, what I learned.
4. **Reproducibility:** every artifact has a single command that reproduces the key
   figure or metric (`make -C portfolio/NN_* reproduce`).

## Template for an artifact README

```markdown
# <Artifact title>

## Problem
One paragraph. What is the question, who cares, why is it non-trivial.

## Method
One or two paragraphs. Key equations. Key design choices.

## Results
Figure / table. Metric with units. Baseline comparison.

## Reproduce
```bash
make reproduce
```

## What I learned
Three bullets. Honest. Include at least one surprise.

## What I'd do with more compute
Three bullets. Concrete.
```
