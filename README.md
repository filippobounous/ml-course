# ML/AI Course — for Quantitative Minds

A 12-week, ~20 hr/week intensive course that takes a quantitatively-literate learner
(theoretical physics, maths, or equivalent) from **tutorial-level ML** to **research-grade
fluency** in modern machine learning, deep learning, LLMs, diffusion, RL, and their
applications to physics, quantitative finance, NLP, and computer vision.

The course is designed to run on **CPU / Apple Silicon (MPS)** — every experiment fits
in under an hour on an M-series Mac. No CUDA GPU required.

Target balance: **30% theory** (derivations, proofs) / **40% hands-on code** (NumPy,
PyTorch, MLX, HuggingFace) / **30% applied projects** (physics, quant, NLP, vision).

Learners ship a **portfolio of artifacts** — from a micrograd-style autograd engine to a
DPO-tuned TinyLlama with a Gradio demo — that is recruiter-ready by week 12.

---

## Verified vs aspirational (honesty table)

The course promises concrete numbers (accuracies, Sharpes, win rates). Some of
those numbers have been **run on real hardware and committed as reference
outputs**; others are still **aspirational targets** — the code runs but the
claim hasn't been independently verified on someone else's machine. This
table is updated as each learner confirms a number.

| Artifact | Code runs end-to-end | Reported metric verified | Runtime verified |
|---|---|---|---|
| W2 NumPy linreg (`02_numpy_linreg`) | ✅ | ✅ matches sklearn to 1e-9 | ✅ < 1 min |
| W3 tabular benchmark (`03_tabular_benchmark`) | ✅ (needs OpenML) | ⏳ aspirational | ⏳ |
| W4 PCA stat-arb (`04_pca_statarb`) | ✅ | ✅ IS Sharpe ≈ 3.2 / OOS ≈ 2.9 on sim | ✅ seconds |
| W5 micrograd (`05_micrograd`) | ✅ | ✅ ≥ 88% two-moons acc (slow test) | ✅ < 1 min |
| W6 Trainer harness (`06_trainer`) | ✅ (needs torch) | ⏳ aspirational | ⏳ |
| W7 vision classifier (`07_vision_classifier`) | ✅ (needs torch) | ⏳ 90% CIFAR-10 aspirational | ⏳ 30 min MPS |
| W8 tiny GPT (`08_tinygpt`) | ✅ (needs torch + tokenizers) | ⏳ "coherent samples" aspirational | ⏳ ~6 h MPS |
| W9 DPO TinyLlama (`09_dpo_tinyllama`) | ✅ (needs HF + TRL) | ⏳ ~55–60% win-rate aspirational | ⏳ ~3 h MPS |
| W10 DDPM (`10_ddpm`) | ✅ (needs torch) | ⏳ pixel-stat distance aspirational | ⏳ ~2 h MPS |
| W11 PPO + agent (`11_rl_agent`) | ✅ (torch + gymnasium; agent torch-free) | ⏳ PPO aspirational / agent ✅ | ⏳ |
| W12 PINN + stat-arb (`12_capstone`) | ✅ (needs torch) | ⏳ $L^2 \le 10^{-2}$ aspirational | ⏳ 20 min MPS |
| W13 LLM judge + MCP (`13_dev_surface`) | ✅ | ✅ (cost-model and judge parser unit-tested) | ✅ seconds |

**Legend.** ✅ verified on a real machine; ⏳ aspirational (target stated, not yet confirmed). Confirm
a row by running the artifact's `demo.py`, filing a PR that ticks the box, and
attaching the resulting log / figure to `portfolio/<artifact>/verified.md`.

## Start here

- **Read the syllabus:** [`SYLLABUS.md`](SYLLABUS.md) — week-by-week plan, readings, problem sets, artifacts.
- **Browse the portfolio:** [`PORTFOLIO.md`](PORTFOLIO.md) — what you will ship and how to present it.
- **Known gaps:** [`TODO.md`](TODO.md) — review-surfaced issues not yet closed.
- **Install the environment:** see **Setup** below.

## Course map

| Week | Theme | Headline artifact |
|---|---|---|
| 1 | Math foundations (linalg, probability, info theory, optimization, SDE primer) | Lecture notes + proofs problem set |
| 2 | Statistical learning: ERM, MLE/MAP, bias–variance, PAC, MDP primer | NumPy linear-models mini-library |
| 3 | Classical supervised: logistic, SVM, kernels, trees, gradient boosting | Tabular benchmark + XGBoost report |
| 4 | Classical unsupervised: PCA/SVD, k-means, GMM/EM, density estimation | **PCA stat-arb notebook** (quant finance) |
| 5 | Neural nets from scratch: autograd, backprop, SGD/Adam, init/norm | **Micrograd-style autograd engine** |
| 6 | PyTorch deep-dive + reproducibility stack (Hydra, W&B, seeds, MPS) | **Reusable `mlcourse.Trainer` harness** |
| 7 | CNNs & vision: ResNet, BN, transfer learning, Grad-CAM | **CIFAR-10 classifier + failure analysis** |
| 8 | Transformers from scratch: attention, multi-head, BPE — **capstone kickoff** | **Tiny GPT on TinyStories** |
| 9 | HF ecosystem, scaling laws, SFT, DPO with LoRA + MLX | **DPO-tuned TinyLlama + Gradio Space** |
| 10 | Diffusion & multimodal: score matching, DDPM, DDIM, CLIP | **DDPM vs DDIM ablation** |
| 11 | RL + agents: Bellman, policy gradient, PPO, tool-use agents | **PPO on custom env + from-scratch agent** |
| 12 | Applied tracks (physics / quant) + **capstone delivery** | **PINN or stat-arb capstone** + paper reproduction |

See [`SYLLABUS.md`](SYLLABUS.md) for the detailed week-by-week breakdown and reading list.

## Weekly rhythm

Each week follows the same shape:

1. **Readings + lecture notes** (in `modules/NN_*/README.md` and `readings.md`)
2. **Problem set**: 2 theory (proofs/derivations) + 2 implementation + 1–2 applied.
   Graded where possible via `pytest` (see `tests/week_NN/`).
3. **Portfolio artifact** (in `portfolio/NN_*/`) — the shareable piece.

## Setup

### Option A — Local on macOS / Linux (recommended)

```bash
# 0) Python 3.11+
python3 --version

# 1) Create the venv and install dev tools
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

# 2) Register the Jupyter kernel
python -m ipykernel install --user --name mlcourse --display-name "mlcourse (.venv)"

# 3) Launch JupyterLab
jupyter lab
```

Or use the macOS bootstrap helper (runs an MPS sanity check and optional HF/MLX setup):

```bash
bash scripts/bootstrap_macos.sh
```

### Option B — Docker Jupyter

```bash
docker compose up --build jupyter
# http://localhost:8888  (token set in docker-compose.yml)
```

### Option C — Full dev Docker image (CPU torch + every extra)

Preinstalled: base + dev + dl (CPU wheels) + llm + diffusion + rl + sciml + ops.
Useful when you want every artifact runnable without fighting platform wheels.

```bash
make docker-dev              # build
make docker-dev-shell        # drop into /work
make docker-dev-test         # pytest --run-slow inside the container
```

Not for MPS — the course's MPS path is native (`scripts/bootstrap_macos.sh`).

### Installing the per-week dependency groups

Each week has its own optional-dependency group. Install them as you progress, so you
are not fighting platform-specific wheels on week 1.

```bash
make week-5       # autograd  -> installs 'dl' group (torch, torchvision, lightning)
make week-6       # trainer   -> installs 'dl,ops'
make week-7       # CNNs      -> installs 'dl,ops'
make week-8       # tiny GPT  -> installs 'dl,ops'
make week-9       # LLMs/DPO  -> installs 'dl,llm,ops'
make week-10      # diffusion -> installs 'dl,diffusion,ops'
make week-11      # RL        -> installs 'dl,rl,ops'
make week-12      # capstone  -> installs 'dl,sciml,ops'
```

Or install everything at once:

```bash
python -m pip install -e ".[all]"
```

## Common commands

```bash
make format       # ruff format + fix
make lint         # ruff + mypy
make test         # pytest (all weeks)
make test-week-N  # pytest just week N's problem set
```

## Repo layout

```
ml-course/
├── SYLLABUS.md              # week-by-week detailed schedule
├── PORTFOLIO.md             # recruiter-facing index of artifacts
├── modules/                 # course content (lecture notes, readings, problem sets)
│   ├── 01_math_foundations/
│   ├── 02_stat_learning/
│   ├── ...
│   └── 12_applied_capstone/
├── portfolio/               # per-artifact shareable repos
├── src/mlcourse/            # reusable library code (Trainer, autograd, configs, utils)
├── capstone/                # your in-progress capstone (kicks off week 8)
├── data/                    # raw / interim / processed (gitignored)
├── models/                  # saved checkpoints (gitignored)
├── reports/                 # figures, write-ups
├── tests/                   # pytest (smoke + per-week problem-set graders)
└── scripts/                 # bootstrap, dataset fetchers, CI helpers
```

## Contributing / issues

This repo is structured as a personal course workspace. For bug reports or curriculum
suggestions see `CONTRIBUTING.md`. CI runs ruff, mypy, and pytest on pushes and PRs.

## License

See `LICENSE`.
