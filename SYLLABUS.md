# Syllabus — 12-Week ML/AI Course (+ optional Week 13)

An intensive, project-driven curriculum for learners with a strong quantitative
background (theoretical physics, maths) and **tutorial-level ML priors**. Fits on
**CPU / Apple Silicon (MPS)** — no CUDA GPU required.

- **Duration:** 12 weeks, ~20 hr/week (plus an optional Week 13 on LLMs-as-dev-surface).
- **Balance:** 30% theory / 40% code / 30% applied projects
- **Outputs each week:** problem set (graded) + portfolio artifact

## Weekly rhythm (~20 hrs)

| Block | Hours | Activity |
|---|---|---|
| Readings + lecture notes | 4–6 | Paper / textbook chapters in `modules/NN_*/readings.md` |
| Problem set | 6–8 | 2 theory, 2 implementation, 1–2 applied. Test-graded where possible. |
| Portfolio artifact | 6–8 | Shippable piece in `portfolio/NN_*/`, under git with a clean README |
| Office-hours / review | 1–2 | Sync against reference solutions, log surprises |

## Capstone

A final capstone kicks off **week 8** and runs in parallel with weeks 9–12.
Learner picks **physics/scientific ML** (e.g. PINN for Burgers') or
**quantitative finance** (e.g. factor-model stat-arb with walk-forward validation),
plus a **paper-reproduction ablation** (e.g. LoRA or DDPM figure).

---

## Week 1 — Mathematical foundations

**Goal:** a unified linear-algebra / probability / information-theory / optimization
toolkit, framed for ML.

- Linear algebra as **functions**: SVD, eigendecomposition, pseudoinverse, matrix calculus
- Probability: joint/marginal/conditional, exponential families, KL / cross-entropy / MI
- Optimization: convex sets and functions, Lagrangian duality, gradient descent
  convergence rates, stochastic approximation
- **Primer:** Fokker–Planck and Langevin dynamics (sets up diffusion in W10)

**Readings:** Strang *Linear Algebra and Learning from Data* Ch.1–4 · MacKay *ITILA* Ch.2,4,8 · Boyd *Convex Optimization* Ch.2–3, 9.

**Problem set:** SVD identities · KL properties · convexity proofs · gradient-descent rate derivation.

**Portfolio:** none (notes + proofs).

## Week 2 — Statistical learning & ERM

**Goal:** ground ML in the decision-theoretic framework.

- Empirical risk minimization, Rademacher complexity, PAC learning (skim)
- MLE vs MAP vs Bayesian; bias–variance decomposition
- Linear regression: closed form, SGD, ridge/lasso, cross-validation
- **MDP primer** for W11: states, actions, rewards, Bellman operators

**Readings:** Bishop PRML Ch.1, 3 · Shalev-Shwartz & Ben-David *UML* Ch.2–5 · Murphy PML-1 Ch.4, 11.

**Problem set:** bias–variance decomposition derivation · ridge-vs-MAP equivalence · closed-form normal equations · CV bias proof.

**Portfolio artifact:** `portfolio/02_numpy_linreg/` — NumPy-only linear-models mini-library with closed-form + SGD solvers, ridge/lasso, K-fold CV, unit tests, a README with figures.

## Week 3 — Classical supervised learning

- Logistic regression, softmax, multiclass losses
- SVM: margin, kernels, RKHS intuition
- Decision trees, random forests, gradient boosting (**XGBoost, LightGBM**)

**Readings:** ESL Ch.4, 9, 10, 15 · Murphy PML-1 Ch.17 · Chen & Guestrin *XGBoost* (2016).

**Problem set:** dual SVM derivation · logistic-loss convexity · information-gain computation · XGBoost vs LightGBM empirical comparison.

**Portfolio artifact:** `portfolio/03_tabular_benchmark/` — UCI benchmark (Adult or Covertype) with logistic / RF / XGBoost / LightGBM and a calibration + ROC report.

## Week 4 — Classical unsupervised learning

- PCA / SVD as variance maximisation and as a denoising tool
- k-means and Lloyd's algorithm; GMMs and EM
- Density estimation: KDE, normalizing flows (sneak preview)

**Readings:** ESL Ch.14 · Bishop Ch.9, 12 · Avellaneda & Lee "Statistical Arbitrage in the US Equities Market" (2008).

**Problem set:** EM E-step / M-step derivation · PCA as MLE of probabilistic PCA · k-means convergence argument · implement GMM on Old Faithful.

**Portfolio artifact:** `portfolio/04_pca_statarb/` — **PCA stat-arb notebook** on Ken French industry portfolios with in-sample/out-of-sample Sharpe and a short write-up. Quant finance.

## Week 5 — Neural networks from scratch

- Feedforward networks as compositions of affine + nonlinearities
- Backpropagation as reverse-mode autodiff (derive on paper, implement in NumPy)
- Optimisers: SGD, momentum, Adam; initialisation (Glorot, He); normalisation (BN, LN)

**Readings:** Goodfellow DL Ch.6, 8 · Karpathy *micrograd* repo + *Yes you should understand backprop* post · Glorot 2010 · He 2015.

**Problem set:** derive backprop for a 2-layer MLP · prove that Glorot init preserves variance · implement Adam from the update equations.

**Portfolio artifact:** `portfolio/05_micrograd/` — **micrograd-style autograd engine** in ≤300 lines, with unit tests and a blog-post README deriving backprop from scratch. Classic recruiter signal.

## Week 6 — PyTorch deep-dive + reproducibility stack

- `nn.Module`, `DataLoader`, training loops, hooks, profiling
- **MPS perf and caveats**: dtypes on M1/M2/M3, `torch.autocast("mps")`, `torch.compile` pitfalls
- **Reproducibility stack**: seeds, Hydra configs, Weights & Biases (free tier), `torchinfo` summaries, model cards

**Readings:** PyTorch MPS docs · *The Annotated Transformer* (skim) · Lightning docs · Hydra docs.

**Problem set:** port your W5 autograd example to PyTorch · profile CPU vs MPS · write a Hydra config that sweeps learning rate · add W&B logging.

**Portfolio artifact:** `portfolio/06_trainer/` — **reusable `mlcourse.Trainer` harness** (the skeleton lives in `src/mlcourse/trainer.py` and is extended each subsequent week). Hydra configs in `src/mlcourse/configs/`.

## Week 7 — CNNs and vision

- Convolutions, receptive fields, pooling, batch norm
- ResNet, residual connections, transfer learning
- Explainability: Grad-CAM, failure-mode analysis
- **Preview:** Vision Transformer (ViT)

**Readings:** DL Ch.9 · He et al. *ResNet* (2015) · Selvaraju *Grad-CAM* (2017) · Dosovitskiy *ViT* (2020, first pass).

**Problem set:** backprop through a conv layer · count ResNet-18 parameters · compare from-scratch vs transfer learning on CIFAR-10 · adversarial FGSM example.

**Portfolio artifact:** `portfolio/07_vision_classifier/` — **CIFAR-10 classifier** from scratch (ResNet-18) + transfer-learning baseline, with Grad-CAM visualisations and a **failure-mode analysis** report. Computer vision.

## Week 8 — Transformers from scratch (+ **capstone kickoff**)

- Self-attention, multi-head attention, positional encodings
- Causal masking, teacher forcing, sampling
- BPE tokenization with `tokenizers`
- **Capstone proposal due by end of week.**

**Readings:** Vaswani *Attention Is All You Need* · Karpathy *Let's build GPT* · Radford *GPT-2* · Sennrich *BPE*.

**Problem set:** derive attention's gradient · prove softmax is invariant to shifts · implement multi-head from single-head · train a BPE tokenizer on TinyStories.

**Portfolio artifact:** `portfolio/08_tinygpt/` — **tiny GPT (~10M params)** trained from scratch on TinyStories; generation samples, attention-map visualisations, a short technical write-up.

## Week 9 — Language models at scale (SFT, DPO, scaling)

- HuggingFace ecosystem: `transformers`, `datasets`, `peft`, `trl`, `accelerate`
- Scaling laws (Kaplan, Chinchilla) demonstrated at tiny scale
- **SFT** (supervised fine-tuning) with LoRA
- **DPO** (direct preference optimization) — the modern alternative to RLHF-PPO
- Apple-native **MLX** for fast on-device tuning and inference

**Readings:** Kaplan *Scaling Laws* (2020) · Hoffmann *Chinchilla* (2022) · Ouyang *InstructGPT* (2022) · Rafailov *DPO* (2023) · Hu *LoRA* (2021).

**Problem set:** derive the DPO loss from the RLHF objective · prove LoRA's parameter-count reduction · write an MT-Bench-style eval · document a model card.

**Portfolio artifact:** `portfolio/09_dpo_tinyllama/` — **DPO-tuned TinyLlama-1.1B** (via LoRA), with a clean eval harness, a HuggingFace model card, and a **Gradio demo** (HF Space). This is a top-of-portfolio piece.

## Week 10 — Diffusion and multimodal

- Denoising score matching, DDPM, DDIM
- Continuous-time view: score SDEs, Fokker–Planck connection
- CLIP contrastive pretraining; brief VLM tour (LLaVA, 4-bit inference only)

**Readings:** Ho *DDPM* (2020) · Song *Score-based SDEs* (2021) · Radford *CLIP* (2021) · Rombach *LDM* (2022) · Liu *LLaVA* (2023).

**Problem set:** derive DDPM's variational bound · relate DDPM and score matching · compute CLIP cosine-similarity on a toy corpus · implement DDIM deterministic sampling.

**Portfolio artifact:** `portfolio/10_ddpm/` — **DDPM vs DDIM ablation** on FashionMNIST: sample quality, step-count trade-offs, reproduced figure + table, mini-paper. Multimodal companion: zero-shot CLIP retrieval demo.

## Week 11 — Reinforcement learning & agents

- MDPs, Bellman operators, value iteration
- Policy gradient, REINFORCE, actor–critic, **PPO**
- Agents from first principles: tool-use loop, termination, eval harness

**Readings:** Sutton & Barto Ch.3, 4, 6, 13 · Schulman *PPO* (2017) · Christiano *RLHF* (2017).

**Problem set:** derive the policy-gradient theorem · PPO's clipped objective geometry · GAE derivation · CartPole PPO convergence analysis.

**Portfolio artifact:** `portfolio/11_rl_agent/` — **custom environment + PPO from scratch** (e.g. simple market-maker or 1-D physics sim) plus a minimal **from-scratch tool-use agent** with an eval harness. Stands out more than stock CartPole.

## Week 12 — Applied tracks + capstone delivery

Pick **one primary track** for the capstone, touch the other.

### Track A — Physics / scientific ML
- **PINNs** for Burgers' equation with error-vs-analytical comparison
- **Neural ODEs** via `torchdiffeq` on Lotka–Volterra / spirals
- **Symbolic regression** via **PySR**

### Track B — Quantitative finance
- Factor models with DL (time-series transformer / N-BEATS-lite)
- **Walk-forward backtest** with transaction-cost-aware Sharpe
- Regime detection (HMM / GARCH, with `arch`)

**Readings:** Raissi *PINNs* (2019) · Chen *Neural ODEs* (2018) · Cranmer *PySR* (2023) · López de Prado *Advances in Financial ML* Ch.2, 3, 7.

**Problem set:** derive the PINN loss for a heat equation · implement a walk-forward split · prove adjoint-method correctness for Neural ODEs.

**Portfolio artifact:** `portfolio/12_capstone/` — **capstone project** (PINN or stat-arb), plus a **paper-reproduction ablation** (e.g. LoRA or DDPM figure) as a bonus artifact.

## Week 13 (optional) — LLMs as a development surface

Covers the piece of the brief Weeks 1–12 skipped: using frontier LLMs as an
**external tool**, not as an implementation target. Claude Code / MCP /
LLM-as-judge / cost modelling. The artifact integrates with Week 9's eval
harness (drop-in LLM judge) and Week 11's agent (drop-in MCP tools).

- Topics: agentic-coding workflows; Model Context Protocol (server anatomy, tools/resources); LLM-as-judge with position-bias mitigation; cost / latency modelling.
- Readings: Anthropic Claude Code docs · MCP spec · Zheng et al. *MT-Bench* (2024) · Dubois et al. *AlpacaEval 2* (2024).
- Portfolio: `portfolio/13_dev_surface/` — mockable LLM-judge wrapper + cost model + minimal MCP server + end-to-end demo.

---

## Grading & portfolio

- Weekly problem sets: aim for ≥80% before moving on. Auto-graded tests live in
  `tests/week_NN/`.
- Portfolio export: `make portfolio-build` renders `PORTFOLIO.md` and each
  artifact README into a static index suitable for sharing with recruiters.

## Prerequisites & expectations

- Python fluency (no prior DL framework needed).
- Comfort with university-level linear algebra, probability, calculus, ODEs.
- Willingness to read papers. The course assumes you will engage with primary sources.
