# Portfolio gallery

The shareable artifacts produced across W2 → W13. Each card links to
the artifact's README and notes the canonical reproduction command.
Sorted by week.

## Headline artifacts

<div class="grid cards" markdown>

- :material-graph: **W5 — Micrograd autograd engine**

    ---

    Scalar `Value` class + topological-sort backward + `Neuron / Layer / MLP`,
    trained on two-moons. ~300 lines, gradient-checked against
    `torch.autograd`. Classic "do you understand backprop" signal.

    [:octicons-arrow-right-24: open W5](portfolio/05/index.md)

- :material-text-recognition: **W8 — Tiny GPT on TinyStories**

    ---

    Multi-head attention from scratch, BPE tokenizer, ~10 M-param
    decoder-only transformer trained to loss ≲ 2.0. End-to-end
    transformer engineering in one repo.

    [:octicons-arrow-right-24: open W8](portfolio/08/index.md)

- :material-tune-vertical: **W9 — DPO-tuned TinyLlama**

    ---

    SFT + DPO on TinyLlama-1.1B-Chat with LoRA + MLX (Apple Silicon
    native). Win-rate eval harness; model card; Gradio Space.

    [:octicons-arrow-right-24: open W9](portfolio/09/index.md)

- :material-finance: **W12 — Applied capstone**

    ---

    Pick a track. **PINN**: Burgers' equation with NTK-balanced loss
    weighting, $L^2 \le 10^{-2}$ vs Cole–Hopf. **Stat-arb**:
    walk-forward PCA on Ken French industries with TCA-Sharpe.

    [:octicons-arrow-right-24: open W12](portfolio/12/index.md)

</div>

## Supporting artifacts

<div class="grid cards" markdown>

- :material-table: **W2 — NumPy linear models**

    ---

    Closed-form OLS, SGD, ridge, lasso (coordinate descent), K-fold CV.
    Matches sklearn to $10^{-9}$.

    [:octicons-arrow-right-24: open W2](portfolio/02/index.md)

- :material-chart-box: **W3 — Tabular benchmark**

    ---

    Logistic / RandomForest / XGBoost / LightGBM on UCI Adult, with
    ROC + PR + calibration + Brier decomposition.

    [:octicons-arrow-right-24: open W3](portfolio/03/index.md)

- :material-finance: **W4 — PCA stat-arb**

    ---

    Avellaneda–Lee residuals + rolling z-score on simulated returns.
    IS Sharpe ≈ 3.2, OOS Sharpe ≈ 2.9 (sim).

    [:octicons-arrow-right-24: open W4](portfolio/04/index.md)

- :material-cog: **W6 — Trainer harness**

    ---

    Reusable `mlcourse.Trainer` with grad accumulation, mixed
    precision, MPS-aware autocast, deterministic checkpoint round-trip,
    Hydra config tree.

    [:octicons-arrow-right-24: open W6](portfolio/06/index.md)

- :material-image-multiple: **W7 — CIFAR-10 classifier**

    ---

    ResNet-18 from scratch via the Week-6 Trainer + Grad-CAM +
    FGSM adversarial sweep + failure-mode analysis.

    [:octicons-arrow-right-24: open W7](portfolio/07/index.md)

- :material-image: **W10 — DDPM vs DDIM ablation**

    ---

    UNet-DDPM on FashionMNIST, classifier-free guidance, step-count
    ablation across DDPM(1000) and DDIM η=0 at {10, 20, 50, 100}.
    Reports **FID** (InceptionV3) plus a pixel-stat proxy.

    [:octicons-arrow-right-24: open W10](portfolio/10/index.md)

- :material-robot: **W11 — Custom-env PPO + tool-use agent**

    ---

    CleanRL-style PPO (with the Huang-2022 "37 details" subset) on a
    custom market-maker env, plus a torch-free ReAct agent with a
    deterministic eval harness.

    [:octicons-arrow-right-24: open W11](portfolio/11/index.md)

- :material-api: **W13 — LLM dev surface**

    ---

    LLM-as-judge wrapper + minimal MCP server demo. Plug-compatible
    with the W9 eval harness.

    [:octicons-arrow-right-24: open W13](portfolio/13/index.md)

</div>

## Bonus

<div class="grid cards" markdown>

- :material-file-document-multiple: **Paper reproduction — PPO clip ablation**

    ---

    Schulman 2017 Figure 6 at tiny scale on CartPole-v1: $\varepsilon
    \in \{0.1, 0.2, 0.3, \text{no-clip}\}$, 4 seeds per config. Uses
    the W11 PPO end-to-end.

    Lives under `portfolio/12_capstone/paper_reproduction/`.

    [:octicons-arrow-right-24: see the W12 page](portfolio/12/index.md)

</div>

## How to present this gallery

1. **Top-level pitch** — pick the headline artifact closest to the
   role (W9 for LLM roles, W12 for quant/physics, W8 for research
   engineering, W11 for RL/agents).
2. **Per-artifact README** answers: problem, method, results,
   what I'd do with more compute, what I learned.
3. **Reproducibility** — every artifact has a one-command path
   (`make reproduce` or `python demo.py`).

Reading order matches the course flow: see [`STUDY_GUIDE.md`](STUDY_GUIDE.md)
for the recommended week-by-week trajectory.
