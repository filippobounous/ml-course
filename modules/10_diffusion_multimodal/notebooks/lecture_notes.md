# Week 10 — Diffusion and multimodal models (lecture notes)

*Reading pair: Ho *DDPM* 2020 · Song *Score SDEs* 2021 · Radford *CLIP* 2021 · Rombach *LDM* 2022 · Liu *LLaVA* 2023.*

---

## 1. The forward diffusion process

Start with data $x_0 \sim p_\text{data}$. Define a discrete-time Markov chain $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)$ for $t = 1, \dots, T$, with a pre-specified variance schedule $\beta_1 < \dots < \beta_T$ (linear or cosine).

**Closed-form $q(x_t | x_0)$.** Setting $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s \le t} \alpha_s$,

$$q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I).$$

Two implications:
1. For large $T$ and a well-chosen schedule, $x_T \approx \mathcal{N}(0, I)$ — every datapoint diffuses to pure noise.
2. We can **sample any intermediate $x_t$ directly** from $x_0$ without running the chain step by step: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$.

## 2. The reverse process and the ELBO

The reverse kernels $q(x_{t-1} | x_t, x_0)$ are Gaussian with known mean and variance (Bayes rule on Gaussians). We approximate them by $p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$.

The variational ELBO decomposes into $T$ terms, each a KL between $q(x_{t-1} | x_t, x_0)$ and $p_\theta(x_{t-1} | x_t)$. Ho et al. (2020) reparameterise in terms of the noise $\varepsilon$ that was added during the forward pass, giving the shockingly simple training loss

$$\mathcal{L}_\text{simple}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\varepsilon, t)\|^2\right].$$

A UNet predicts the noise that was added; train for it, done.

## 3. DDPM ≡ denoising score matching

Tweedie's formula relates the score $\nabla_{x_t} \log q(x_t)$ to the noise:

$$\nabla_{x_t} \log q(x_t) = -\varepsilon_\theta(x_t, t) / \sqrt{1 - \bar\alpha_t}.$$

So training a noise predictor is equivalent — up to a rescaling — to training a **score model**. Denoising score matching (Vincent 2011) and DDPM (Ho 2020) arrive at the same objective from different angles.

## 4. DDIM: non-Markovian sampling

DDPM sampling takes $T$ steps (usually 1000). DDIM (Song, Meng, Ermon 2021) reinterprets the reverse process as a non-Markovian chain with a free parameter $\eta \in [0, 1]$ that interpolates between fully-stochastic ($\eta = 1$, recovers DDPM) and fully-deterministic ($\eta = 0$). The deterministic limit corresponds to integrating the **probability-flow ODE** of Song et al. (2021).

The one-liner: DDIM at $\eta = 0$ lets you use **10–50 sampling steps** instead of 1000 with minimal quality loss — the practical difference between "generates images overnight" and "generates images in a few minutes".

### DDIM update

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2} \cdot \varepsilon_\theta(x_t, t) + \sigma_t \cdot z,$$

with $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t} \varepsilon_\theta) / \sqrt{\bar\alpha_t}$, $z \sim \mathcal{N}(0, I)$, and $\sigma_t^2 = \eta^2 \cdot \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \cdot (1 - \alpha_t)$.

## 5. Continuous-time view: score SDEs

Song et al. (2021) generalise DDPM to continuous time via SDEs:

- **VP-SDE**: $dx = -\tfrac{1}{2} \beta(t) x \, dt + \sqrt{\beta(t)} \, dW$ — continuous-time analogue of DDPM.
- **VE-SDE**: $dx = \sqrt{2\sigma(t)\dot\sigma(t)} \, dW$ — "variance-exploding", alternative choice.

The reverse-time SDE uses the score $\nabla_x \log p_t(x)$; the deterministic probability-flow ODE gives yet another sampler (e.g. Heun or DPM-Solver).

The Langevin primer from Week 1 connects directly: run the Euler–Maruyama scheme backwards with a learned score and you get a DDPM sampler.

## 6. CLIP: contrastive image–text pretraining

Radford et al. (2021) learn matched image and text embeddings with the symmetric InfoNCE loss

$$\mathcal{L} = \tfrac{1}{2} \left( \mathcal{L}_\text{i→t} + \mathcal{L}_\text{t→i} \right),$$

where

$$\mathcal{L}_\text{i→t} = -\tfrac{1}{N} \sum_{i=1}^N \log \frac{\exp(f(x_i)^\top g(y_i) / \tau)}{\sum_{j} \exp(f(x_i)^\top g(y_j) / \tau)}.$$

Zero-shot classification: for each class label $c$, form the prompt "a photo of a $c$", compute its text embedding, and classify an image by nearest text embedding. Surprisingly strong on many benchmarks for free.

Use `open_clip` (LAION's open-weights implementation) for retrieval / zero-shot experiments on Apple Silicon — works out of the box on MPS for inference.

## 7. Latent diffusion and beyond

- **Latent Diffusion / Stable Diffusion** (Rombach 2022). Run diffusion in the latent space of a pretrained autoencoder rather than in pixel space; much cheaper at HD resolutions.
- **Classifier-free guidance** (Ho & Salimans 2022). At training time, randomly drop the conditioning; at sampling time, extrapolate via $\epsilon_\theta^\text{guided} = (1 + w) \epsilon_\theta^\text{cond} - w \epsilon_\theta^\text{uncond}$. The standard way to crank up conditioning fidelity.
- **EDM** (Karras 2022). Systematically redesigns the schedule / parameterisation; state of the art at small scale.
- **Flow matching** (Lipman 2023). An alternative to denoising score matching that learns a velocity field; equivalent in the continuous limit, often easier to train.

## 8. Vision-language models

- **LLaVA** (Liu 2023). Bolt a visual encoder (CLIP-ViT-L/14) onto a language-model backbone (Llama-based) via a thin projection layer. Fine-tune on synthetic instruction data.
- On Apple Silicon, run LLaVA-1.5-7B at 4-bit via `mlx-lm` or `llama.cpp` — inference only; training is out of scope.

## What to do with these notes

Work the problem set in `../problems/README.md`. Build the portfolio artifact
in `../../../portfolio/10_ddpm/`: UNet-DDPM on FashionMNIST, DDPM-vs-DDIM
ablation (step count vs sample quality), classifier-free guidance,
CLIP zero-shot retrieval demo.

---

## Time budget (≈ 20 hr)

| Block | Hours | Focus |
|---|---|---|
| §1–§2 forward / reverse | 4 | Derive α̅ recursion; derive the DDPM ELBO → simple-loss reduction with pen and paper. |
| §3 score matching | 2 | Verify Tweedie's formula; relate ε-prediction and score explicitly. |
| §4 DDIM | 2 | Derive the DDIM update; verify η=0 determinism on a toy problem. |
| §5 SDE view | 2 | Connect Week 1 Langevin to reverse-time VP-SDE. |
| §6 CFG | 2 | Derive CFG from the Bayes rule; implement in the portfolio UNet. |
| §7 multimodal | 2 | Run `open_clip` ViT-B/32 zero-shot on a small image folder. |
| Artifact | 4 | UNet + DDPM+CFG training + DDPM/DDIM ablation figure. |
| Office hours | 2 | Cross-check with `problems/solutions_theory.md`. |

## Self-assessment rubric

1. Can I state the DDPM forward process and derive $q(x_t | x_0)$ in closed form?
2. Can I derive the "simple" ε-prediction loss from the ELBO?
3. Can I explain why DDIM with η=0 is deterministic and why fewer steps suffice?
4. Can I derive the CFG extrapolation formula $\epsilon = (1+w)\epsilon_\text{cond} - w\epsilon_\text{uncond}$ from $\log p(x|y) + w(\log p(x|y) - \log p(x))$?
5. Can I connect the discrete DDPM chain to a continuous-time VP-SDE and identify its reverse?

## Physics bridge

- **Forward diffusion ≡ overdamped Langevin in reverse-temperature time.** The VP-SDE $dx = -\tfrac12\beta(t) x \, dt + \sqrt{\beta(t)} \, dW$ is the heat equation you've seen a dozen times — $\partial_t p = \nabla\cdot(x p) + \Delta p$ — viewed as a diffusion process. The stationary measure as $t \to \infty$ is the isotropic Gaussian.
- **Reverse process ≡ time-reversal with a score term.** By a classic Anderson 1982 result, a diffusion SDE has an exact reverse-time SDE that includes the **score** $\nabla_x \log p_t(x)$. Learning $\epsilon_\theta$ is learning this score; sampling from the reverse SDE with the learned score is stochastic quantisation run backwards.
- **Classifier-free guidance ≡ tempered Gibbs.** With guidance weight $w$, the effective sampled density is $p(x|y)^{1+w} / p(x)^w$, up to a constant — i.e. an "over-temperature" version of the conditional distribution. The connection to tempering in statistical mechanics is exact. This is why very large $w$ produces oversaturated, mode-seeking samples: you're sampling from a very peaked Gibbs distribution.

Diffusion is where the physics / ML bridge pays off most dramatically — every insight from stochastic mechanics has a direct generative-model counterpart.
