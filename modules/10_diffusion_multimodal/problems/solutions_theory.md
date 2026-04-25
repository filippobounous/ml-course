# Week 10 — Theory-problem solutions

## 1. DDPM variational lower bound → the simple loss

Forward $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$. Closed form $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$, $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$.

Variational bound (Ho 2020 eq. 5):

$-\log p_\theta(x_0) \le \mathbb{E}_q\!\left[D_{KL}(q(x_T|x_0) \| p(x_T)) + \sum_{t>1} D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1)\right].$

$q(x_{t-1}|x_t,x_0)$ is Gaussian with mean $\tilde\mu_t(x_t,x_0) = \tfrac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \tfrac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$. Parameterise $p_\theta$ with the same variance, learn only the mean. Reparametrise $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\varepsilon) / \sqrt{\bar\alpha_t}$; plug into $\tilde\mu_t$; the KL simplifies to (up to constants) $\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$. Dropping the scaling factors on each $t$ term gives the "simple" loss

$\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \varepsilon}\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon, t)\|^2.$

Training a noise predictor is equivalent (up to a constant that doesn't affect gradients) to optimising the VLB with uniform $t$-weighting.

## 2. DDPM ≡ denoising score matching

Tweedie's formula: if $y = x + \sigma \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$, then $\mathbb{E}[x | y] = y + \sigma^2 \nabla_y \log p(y)$.

In DDPM at time $t$: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$. So the score is

$\nabla_{x_t} \log q(x_t) = -\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}}.$

A network trained to predict $\varepsilon$ is (up to the known scale) a network trained to predict the score. Denoising score matching (Vincent 2011) and the DDPM epsilon-prediction objective are the same loss up to a deterministic rescaling of the target at each noise level. This is why "score-based" and "diffusion" papers end up with identical sampling schemes.

## 3. DDIM determinism at $\eta=0$

DDIM update: $x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0(x_t) + \sqrt{1-\bar\alpha_{t-1} - \sigma_t^2}\, \varepsilon_\theta(x_t) + \sigma_t z$ with $z \sim \mathcal{N}(0, I)$ and $\sigma_t = \eta \sqrt{(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)\,(1-\alpha_t)}$.

At $\eta = 0$: $\sigma_t = 0$ and the noise term vanishes. The update is a deterministic function of $x_t$, and the resulting chain integrates the **probability-flow ODE** associated with the score SDE (Song et al. 2021). With $\eta = 0$ the reverse chain is a numerical ODE solver, which is why it admits far fewer discretisation steps than the stochastic DDPM: the step-size–accuracy trade-off is that of an ODE integrator, not of a Langevin sampler.

## 4. CLIP InfoNCE

For a batch of $N$ image–text pairs $(x_i, y_i)$, define logits $\ell_{ij} = \langle f(x_i), g(y_j)\rangle / \tau$. Symmetric loss:

$\mathcal{L} = -\tfrac12 \mathbb{E}_i\!\left[\log\frac{e^{\ell_{ii}}}{\sum_j e^{\ell_{ij}}} + \log\frac{e^{\ell_{ii}}}{\sum_j e^{\ell_{ji}}}\right].$

Two softmaxes — one over text candidates given an image, one over image candidates given a text. Both aim to pick the diagonal element. InfoNCE is a lower bound on the MI between matched pairs (van den Oord 2018), which is the theoretical reason CLIP embeddings cluster by semantic content.

Reference: `clip_infonce_loss` in `modules/10_diffusion_multimodal/problems/solutions.py`. Unit test `tests/week_10/test_diffusion.py::test_infonce_identity_embeddings` verifies the identity-embeddings case hits a plausible loss value.
