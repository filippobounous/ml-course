# Week 9 — Theory-problem solutions

## 1. DPO loss from the RLHF objective

Start from

$\max_\pi \mathbb{E}_{x \sim p, y \sim \pi(\cdot|x)}[r(x,y)] - \beta D_{KL}(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x)).$

Solve for the optimal $\pi^\star$ per prompt $x$ via Lagrangian / calculus of variations. The closed-form solution is

$\pi^\star(y | x) = \frac{1}{Z(x)} \pi_\text{ref}(y | x) \exp\!\left(\tfrac{1}{\beta} r(x,y)\right), \quad Z(x) = \sum_y \pi_\text{ref}(y|x) e^{r(x,y)/\beta}.$

Rearranging: $r(x, y) = \beta \log\!\left(\frac{\pi^\star(y|x)}{\pi_\text{ref}(y|x)}\right) + \beta \log Z(x)$.

Under a Bradley–Terry preference model $P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$, the $\log Z(x)$ term cancels (it depends only on $x$, not on $y$):

$r(x, y_w) - r(x, y_l) = \beta \left[\log\frac{\pi^\star(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi^\star(y_l|x)}{\pi_\text{ref}(y_l|x)}\right].$

The MLE of $\pi^\star$ given a preference dataset $\{(x, y_w, y_l)\}$ is the policy $\pi_\theta$ minimising

$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right].$

No reward model. No PPO. The only things that need to be computable are sum-log-probabilities of the chosen and rejected completions under $\pi_\theta$ and $\pi_\text{ref}$.

Matching reference: `dpo_loss` in `modules/09_llms_dpo/problems/solutions.py`; unit test in `tests/week_09/test_dpo_and_eval.py::test_dpo_loss_matches_pytorch_when_available` verifies agreement with a torch implementation.

## 2. LoRA parameter count

A linear layer $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is adapted by $W + BA$ with $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$. Trainable parameters: $r d_\text{in} + r d_\text{out} = r(d_\text{in} + d_\text{out})$.

Compare with full fine-tuning ($d_\text{in} d_\text{out}$ params per layer). At $d_\text{in} = d_\text{out} = 4096, r = 16$: LoRA = $16 \cdot 8192 = 131{,}072$ params vs full = $16{,}777{,}216$ → **0.78%**.

Summed across $L$ attention layers each with 4 projections (Q, K, V, O) at $d = 4096, r = 16$: $\sim 2.1$M trainable params for a 7B model with $L = 32$ → well under 1% of the base.

## 3. Chinchilla compute-optimal tokens

Chinchilla's empirical fit: compute-optimal pair $(N^\star, D^\star)$ satisfies $D^\star \approx 20 N^\star$ (tokens ≈ 20 × params). For a 1B-parameter budget: $D^\star \approx 2 \times 10^{10}$ tokens.

Total FLOPs $C \approx 6 N D$ (the standard rule-of-thumb for autoregressive transformers, accounting for forward + backward + optimiser at fixed precision). At $N = 1$B, $D = 20$B: $C \approx 1.2 \times 10^{20}$ FLOPs. On a single H100 at ~1e15 FLOPs/s (bf16, utilisation ~50%), that's roughly $2.4 \times 10^5$ seconds ≈ 67 GPU-hours, or a few hundred dollars of rented compute. This is why 1B-param LLM pretraining is now a weekend project for an individual.
