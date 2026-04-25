# Problem set — Week 9

## Theory

1. **DPO derivation.** Starting from the RLHF objective $\max_\pi \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi \| \pi_\text{ref})$, derive the DPO loss. Give the closed-form optimal policy and show where the reward model disappears.
2. **LoRA parameter count.** For a linear layer $W \in \mathbb{R}^{d \times d}$ adapted by $W + BA$ with $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$, count parameters and compute the reduction at $d=4096, r=16$.
3. **Scaling laws.** State Chinchilla's compute-optimal relation between params, tokens, and FLOPs. Compute the optimal token count for a 1-B-param budget.

## Implementation (portfolio)

4. **SFT** TinyLlama-1.1B on an instruction dataset (Alpaca-cleaned or UltraChat-subset) with LoRA via either HF TRL on MPS or **mlx-lm** natively. Log train/val loss.
5. **DPO** your SFT checkpoint on UltraFeedback (≤5k pairs). Compare win-rate vs the SFT baseline on a held-out prompt set using an LLM-as-judge harness.
6. Publish an **HF model card** and a **Gradio Space** (guarded by `HF_LOGIN=1`).

## Applied

7. **Eval harness**: implement an MT-Bench-style rubric with 30 curated prompts; judge with an external LLM (or self-consistency if offline). Produce a comparison table.

## Grading

Tests in `tests/week_09/` verify: the DPO loss implementation matches TRL's on a fixed batch; the eval harness returns the correct schema; the Gradio demo module imports without pulling in MPS-only deps.
